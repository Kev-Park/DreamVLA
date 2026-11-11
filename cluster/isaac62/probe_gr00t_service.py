"""Real GR00T CUDA/service parity on fixed synthetic inputs, without Isaac.

Run in the existing model environment on one manually reviewed GPU. The parent
owns a normal startup reservation and always closes its dedicated worker.
This does not validate the new simulator runtime or task performance.
"""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Training/scripts/reinforcement_learning/rsl_rl'))


def worker(args):
    import msgpack_numpy as mnp
    import numpy as np
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from vla_sonic.gr00t_service import PolicyService, capture_rng, publish_descriptor

    torch.set_num_threads(1)
    torch.cuda.set_device('cuda:0')
    policy = Gr00tPolicy(embodiment_tag='unitree_g1_sonic',
                         model_path=str(args.checkpoint), device='cuda:0')
    stats = json.loads((args.checkpoint/'statistics.json').read_text())['unitree_g1_sonic']['state']
    config = policy.modality_configs
    image = np.arange(256*256*3, dtype=np.int64).reshape(256, 256, 3).astype(np.uint8)
    observation = {
        'video': {key: np.repeat(image[None, None], len(config['video'].delta_indices), axis=1)
                  for key in config['video'].modality_keys},
        'state': {key: np.repeat(np.asarray(stats[key]['mean'], np.float32)[None, None],
                                 len(config['state'].delta_indices), axis=1)
                  for key in config['state'].modality_keys},
        'language': {policy.language_key: [['Walk to the object and pick it up.']]},
    }
    service = PolicyService(policy, args.checkpoint, 'cuda:0', 'unitree_g1_sonic')
    service._enable_features()
    policy.get_action(observation)  # Warm kernels and caches before matched calls.
    policy.reset()
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    initial = capture_rng('cuda:0')
    action, info = policy.get_action(observation)
    expected = dict(observation=observation, initial=initial, action=action, info=info,
                    final=capture_rng('cuda:0'), feature=service._feature,
                    worker_peak_memory=torch.cuda.max_memory_allocated())
    (args.output/'baseline.msgpack').write_bytes(mnp.packb(expected))
    policy.reset()
    service.serve(lambda descriptor: publish_descriptor(args.output/'ready.json', descriptor))


def run(args):
    import msgpack_numpy as mnp
    import numpy as np
    import torch
    from vla_sonic.gr00t_service import PolicyServiceClient, capture_rng, restore_rng

    torch.set_num_threads(1)
    torch.cuda.set_device('cuda:0')
    child = None
    client = None
    with (args.output/'worker.log').open('w') as log:
        try:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      '--checkpoint', str(args.checkpoint), '--output', str(args.output),
                                      '--physical-gpu', str(args.physical_gpu), '--worker'],
                                     stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+300
            while not (args.output/'ready.json').is_file():
                if child.poll() is not None:
                    raise RuntimeError('Model worker exited; inspect worker.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Model worker did not become ready within 300 seconds')
                time.sleep(.5)
            baseline = mnp.unpackb((args.output/'baseline.msgpack').read_bytes(), raw=False)
            client = PolicyServiceClient(args.output/'ready.json', args.checkpoint,
                                         'cuda:0', 'unitree_g1_sonic', timeout_ms=60000)
            client.enable_backbone_features()
            restore_rng(baseline['initial'], 'cuda:0')
            action, info = client.get_action(baseline['observation'])
            final = capture_rng('cuda:0')
            assert set(action) == set(baseline['action'])
            differences = {}
            for key, value in action.items():
                assert value.dtype == np.float32 and np.isfinite(value).all() and value.flags.writeable
                np.testing.assert_allclose(value, baseline['action'][key], rtol=1e-5, atol=1e-6)
                differences[key] = dict(shape=list(value.shape),
                                        max_absolute_difference=float(np.max(np.abs(value-baseline['action'][key]))))
            assert info == baseline['info']
            assert baseline['feature'] is not None and client.last_backbone_feature is not None
            np.testing.assert_array_equal(client.last_backbone_feature, baseline['feature'])
            assert mnp.packb(final) == mnp.packb(baseline['final']), 'RNG advancement differs'
            report = dict(completed=True, checkpoint=str(args.checkpoint), physical_gpu=args.physical_gpu,
                          action_differences=differences, backbone_features_identical=True,
                          all_rng_advancement_identical=True, worker_peak_memory=baseline['worker_peak_memory'],
                          packages={name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'transformers']},
                          scope='Real checkpoint and CUDA loopback; existing model environment, synthetic observations; no simulator/task claim')
            (args.output/'report.json').write_text(json.dumps(report, indent=2))
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    client.shutdown_server()
                with contextlib.suppress(Exception):
                    client.close()
            if child is not None:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        parser.error('Expose exactly the reviewed physical GPU through CUDA_VISIBLE_DEVICES')
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    slots = [sys.executable, str(ROOT/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    reserved = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                               '--mb', '24000', '--gpu', str(args.physical_gpu)],
                              capture_output=True, text=True, check=True, timeout=20)
    if reserved.stdout.strip() != str(args.physical_gpu):
        raise RuntimeError('Reviewed GPU could not be reserved; no model launched')
    try:
        run(args)
    finally:
        subprocess.run([*slots, 'release', '--owner', owner, '--gpu', str(args.physical_gpu)],
                       check=True, timeout=20)


if __name__ == '__main__':
    main()
