"""Fresh-process contributed GR00T service gate using an exported model.

Exercises the actual loopback protocol, model loader, reset, inference, and
shutdown. Uses the converted synthetic fixture; makes no task-success claim.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5781)
    args = parser.parse_args()
    assert os.name == 'posix' and os.environ.get('CUDA_VISIBLE_DEVICES', '').isdecimal()
    import cv2
    import numpy as np
    import pandas as pd
    from dreamvla_drawer_button.bridge import BridgeClient

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, scope=__doc__)
    process = None
    client = None
    try:
        # Check the port before allocating a service; never contact another server.
        with socket.socket() as check:
            check.bind(('127.0.0.1', args.port))
        parquet = list(args.dataset.glob('data/**/*.parquet'))
        videos = list(args.dataset.glob('videos/**/*.mp4'))
        assert len(parquet) == len(videos) == 1
        row = pd.read_parquet(parquet[0]).iloc[20]
        joints = np.asarray(row['observation.state'], dtype=np.float32)
        assert joints.shape == (43,)
        capture = cv2.VideoCapture(str(videos[0]))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 20)
            ok, frame = capture.read()
            assert ok
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        assert image.shape == (480, 640, 3) and image.max() > 0
        request = dict(kind='infer', prompt='open the drawer', ego_view=image,
                       projected_gravity=np.asarray(row['observation.projected_gravity'], dtype=np.float32))
        for name, start, stop in [('left_leg', 0, 6), ('right_leg', 6, 12),
                                 ('waist', 12, 15), ('left_arm', 15, 22),
                                 ('left_hand', 22, 29), ('right_arm', 29, 36),
                                 ('right_hand', 36, 43)]:
            request[name] = joints[start:stop]
        command = [sys.executable, '-m', 'dreamvla_drawer_button.serve',
                   '--model-path', str(args.model), '--device', 'cuda:0',
                   '--port', str(args.port), '--seed', '17', '--denoising-steps', '4']
        report['command'] = command
        with (args.output/'server.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            report['server_pid'] = process.pid
            deadline = time.monotonic()+420
            while time.monotonic() < deadline:
                assert process.poll() is None, 'Service exited before ready; inspect server.log'
                if '"status": "ready"' in (args.output/'server.log').read_text():
                    break
                time.sleep(1)
            else:
                raise TimeoutError('Service readiness deadline exceeded')
            client = BridgeClient(args.port, timeout=300)
            report['policy_sha256'] = client.policy_sha256
            assert len(client.policy_sha256) == 64
            outputs = []
            for seed in (17, 19):
                client.reset(seed)
                result = client.call(request)
                stats = {}
                for name, width in [('motion_token', 64), ('left_hand_joints', 7), ('right_hand_joints', 7)]:
                    value = result[name]
                    assert value.shape == (40, width) and np.isfinite(value).all()
                    stats[name] = dict(shape=list(value.shape), minimum=float(value.min()), maximum=float(value.max()))
                np.savez(args.output/f'prediction_seed{seed}.npz', **result)
                outputs.append(dict(seed=seed, seconds=float(result['inference_seconds']), arrays=stats))
            client.call({'kind': 'shutdown'})
            client.close()
            client = None
            assert process.wait(timeout=45) == 0
        report.update(completed=True, predictions=outputs, server_exit=0,
                      action_width=78, horizon=40, synthetic_observation=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if client is not None:
            client.close()
        if process is not None and process.poll() is None:
            # Only our own fresh child, after a failed bounded fixture.
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
