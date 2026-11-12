"""Bounded GR00T or existing direct-expert Isaac 6.2/SONIC migration gate.

Uses the existing model interpreter in a separate process and the new simulator
interpreter for the actual evaluator. No study performance claim is made.
"""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    if os.name != 'posix' or importlib.metadata.version('isaacsim') != '6.2.0.0':
        raise SystemExit('Requires the isolated Linux Isaac Sim 6.2 environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise SystemExit('Expose only the reviewed physical GPU 0')
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--checkpoint', type=Path)
    modes.add_argument('--expert-checkpoint', type=Path)
    parser.add_argument('--model-python', type=Path)
    for key in ('sonic-pt', 'ref-motions-path', 'asset-cwd', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    for key in ('checkpoint', 'expert_checkpoint', 'sonic_pt', 'ref_motions_path', 'asset_cwd'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve(strict=True))
    # A venv launcher is commonly a symlink to its base Python. Resolving that
    # symlink discards pyvenv.cfg discovery and launches the wrong environment.
    if args.checkpoint:
        if args.model_python is None:
            parser.error('--checkpoint requires --model-python')
        args.model_python = args.model_python.expanduser().absolute()
        if not args.model_python.is_file():
            parser.error('Model interpreter does not exist')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_output = output/'service'
    model_output.mkdir()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'Training/scripts/reinforcement_learning/rsl_rl'))
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    worker = client = None
    reserved = False
    receipt = dict(started=time.time(), completed=False, checkpoint=str(args.checkpoint or args.expert_checkpoint),
                   policy_path='gr00t' if args.checkpoint else 'direct_expert_snap',
                   model_python=str(args.model_python), simulator_python=sys.executable,
                   scientific_evaluation=False, maximum_steps=32,
                   inherited_hs_settings={k: v for k, v in os.environ.items() if k.startswith('HS_')})
    try:
        r = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                            '--mb', '24000', '--gpu', '0'], capture_output=True,
                           text=True, check=True, timeout=20)
        if r.stdout.strip() != '0':
            raise RuntimeError('Reviewed GPU reservation unavailable')
        reserved = True
        if args.checkpoint:
            with (output/'service.log').open('w') as stream:
                worker = subprocess.Popen([str(args.model_python), str(root/'cluster/isaac62/probe_gr00t_service.py'),
                                           '--checkpoint', str(args.checkpoint), '--output', str(model_output),
                                           '--physical-gpu', '0', '--worker'], stdout=stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+300
            while not (model_output/'ready.json').is_file():
                if worker.poll() is not None:
                    raise RuntimeError('Model worker failed; inspect service.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Model service readiness exceeded 300 seconds')
                time.sleep(.5)
        command = [sys.executable, str(root/'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'),
                   '--headless', '--device', 'cuda:0', '--num-envs', '1', '--num-episodes', '1',
                   '--max-steps-per-episode', '32', '--seed', '17', '--waist-dof', '29',
                   '--skip-start-frames', '0', '--task', 'Isaac-Motion-Tracking-Pick-Cam-HOI-v0',
                   '--sonic-pt', str(args.sonic_pt), '--ref-motions-path', str(args.ref_motions_path),
                   '--language', 'Walk to the object and pick it up.',
                   '--traj-dump', str(output/'trace'), '--grasp-dump', str(output/'grasp.npz')]
        if args.checkpoint:
            command += ['--vla-checkpoint', str(args.checkpoint), '--gr00t-service', str(model_output/'ready.json')]
        else:
            command += ['--expert-checkpoint', str(args.expert_checkpoint), '--expert-token-mode', 'snap']
        receipt['command'] = command
        with (output/'eval.log').open('w') as stream:
            r = subprocess.run(command, cwd=args.asset_cwd, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=300)
        receipt['returncode'] = r.returncode
        if r.returncode:
            raise RuntimeError('Actual VLA evaluator failed; inspect eval.log')
        import numpy as np
        traces = list(output.glob('trace_ep0_m0_traj.npz'))
        if len(traces) != 1 or not (output/'grasp.npz').is_file():
            raise RuntimeError('Real evaluator did not save trajectory and predicted-hand records')
        with np.load(traces[0], allow_pickle=False) as trace:
            frames = len(trace['root_pos'])
            if frames < 1 or frames > 32:
                raise RuntimeError('Invalid bounded trajectory length')
            for key in trace.files:
                if key != 'body_names' and not np.isfinite(trace[key]).all():
                    raise RuntimeError('Nonfinite real evaluator state: '+key)
            receipt['trace_frames'] = frames
            receipt['body_names'] = trace['body_names'].tolist()
            requested = trace['requested_action'].copy()
            body, reference = trace['body_pos'], trace['ref_body_pos']
            receipt['trace_mpjpe_cm'] = {
                'world': float(100 * np.linalg.norm(body-reference, axis=-1).mean()),
                'root_relative': float(100 * np.linalg.norm(
                    (body-trace['root_pos'][:, None]) -
                    (reference-trace['ref_root_pos'][:, None]), axis=-1).mean()),
                'frames': frames,
            }
        with np.load(output/'grasp.npz', allow_pickle=False) as grasp:
            predicted = grasp['ep0_pred_meanq']
            if not len(predicted) or not np.isfinite(predicted).all():
                raise RuntimeError('Missing/invalid predicted finger commands')
            if args.expert_checkpoint:
                # pred_meanq is a placeholder in this existing evaluation mode;
                # validate the actual scalar commands, not that placeholder.
                close = grasp['ep0_cmd_close']
                if len(close) < frames or not np.array_equal(close[:frames], requested[:, 64] < 0):
                    raise RuntimeError('Direct actor finger command differs from saved action')
                receipt['actor_finger_frames'] = frames
                receipt['predicted_hand_placeholder'] = True
            else:
                receipt['predicted_hand_frames'] = len(predicted)

        # Exercise the production video entry point as well as the statistical
        # evaluator. Keep its trace separate from the evaluator's evidence.
        render_command = command.copy()
        render_command[1] = str(root/'Training/scripts/reinforcement_learning/rsl_rl/play_vla_sonic.py')
        index = render_command.index('--grasp-dump')
        del render_command[index:index+2]
        render_command[render_command.index('--traj-dump')+1] = str(output/'render_trace')
        render_command += ['--record-video', str(output/'policy')]
        receipt['render_command'] = render_command
        with (output/'render.log').open('w') as stream:
            r = subprocess.run(render_command, cwd=args.asset_cwd, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=300)
        if r.returncode:
            raise RuntimeError('Actual VLA video entry point failed; inspect render.log')
        with np.load(output/'render_trace_ep0_m0_traj.npz', allow_pickle=False) as trace:
            rendered_frames = len(trace['root_pos'])
            if not 1 <= rendered_frames <= 32:
                raise RuntimeError('Invalid bounded render trajectory length')
        ffprobe, ffmpeg = shutil.which('ffprobe'), shutil.which('ffmpeg')
        if not ffprobe or not ffmpeg:
            raise RuntimeError('ffprobe/ffmpeg required for real video validation')
        videos = []
        for view in ('third_person', 'ego'):
            path = output/f'policy_ep0_m0_{view}.mp4'
            r = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
                                '-of', 'json', str(path)], capture_output=True, text=True, check=True, timeout=30)
            video = json.loads(r.stdout)['streams'][0]
            if int(video['nb_read_frames']) != rendered_frames or min(int(video['width']), int(video['height'])) < 16:
                raise RuntimeError('Invalid real VLA video dimensions/frame count')
            frame = output/f'policy_{view}.png'
            subprocess.run([ffmpeg, '-nostdin', '-loglevel', 'error', '-i', str(path),
                            '-vf', f'select=eq(n\\,{min(15, rendered_frames-1)})', '-frames:v', '1', str(frame)], check=True, timeout=30)
            from PIL import Image
            pixels = np.asarray(Image.open(frame).convert('RGB'))
            if pixels.max() == 0 or pixels.std() == 0:
                raise RuntimeError('Black/constant real VLA camera frame')
            videos.append(dict(file=path.name, ffprobe=video, frame=frame.name,
                               frame_mean=float(pixels.mean()), frame_std=float(pixels.std())))
        receipt['videos'] = videos
        receipt.update(completed=True, finished=time.time())
    except BaseException:
        receipt.update(error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        if worker is not None:
            if worker.poll() is None and (model_output/'ready.json').is_file():
                with contextlib.suppress(Exception):
                    from vla_sonic.gr00t_service import PolicyServiceClient
                    client = PolicyServiceClient(model_output/'ready.json', args.checkpoint,
                                                 'cuda:0', 'unitree_g1_sonic', timeout_ms=10000)
                    client.shutdown_server()
            if client is not None:
                client.close()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=10)
        try:
            if reserved:
                subprocess.run([*slots, 'release', '--owner', owner, '--gpu', '0'], check=True, timeout=20)
                receipt['reservation_released'] = True
        finally:
            (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
