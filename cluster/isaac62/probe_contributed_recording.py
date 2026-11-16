"""Run the production successful-episode writer on a marked synthetic fixture.

This checks the 499-frame recording boundary, arrays and encoded video; it does
not claim that a real expert completed a task or passed the physical filter.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from dreamvla_drawer_button.contracts import scene, validate_episode
    from dreamvla_drawer_button.convert import FIELDS
    from dreamvla_drawer_button.io import write_json
    from dreamvla_drawer_button.recording import write_episode_media
    from dreamvla_drawer_button.paths import ffmpeg_path

    class Frames:
        def __len__(self):
            return 499

        def __getitem__(self, i):
            if not 0 <= i < len(self):
                raise IndexError(i)
            return np.full((480, 640, 3), i * 255 // 498, np.uint8)

    arrays = {k: np.zeros((499, width), np.float64) for k, width in FIELDS.items()}
    arrays['diagnostic.control_step'] = np.arange(499)[:, None]
    arrays['observation.state'][:] = np.arange(499)[:, None] / 1000
    arrays['observation.projected_gravity'][:, 2] = -1
    for key in ['observation.root_orientation', 'observation.cpp_rotation_offset',
                'observation.init_base_quat']:
        arrays[key][:, 0] = 1
    episode = args.output/'episode_000000'
    episode.mkdir()
    started = time.time()
    write_episode_media(episode, arrays, Frames())
    write_json(episode/'metadata.json', dict(
        accepted=True, evaluation_only=False, synthetic_fixture=True,
        task='open the drawer', frames=499, fps=50,
        checkpoint_sha256='synthetic-recording-fixture-only',
        environment_contract=scene('drawer')), durable=True)
    validate_episode(episode, 'drawer',
                     checkpoint_sha256='synthetic-recording-fixture-only')
    with np.load(episode/'data.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name in arrays:
            np.testing.assert_array_equal(saved[name], arrays[name])
    decoder = subprocess.Popen(
        [ffmpeg_path(), '-nostdin', '-v', 'error', '-threads', '1', '-i', str(episode/'ego_view.mp4'),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    means = []
    try:
        while True:
            raw = decoder.stdout.read(480*640*3)
            if not raw:
                break
            assert len(raw) == 480*640*3
            rgb = np.frombuffer(raw, np.uint8).reshape(480, 640, 3)
            assert rgb.shape == (480, 640, 3)
            means.append(float(rgb.mean()))
        assert decoder.wait(timeout=30) == 0, decoder.stderr.read().decode()
    finally:
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait(timeout=10)
        decoder.stdout.close()
        decoder.stderr.close()
    assert len(means) == 499 and means[0] < 1 and means[-1] > 254
    write_json(args.output/'receipt.json', dict(
        completed=True, scientific_demonstration=False, synthetic_fixture=True,
        writer='dreamvla_drawer_button.recording.write_episode_media',
        frames=499, fps=50, duration_s=499/50, width=640, height=480,
        exact_array_keys=sorted(arrays), started=started, finished=time.time(),
        limitation='Synthetic successful-record fixture; physical acceptance not tested by this probe'), durable=True)


if __name__ == '__main__':
    main()
