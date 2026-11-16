"""Exercise the contributed production converter and real SONIC exporter on CPU.

The input is explicitly synthetic, not an expert or task-success demonstration.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    from dreamvla_drawer_button.contracts import scene
    from dreamvla_drawer_button.convert import FIELDS, main as convert
    from dreamvla_drawer_button.io import digest, read, write_json
    from dreamvla_drawer_button.video import VideoWriter
    raw = args.output/'raw'
    episode = raw/'episode_000000'; episode.mkdir(parents=True)
    frames = 50
    arrays = {key: np.zeros((frames, width), np.float64) for key,width in FIELDS.items()}
    for key in ('observation.root_orientation', 'observation.cpp_rotation_offset',
                'observation.init_base_quat'):
        arrays[key][:,0] = 1
    arrays['observation.projected_gravity'][:,2] = -1
    arrays['diagnostic.control_step'] = np.arange(frames)[:,None]
    np.savez(episode/'data.npz', **arrays)
    write_json(episode/'metadata.json', dict(accepted=True, evaluation_only=False,
               frames=frames, fps=50, task='open the drawer',
               environment_contract=scene('drawer'), synthetic_fixture=True))
    writer = VideoWriter(episode/'ego_view.mp4', 640, 480, 50)
    try:
        for i in range(frames):
            writer.add_frame(np.full((480,640,3), i*4, np.uint8))
    finally:
        writer.stop()
    original = {p.name:digest(p) for p in episode.iterdir()}
    output = args.output/'lerobot'
    with patch.object(sys, 'argv', ['convert', str(raw), str(output)]):
        convert()
    info = read(output/'meta/info.json')
    assert info['total_episodes'] == 1 and info['total_frames'] == frames
    assert info['script_config']['raw_episode_count'] == 1
    import pyarrow.parquet as pq
    parquet = list((output/'data').rglob('*.parquet')); assert len(parquet) == 1
    table = pq.read_table(parquet[0])
    for name in FIELDS:
        np.testing.assert_array_equal(np.asarray(table[name].to_pylist()), arrays[name])
    for name in ('teleop.root_pos_w', 'teleop.object_pos_w', 'teleop.object_quat_w'):
        assert name not in table.column_names and name not in info['features']
    videos = list((output/'videos').rglob('*.mp4')); assert len(videos) == 1
    capture = cv2.VideoCapture(str(videos[0])); decoded = []
    try:
        while True:
            ok, rgb = capture.read()
            if not ok:
                break
            decoded.append(float(rgb.mean()))
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    assert len(decoded) == frames and decoded[-1] > decoded[0]+150
    assert abs(fps-50) < 0.01
    before_resume = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    with patch.object(sys, 'argv', ['convert', str(raw), str(output), '--resume']):
        convert()
    assert before_resume == {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    assert original == {p.name:digest(p) for p in episode.iterdir()}
    report = dict(completed=True, synthetic_fixture=True, episodes=1, frames=frames,
                  decoded_frames=len(decoded), fps=fps, resume_files_unchanged=True,
                  all_recorded_policy_arrays_exact=True, invented_world_pose_fields=False,
                  source_hashes=original, scope='Actual contributed converter and SONIC exporter; no simulation or expert success')
    write_json(args.output/'receipt.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
