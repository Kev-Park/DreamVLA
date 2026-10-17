"""Read-only exact observation/label audit; run on one cluster CPU, no GPU."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert len(os.sched_getaffinity(0)) == 1, 'Keep this audit on one CPU'
    started = time.time()
    root = Path.home() / 'kevin'
    dataset = root / 'datasets/af60v8f_base/lerobot/ds'
    info = json.loads((dataset / 'meta/info.json').read_text())
    assert (info['total_episodes'], info['total_frames']) == (342, 170658)
    tasks = {row['task_index']: row['task'] for row in
             map(json.loads, (dataset / 'meta/tasks.jsonl').read_text().splitlines())}
    seen = {}
    duplicates = {}
    labels = {}
    sources = []
    windows = 0
    for episode in range(342):
        file = dataset / info['data_path'].format(episode_chunk=episode // 1000, episode_index=episode)
        columns = ['observation.state', 'observation.projected_gravity', 'action.motion_token',
                   'task_index', 'frame_index', 'episode_index']
        table = pq.read_table(file, columns=columns, use_threads=False)
        state = np.concatenate([np.asarray(table[name].to_pylist(), dtype=np.float32)
                                for name in columns[:2]], axis=1)
        token = np.asarray(table['action.motion_token'].to_pylist(), dtype=np.float32)
        assert state.shape == (499, 46) and token.shape == (499, 64)
        assert np.isfinite(state).all() and np.isfinite(token).all()
        assert table['frame_index'].to_pylist() == list(range(499))
        assert set(table['episode_index'].to_pylist()) == {episode}
        state[state == 0] = 0  # canonicalize signed zero
        token[token == 0] = 0
        task_ids = table['task_index'].to_pylist()
        for frame in range(len(state) - 39):
            key = (state[frame].tobytes(), tasks[task_ids[frame]])
            label = token[frame].tobytes()
            item = (episode, frame)
            if key in seen:
                if key not in duplicates:
                    duplicates[key] = [seen[key][0]]
                    labels[key] = {seen[key][1]}
                duplicates[key].append(item)
                labels[key].add(label)
            else:
                seen[key] = (item, label)
            windows += 1
        sources.append(dict(episode=episode, sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    divergent = [key for key in duplicates if len(labels[key]) > 1]
    cross_episode = [key for key in divergent if len({p[0] for p in duplicates[key]}) > 1]
    # Check at most 64 representative pairs; report this bound explicitly.
    pairs = []
    cache = {}

    def image(episode, frame):
        key = (episode, frame)
        if key not in cache:
            video = dataset / info['video_path'].format(episode_chunk=episode // 1000,
                episode_index=episode, video_key='observation.images.ego_view')
            command = ['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(video),
                       '-vf', f'select=eq(n\\,{frame})', '-frames:v', '1', '-threads', '1',
                       '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
            p = subprocess.run(command, capture_output=True, timeout=30)
            assert p.returncode == 0, p.stderr.decode(errors='replace')
            assert len(p.stdout) == 480 * 640 * 3
            cache[key] = hashlib.sha256(p.stdout).hexdigest()
        return cache[key]

    for key in cross_episode[:64]:
        first = duplicates[key][0]
        second = next(item for item in duplicates[key] if item[0] != first[0])
        # Pair labels explicitly; a multi-label group can still contain same-label pairs.
        def token_at(item):
            ep, fr = item
            f = dataset / info['data_path'].format(episode_chunk=ep // 1000, episode_index=ep)
            return np.asarray(pq.read_table(f, columns=['action.motion_token'], use_threads=False)
                              ['action.motion_token'][fr].as_py(), dtype=np.float32)
        a, b = token_at(first), token_at(second)
        pairs.append(dict(first=first, second=second, recorded_proprio_task_exact=True,
                          decoded_rgb_exact=image(*first) == image(*second),
                          motion_token_differs=bool(np.any(a != b)),
                          motion_token_max_abs_difference=float(np.max(np.abs(a - b)))))
    result = dict(time=time.time(), elapsed_seconds=time.time() - started,
        dataset='~/kevin/datasets/af60v8f_base/lerobot/ds', episodes=342,
        raw_frames=170658, valid_windows=windows, action_window=40,
        observation='Current 43 joint values plus projected gravity, cast float32 as in the loader; exact task text.',
        unique_proprio_task_keys=len(seen), duplicate_proprio_task_groups=len(duplicates),
        divergent_immediate_token_groups=len(divergent), cross_episode_divergent_groups=len(cross_episode),
        image_pair_limit=64, image_pairs_checked=len(pairs), pairs=pairs,
        full_recorded_observation_pairs_with_different_token=sum(
            p['decoded_rgb_exact'] and p['motion_token_differs'] for p in pairs),
        source_sha256=sources, no_gpu=True, single_cpu=True, data_modified=False,
        limitations=['Exact equality audit only; does not test near matches, BF16 rounding, or perceptual similarity.',
                     'Immediate raw motion-token labels only; no action-chunk or normalized-target equivalence claim.',
                     'Camera frames are decoded RGB before augmentation, not the final randomized training tensors.',
                     'A negative result does not rule out partial observability. A positive result alone does not explain success rates.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_sha256', 'pairs']}), flush=True)


if __name__ == '__main__':
    main()
