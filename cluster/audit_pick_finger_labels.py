"""Measure finger-label semantics in an existing LeRobot dataset; no inference."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    root = Path.home() / 'kevin'
    dataset = root / spec['dataset']
    info = json.loads((dataset / 'meta/info.json').read_text())
    modality = json.loads((dataset / 'meta/modality.json').read_text())
    hand = modality['state']['right_hand']
    state_column = hand.get('original_key', 'observation.state')
    target_column = modality['action']['right_hand_joints']['original_key']
    bounds = spec['cumulative_episode_counts']
    assert bounds == [342, 542, 711, 892]
    assert info['total_episodes'] == bounds[-1]
    assert info['fps'] == 50
    offsets = [0, 1, 7, 39]
    totals, episodes = {}, []
    for index in range(info['total_episodes']):
        path = dataset / info['data_path'].format(
            episode_chunk=index // info['chunks_size'], episode_index=index)
        data = pd.read_parquet(path, columns=[state_column, target_column])
        state = np.stack(data[state_column].to_numpy())[:, hand['start']:hand['end']].astype(np.float32)
        target = np.stack(data[target_column].to_numpy()).astype(np.float32)
        assert state.shape == target.shape and state.shape[1] == 7
        assert np.isfinite(state).all() and np.isfinite(target).all()
        stage = next(i for i, bound in enumerate(bounds) if index < bound)
        row = dict(episode=index, stage=stage, frames=len(state), offsets={})
        for offset in offsets:
            if offset >= len(state):
                continue
            current = state[:len(state) - offset] if offset else state
            label = target[offset:]
            error = np.abs(current.astype(np.float64) - label.astype(np.float64))
            state_closed = np.abs(current).mean(axis=1) > 0.6
            target_closed = np.abs(label).mean(axis=1) > 0.6
            values = dict(frames=len(current), scalar_values=error.size,
                exact_copy_frames=int(np.all(current == label, axis=1).sum()),
                near_copy_frames=int(np.all(error <= 1e-7, axis=1).sum()),
                absolute_error_sum=float(error.sum()),
                observed_closed_frames=int(state_closed.sum()),
                target_closed_frames=int(target_closed.sum()),
                close_target_while_observed_open=int((target_closed & ~state_closed).sum()),
                open_target_while_observed_closed=int((~target_closed & state_closed).sum()))
            row['offsets'][str(offset)] = values
            for group in [f'added_stage_{stage}', 'cumulative_all',
                          'base' if stage == 0 else 'dagger']:
                total = totals.setdefault(group, {}).setdefault(str(offset), {key:0 for key in values})
                for key, value in values.items():
                    total[key] += value
        episodes.append(row)
    assert totals['cumulative_all']['0']['frames'] == info['total_frames'] == 443502
    for group in totals.values():
        for values in group.values():
            values['exact_copy_fraction'] = values['exact_copy_frames'] / values['frames']
            values['mean_absolute_error_rad'] = values['absolute_error_sum'] / values['scalar_values']
    result = dict(time=time.time(),dataset=str(dataset),episodes=episodes,totals=totals,
        offsets_frames=offsets, fps=info['fps'], closure_threshold=0.6,
        limitations=[
            'This compares recorded targets with observed joint positions; it does not evaluate model predictions or task success.',
            'Offset zero matching does not imply all action-chunk targets copy the current observation; offsets 1, 7 and 39 are reported separately.',
            'Stages use the historical cumulative episode ordering, with boundaries 342/542/711/892.',
            'The 0.6 threshold reproduces the existing evaluation conversion; near-threshold observed positions are not independently judged as physical closure.'])
    output = root / spec['output']
    assert not output.exists(), 'Preserve previous audit output'
    output.write_text(json.dumps(result,indent=2) + '\n')
    print(json.dumps(dict(output=str(output),totals=totals)),flush=True)


if __name__ == '__main__':
    main()
