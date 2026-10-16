"""Read-only provenance audit for the historical and fixed-budget Pick datasets.

Run on a cluster in dreamcontrol_51, on one CPU with BLAS threads limited.
Reads HDF5 metadata/array shapes and eight calibration episodes per candidate
demo root. Does not load images, score behavior, modify data, or allocate a GPU.
"""
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

ROOT = Path.home() / 'kevin'
TAG = 'dagger57_20261013'


def read_episode(path):
    with h5py.File(path, 'r', locking=False) as handle:
        metadata = json.loads(handle.attrs['metadata_json'])
        group = handle['data/demo_0']
        frames = int(group['actions'].shape[0])
    dagger = metadata.get('dagger') or {}
    rescue = metadata.get('rescue') or {}
    return dict(source=str(path), frames=frames, metadata=metadata,
                student_steps=dagger.get('vla_steps'),
                expert_steps=dagger.get('expert_steps'),
                rescue_pass=rescue.get('pass'),
                step_count_matches_metadata=metadata.get('num_steps') == frames,
                drive_counts_match_frames=(
                    dagger.get('vla_steps', 0) + dagger.get('expert_steps', 0) == frames
                    if dagger else None))


def summarize(records):
    frames = sum(row['frames'] for row in records)
    with_counts = [r for r in records if r['student_steps'] is not None
                   and r['expert_steps'] is not None]
    student = sum(r['student_steps'] for r in with_counts)
    expert = sum(r['expert_steps'] for r in with_counts)
    passes = {label: sum(r['rescue_pass'] == label for r in records)
              for label in ('A', 'B', None)}
    return dict(episodes=len(records), frames=frames, rescue_pass_counts=passes,
                episodes_with_drive_counts=len(with_counts),
                student_driven_steps=student, expert_driven_steps=expert,
                known_student_steps_fraction_all_saved_frames=student / frames if frames else None,
                student_fraction_of_steps_with_drive_metadata=(
                    student / (student + expert) if student + expert else None),
                drive_count_mismatch_sources=[r['source'] for r in records
                                              if r['drive_counts_match_frames'] is False],
                metadata_step_mismatch_sources=[r['source'] for r in records
                                                if not r['step_count_matches_metadata']])


def calibration(root):
    paths = sorted(root.rglob('*.hdf5'))[:8]
    total, count = None, 0
    for path in paths:
        with h5py.File(path, 'r', locking=False) as handle:
            group = handle['data/demo_0']
            closed = group['actions'][()][:, 64] < 0
            if closed.any():
                fingers = group['teleop/finger_joints/right'][()][closed]
                total = fingers.sum(0) if total is None else total + fingers.sum(0)
                count += len(fingers)
    return dict(root=str(root), source_files=list(map(str, paths)), closed_frames=count,
                closed_pose=(total / count).astype(np.float64).tolist() if count else None)


def main():
    started = time.time()
    cache, datasets, checkpoints, errors = {}, [], {}, []
    chains = {
        'historical_full': ['af60v8f_base'] + [f'af60v8f_ds_dagger{i}' for i in (1, 2, 3)],
        **{f'pick{b}': [f'pick{b}_base5p7'] + [f'pick{b}_{TAG}_r{i}' for i in (1, 2, 3)]
           for b in (10, 50)},
    }
    for chain, names in chains.items():
        previous = set()
        for round_id, name in enumerate(names):
            directory = ROOT / 'datasets' / name
            links = sorted((directory / 'hdf5').glob('*.hdf5'))
            sources = {p.resolve() for p in links}
            assert previous <= sources, (chain, round_id, 'cumulative prefix missing')
            assert len(sources) == len(links), (name, 'duplicate source episode')
            for source in sorted(sources):
                if str(source) in cache:
                    continue
                try:
                    cache[str(source)] = read_episode(source)
                except (OSError, KeyError, ValueError) as error:
                    errors.append(dict(source=str(source), error=repr(error)))
            available = [cache[str(p)] for p in sorted(sources) if str(p) in cache]
            added = [cache[str(p)] for p in sorted(sources - previous) if str(p) in cache]
            info_path = directory / 'lerobot/ds/meta/info.json'
            info = json.loads(info_path.read_text()) if info_path.exists() else None
            datasets.append(dict(chain=chain, round=round_id, dataset=str(directory),
                raw_source_count=len(sources), cumulative=summarize(available), added=summarize(added),
                lerobot_episodes=info['total_episodes'] if info else None,
                lerobot_frames=info['total_frames'] if info else None,
                raw_and_lerobot_counts_match=(
                    len(available) == info['total_episodes'] and
                    sum(r['frames'] for r in available) == info['total_frames'] if info else False),
                added_sources=list(map(str, sorted(sources - previous)))))
            previous = sources
    for run_dir in sorted((ROOT / 'checkpoints').iterdir()):
        if not (run_dir.name.startswith('af60v8f57') or
                any(run_dir.name.startswith(f'pick{b}_{TAG}_') for b in (10, 50))):
            continue
        for checkpoint in sorted(run_dir.glob('checkpoint-*')):
            files = {}
            for name in ('experiment_cfg/dataset_statistics.json', 'experiment_cfg/conf.yaml',
                         'experiment_cfg/config.yaml'):
                path = checkpoint / name
                if path.exists():
                    data = path.read_bytes()
                    files[name] = dict(sha256=hashlib.sha256(data).hexdigest(), text=data.decode())
            checkpoints[str(checkpoint)] = files
    candidates = [ROOT / 'datasets/af60v8f_base/hdf5',
                  ROOT / 'datasets/af60v8f_demos'] + [
                      ROOT / f'dline/pick{b}_{TAG}/demo_root' for b in (10, 50)]
    calibrations = []
    for root in candidates:
        try:
            calibrations.append(calibration(root))
        except (OSError, KeyError, ValueError) as error:
            errors.append(dict(calibration_root=str(root), error=repr(error)))
    result = dict(time=time.time(), elapsed_seconds=time.time() - started,
        datasets=datasets, episodes=cache, checkpoints=checkpoints,
        candidate_closed_pose_calibrations=calibrations, errors=errors,
        limitations=[
            'Calibration roots are candidates; confirm historical collector arguments before attributing a vector to that run.',
            'Student-driven counts include actions before their resulting state; this reports execution counts, not a new state-distribution distance metric.',
            'Known student steps divided by all frames is a lower bound when any episode lacks drive metadata; use the metadata-only fraction for measured execution mixing.',
            'No success criterion is recomputed and no robot behavior is judged.'])
    output = ROOT / 'eval_videos/pick_scaling_dataset_composition.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), 'Preserve the existing audit; select a new output before repeating'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), episodes=len(cache), errors=errors,
                         elapsed_seconds=result['elapsed_seconds'])), flush=True)
    if errors or not all(d['raw_and_lerobot_counts_match'] for d in datasets):
        raise SystemExit('Audit incomplete; inspect the saved evidence')


if __name__ == '__main__':
    main()
