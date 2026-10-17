"""Prepare the round-3 inventory only after the real round-2 handoff.

Run on Adroit after copying the final inference checkpoint and the original
round-2 .trained.json receipt. This does not submit jobs or approve their
environment, hardware, or storage allocation.
"""
import hashlib
import json
from pathlib import Path
import time


TAG = 'walk200_v3_dagger57_20261013'
PREFIX = Path('datasets') / (TAG + '_r2') / 'lerobot/ds'
FRAMES = 260958
EPISODES = 416
STEPS = (FRAMES * 57 + 159) // 160
TEACHER_SHA = '4912a72813cbaa70cf04b678c5e312fde35c88c88838d2c4d1e5e1ac6813c57b'
BOX_SHA = 'fdd462b05e1fd3671778d5e10f0a9378bdb2bde453434e979b632fcf4ba25a5b'
EXCLUDED = {'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def student_inputs(root):
    """Cross-check independent completion, dataset and checkpoint metadata."""
    work = root / 'dline' / TAG
    info_path = root / PREFIX / 'meta/info.json'
    receipt_path = work / (TAG + '_r2.trained.json')
    info = json.loads(info_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    assert info['total_frames'] == FRAMES and info['total_episodes'] == EPISODES
    assert not info.get('discarded_episode_indices')
    checkpoint = root / 'checkpoints' / (TAG + '_r2') / f'checkpoint-{STEPS}'
    assert receipt['run'] == TAG + '_r2'
    assert receipt['steps'] == STEPS and receipt['frames'] == FRAMES
    assert receipt['episodes'] == EPISODES and receipt['batch_size'] == 16
    assert Path(receipt['checkpoint']).parts[-3:] == checkpoint.parts[-3:]
    assert Path(receipt['dataset']).parts[-len(PREFIX.parts):] == PREFIX.parts
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == STEPS
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    assert shards
    for name in shards:
        assert Path(name).name == name and name.endswith('.safetensors')
        assert (checkpoint / name).stat().st_size > 8
    return checkpoint, info_path, receipt_path


def main():
    root = (Path.home() / 'kevin').resolve(strict=True)
    work = root / 'dline' / TAG
    output = work / 'adroit_round3_inputs.ready.json'
    assert not output.exists(), 'Reconcile any existing manifest; never overwrite it'
    checkpoint, info_path, receipt_path = student_inputs(root)
    teacher = (root / 'eval_videos/T20_max_v3_env2048/early_stop/best_evaluated.pt').resolve(strict=True)
    assert digest(teacher) == TEACHER_SHA
    box = work / 'box.json'
    assert digest(box) == BOX_SHA
    demos = sorted((work / 'demo_root').glob('*.hdf5'))
    assert len(demos) == 8
    files = [teacher, box, info_path, receipt_path, *demos]
    for pool, count in [('StitchMAX200v3', 200), ('StitchMAX100v3_eval', 100)]:
        refs = sorted((root / 'ref_motions' / pool).glob('*.pkl'))
        assert len(refs) == count
        files.extend(refs)
    files.extend(p for p in checkpoint.rglob('*') if p.is_file() and p.name not in EXCLUDED)
    inventory = []
    seen = set()
    for path in files:
        path = path.resolve(strict=True)
        assert path.is_relative_to(root) and path not in seen
        seen.add(path)
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        inventory.append({'relative': path.relative_to(root).as_posix(),
                          'bytes': after.st_size, 'sha256': sha})
    result = {'at': time.time(), 'round': 3, 'student_run': TAG + '_r2',
              'student_checkpoint_relative': checkpoint.relative_to(root).as_posix(),
              'student_frames': FRAMES, 'student_steps': STEPS,
              'reference_pool': 'StitchMAX200v3', 'student_attempts': 200,
              'teacher_sha256': TEACHER_SHA, 'input_files': inventory,
              'allocation_approval_granted': False, 'jobs_submitted': 0}
    # Exclusive creation protects a previously reviewed inventory.
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'manifest': str(output), 'files': len(inventory),
                      'student_steps': STEPS, 'jobs_submitted': 0}))


if __name__ == '__main__':
    main()
