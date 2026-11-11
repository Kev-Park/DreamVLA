"""Compare two walking policies on identical, saved expert/DAgger observations."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    root = Path.home() / 'kevin'
    repo = Path(__file__).resolve().parent.parent
    work = root / 'dline/walk_gap_offline20261022'
    review = json.loads((work / 'operator_review.json').read_text())
    assert review['existing_gr00t_environment_approved']
    assert review['account_resources_manually_reviewed']
    assert 0 <= time.time() - review['at'] < 120
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(review['gpu'])
    assert set(os.sched_getaffinity(0)) <= set(review['cpus'])
    assert sys.prefix == str(root / 'Isaac-GR00T/.venv')
    assert not (work / 'launch.json').exists()
    atomic(work / 'launch.json', {'pid': os.getpid(), 'at': time.time(), 'review': review})
    reserved = False
    try:
        selected = subprocess.check_output([sys.executable, str(repo / 'cluster/gpu_slots.py'),
            'reserve', '--owner', str(os.getpid()), '--count', '1', '--mb', '15000',
            '--gpu', str(review['gpu'])], text=True).strip()
        assert selected == str(review['gpu']), 'Reviewed capacity changed'
        reserved = True
        import numpy as np
        import torch
        from gr00t.data.utils import parse_modality_configs
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.eval.open_loop_eval import parse_observation_gr00t
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        torch.set_num_threads(2)
        models = {
            'historical_r2': root / 'checkpoints/T20maxf_dagger2_run01/checkpoint-24000',
            'current_r2': root / 'checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967'}
        datasets = {
            'legacy': (root / 'datasets/T20maxf_ds_dagger2/lerobot/ds', 117, 269),
            'current': (root / 'datasets/walk200_v3_dagger57_20261013_r2/lerobot/ds', 200, 416)}
        config = json.loads((models['current_r2'] / 'processor_config.json').read_text())['processor_kwargs']
        modality = {k: v for k, v in parse_modality_configs(config['modality_configs'])['unitree_g1_sonic'].items() if k != 'rl_info'}
        loaders, plan = {}, []
        for label, (path, base_count, total) in datasets.items():
            loader = LeRobotEpisodeLoader(dataset_path=str(path), modality_configs=modality,
                video_backend='torchcodec', video_backend_kwargs=None)
            assert len(loader) == total
            loaders[label] = loader
            for stratum, first, last in [('base', 0, base_count), ('dagger', base_count, total)]:
                indices = sorted(np.random.default_rng(20261022).choice(np.arange(first, last), size=12, replace=False).tolist())
                for episode in indices:
                    plan.append({'dataset': label, 'stratum': stratum, 'episode': episode})
        atomic(work / 'sample_plan.json', {'created': time.time(), 'episodes': plan,
            'within_episode_samples': ['10_percent', '50_percent', '90_percent', 'first_closed_target', 'first_closed_target_while_observed_open'],
            'inference_seeds_per_observation': [0, 1], 'evaluated_chunk_frames': 8,
            'hand_close_rule': 'mean(abs(7 right_hand_joints)) > 0.6, identical to simulation',
            'limits': 'Teacher-forced observations from training data; no closed-loop success or causal conclusion.'})
        obs_modality = copy.deepcopy(modality)
        obs_modality.pop('action')
        records, observation_hashes = [], {}
        for model_label, checkpoint in models.items():
            atomic(work / 'workflow.json', {'heartbeat': time.time(), 'status': 'loading_model', 'model': model_label, 'rows': len(records)})
            policy = Gr00tPolicy(embodiment_tag='unitree_g1_sonic', model_path=str(checkpoint), device='cuda:0')
            for item in plan:
                loader = loaders[item['dataset']]
                trajectory = loader[item['episode']]
                gt_hands = np.stack(trajectory['action.right_hand_joints']).astype(np.float32)
                observed_hands = np.stack(trajectory['state.right_hand']).astype(np.float32)
                tokens = np.stack(trajectory['action.motion_token']).astype(np.float32)
                closed = np.abs(gt_hands).mean(axis=1) > .6
                observed_open = np.abs(observed_hands).mean(axis=1) <= .6
                n = len(trajectory)
                points = {int((n - 1) * f): [name] for f, name in [(.1, '10_percent'), (.5, '50_percent'), (.9, '90_percent')]}
                for mask, reason in [(closed, 'first_closed_target'), (closed & observed_open, 'first_closed_target_while_observed_open')]:
                    matches = np.flatnonzero(mask)
                    if len(matches):
                        step = int(matches[0])
                        points.setdefault(step, []).append(reason)
                for step, reasons in sorted(points.items()):
                    data = extract_step_data(trajectory, step, obs_modality, EmbodimentTag.resolve('unitree_g1_sonic'))
                    obs = {**{f'state.{k}': v for k, v in data.states.items()},
                           **{f'video.{k}': np.array(v) for k, v in data.images.items()}}
                    for key in modality['language'].modality_keys:
                        obs[key] = data.text
                    parsed = parse_observation_gr00t(obs, modality)
                    digest = hashlib.sha256()
                    for key, value in sorted(obs.items()):
                        digest.update(key.encode())
                        if isinstance(value, str):
                            digest.update(value.encode())
                        else:
                            array = np.ascontiguousarray(value)
                            digest.update(str((array.shape, array.dtype)).encode())
                            digest.update(array.tobytes())
                    key = (item['dataset'], item['episode'], step)
                    if model_label == 'historical_r2':
                        observation_hashes[key] = digest.hexdigest()
                    else:
                        assert observation_hashes[key] == digest.hexdigest(), 'Cross-model observation changed'
                    horizon = min(8, n - step)
                    gt = gt_hands[step:step + horizon]
                    for seed in [0, 1]:
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        with torch.inference_mode():
                            prediction, _ = policy.get_action(parsed)
                        hand = np.asarray(prediction['right_hand_joints'])[0, :horizon]
                        token = np.asarray(prediction['motion_token'])[0, :horizon]
                        assert hand.shape == gt.shape and token.shape == tokens[step:step + horizon].shape
                        assert np.isfinite(hand).all() and np.isfinite(token).all()
                        pred_close = np.abs(hand).mean(axis=1) > .6
                        row = {**item, 'model': model_label, 'step': step, 'frames': n, 'reasons': reasons,
                            'inference_seed': seed, 'observation_sha256': digest.hexdigest(),
                            'target_close': bool(closed[step]), 'observed_open': bool(observed_open[step]),
                            'predicted_close': bool(pred_close[0]),
                            'predicted_hand_meanabs': float(np.abs(hand[0]).mean()),
                            'target_hand_meanabs': float(np.abs(gt[0]).mean()),
                            'hand_mse_first8': float(np.square(hand - gt).mean()),
                            'token_mse_first8': float(np.square(token - tokens[step:step + horizon]).mean()),
                            'predicted_hand_first8': hand.tolist(), 'predicted_token_first8': token.tolist()}
                        records.append(row)
                atomic(work / 'predictions.json', {'at': time.time(), 'records': records})
                atomic(work / 'workflow.json', {'heartbeat': time.time(), 'status': 'predicting', 'model': model_label,
                    'episode': item, 'rows': len(records)})
                print(model_label, item, 'rows', len(records), flush=True)
                del trajectory
                gc.collect()
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        report = {'at': time.time(), 'sampled_episodes_per_dataset': 24, 'inference_seeds': [0, 1],
            'observations_byte_identical_across_models': True, 'groups': {}}
        for model in models:
            for dataset in datasets:
                for stratum in ['base', 'dagger', 'all']:
                    rows = [r for r in records if r['model'] == model and r['dataset'] == dataset and (stratum == 'all' or r['stratum'] == stratum)]
                    positives = [r for r in rows if r['target_close']]
                    negatives = [r for r in rows if not r['target_close']]
                    transition = [r for r in positives if r['observed_open']]
                    mean = lambda values: float(np.mean(values)) if values else None
                    report['groups']['/'.join([model, dataset, stratum])] = {
                        'rows': len(rows), 'closed_target_rows': len(positives), 'open_target_rows': len(negatives),
                        'false_open_given_close_pct': mean([100 * (not r['predicted_close']) for r in positives]),
                        'false_close_given_open_pct': mean([100 * r['predicted_close'] for r in negatives]),
                        'closed_target_observed_open_rows': len(transition),
                        'false_open_at_closed_target_observed_open_pct': mean([100 * (not r['predicted_close']) for r in transition]),
                        'hand_mse_first8': mean([r['hand_mse_first8'] for r in rows]),
                        'token_mse_first8': mean([r['token_mse_first8'] for r in rows])}
        report['limits'] = 'Stratified diagnostic sample of training episodes and selected frames, two correlated inference draws; not a held-out error estimate or causal test.'
        atomic(work / 'results.json', report)
        atomic(work / 'done.json', {'at': time.time(), 'rows': len(records), 'observations': len(observation_hashes)})
        atomic(work / 'workflow.json', {'heartbeat': time.time(), 'status': 'complete', 'rows': len(records)})
    except BaseException as exc:
        atomic(work / 'abort.json', {'at': time.time(), 'error': repr(exc)})
        raise
    finally:
        if reserved:
            subprocess.run([sys.executable, str(repo / 'cluster/gpu_slots.py'), 'release', '--owner', str(os.getpid()),
                '--gpu', str(review['gpu'])], check=True)


if __name__ == '__main__':
    main()
