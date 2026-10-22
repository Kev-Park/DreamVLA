"""Compare unchanged policies on saved, previously unseen R3 rescue-prefix observations."""
import collections
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

from walk_gap_offline_predictions import atomic


def main():
    root = Path.home() / 'kevin'
    repo = Path(__file__).resolve().parent.parent
    work = root / 'dline/walk_gap_rescue_prefix20261022'
    output = root / 'eval_videos/walk_gap_rescue_prefix20261022'
    review = json.loads((work / 'operator_review.json').read_text())
    assert review['existing_gr00t_environment_approved'] and review['account_resources_manually_reviewed']
    assert 0 <= time.time() - review['at'] < 120
    assert sys.prefix == str(root / 'Isaac-GR00T/.venv')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(review['gpu'])
    assert set(os.sched_getaffinity(0)) <= set(review['cpus'])
    assert not output.exists() and not (work / 'launch.json').exists()
    output.mkdir()
    atomic(work / 'launch.json', {'at': time.time(), 'pid': os.getpid(), 'review': review})
    reserved = False
    try:
        selected = subprocess.check_output([sys.executable, str(repo / 'cluster/gpu_slots.py'),
            'reserve', '--owner', str(os.getpid()), '--gpu', str(review['gpu']), '--count', '1', '--mb', '15000'], text=True).strip()
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
        checkpoints = {'historical_r2': root / 'checkpoints/T20maxf_dagger2_run01/checkpoint-24000',
            'current_r2': root / 'checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967'}
        config = json.loads((checkpoints['current_r2'] / 'processor_config.json').read_text())['processor_kwargs']
        modality = {k: v for k, v in parse_modality_configs(config['modality_configs'])['unitree_g1_sonic'].items() if k != 'rl_info'}
        loader = LeRobotEpisodeLoader(dataset_path=str(root / 'datasets/walk200_v3_dagger57_20261013_r3/lerobot/ds'),
            modality_configs=modality, video_backend='torchcodec', video_backend_kwargs=None)
        assert len(loader) == 535
        obs_modality = copy.deepcopy(modality)
        obs_modality.pop('action')
        records, plan, hashes = [], [], {}

        def sample_points(trajectory, episode):
            n = len(trajectory)
            hand = np.stack(trajectory['action.right_hand_joints'])
            observed = np.stack(trajectory['state.right_hand'])
            closed = np.abs(hand).mean(axis=1) > .6
            opened = np.abs(observed).mean(axis=1) <= .6
            student = np.ones(n, bool) if episode['pass'] == 'A' else np.arange(n) < episode['takeover_at']
            assert n == episode['frames'] and int(student.sum()) == episode['student_frames']
            assert int((student & closed & opened).sum()) == episode['student_close_target_observed_open_frames']
            points = []
            categories = [('student_close_while_observed_open', student & closed & opened, 3),
                ('student_open_target_control', student & ~closed & opened, 1)]
            if episode['pass'] == 'B':
                categories.append(('expert_after_takeover_close_while_observed_open', ~student & closed & opened, 2))
            for name, mask, count in categories:
                positions = np.flatnonzero(mask)
                chunks = np.unique(positions // 8)
                if not len(chunks):
                    continue
                selected = chunks[np.unique(np.linspace(0, len(chunks) - 1, min(count, len(chunks)), dtype=int))]
                for chunk in selected:
                    target = int(positions[positions // 8 == chunk][0])
                    points.append({'episode': episode['episode'], 'pass': episode['pass'], 'takeover_at': episode['takeover_at'],
                        'category': name, 'query_step': int(chunk * 8), 'target_step': target, 'chunk_offset': target % 8,
                        'expert_target_close': bool(closed[target]), 'observed_open_at_target': bool(opened[target]),
                        'student_controls_target': bool(student[target])})
            return points

        for episode in review['episodes']:
            assert 416 <= episode['episode'] < 535
            trajectory = loader[episode['episode']]
            plan.extend(sample_points(trajectory, episode))
            del trajectory
        assert plan and len({(p['episode'], p['category'], p['target_step']) for p in plan}) == len(plan)
        atomic(output / 'sample_plan.json', {'at': time.time(), 'points': plan, 'seeds': [0, 1, 2, 3],
            'models': {k: str(v) for k, v in checkpoints.items()}, 'collector_query_boundary': 'Every8frames; predictions indexed at target_step modulo8',
            'no_input_or_decoder_changes': True, 'model_current_r2_training_data_ends_before_episode416': True,
            'limits': ['Rescue prefixes are independent sibling rollouts for references whose initial passA failed; they are not the discarded failed trajectories.',
                'Inputs use saved compressed videos; original rollout diffusion RNG states were not recorded, so these are new prediction draws, not exact original outputs.']})
        grouped = collections.defaultdict(list)
        for point in plan:
            grouped[point['episode']].append(point)
        for model, checkpoint in checkpoints.items():
            atomic(work / 'workflow.json', {'heartbeat': time.time(), 'status': 'loading_model', 'model': model, 'rows': len(records)})
            policy = Gr00tPolicy(embodiment_tag='unitree_g1_sonic', model_path=str(checkpoint), device='cuda:0')
            for episode_id, points in grouped.items():
                trajectory = loader[episode_id]
                gt_hand = np.stack(trajectory['action.right_hand_joints'])
                gt_token = np.stack(trajectory['action.motion_token'])
                queries = collections.defaultdict(list)
                for point in points:
                    queries[point['query_step']].append(point)
                for query_step, targets in sorted(queries.items()):
                    data = extract_step_data(trajectory, query_step, obs_modality, EmbodimentTag.resolve('unitree_g1_sonic'))
                    observation = {**{f'state.{k}': v for k, v in data.states.items()},
                        **{f'video.{k}': np.array(v) for k, v in data.images.items()}}
                    for key in modality['language'].modality_keys:
                        observation[key] = data.text
                    digest = hashlib.sha256()
                    for key, value in sorted(observation.items()):
                        digest.update(key.encode())
                        if isinstance(value, str):
                            digest.update(value.encode())
                        else:
                            array = np.ascontiguousarray(value)
                            digest.update(str((array.shape, array.dtype)).encode())
                            digest.update(array.tobytes())
                    key = (episode_id, query_step)
                    if model == 'historical_r2':
                        hashes[key] = digest.hexdigest()
                    else:
                        assert hashes[key] == digest.hexdigest()
                    parsed = parse_observation_gr00t(observation, modality)
                    for seed in [0, 1, 2, 3]:
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        with torch.inference_mode():
                            prediction, _ = policy.get_action(parsed)
                        for point in targets:
                            offset = point['chunk_offset']
                            hand = np.asarray(prediction['right_hand_joints'])[0, offset]
                            token = np.asarray(prediction['motion_token'])[0, offset]
                            assert hand.shape == (7,) and token.shape == (64,) and np.isfinite(hand).all() and np.isfinite(token).all()
                            records.append({**point, 'model': model, 'seed': seed, 'observation_sha256': digest.hexdigest(),
                                'predicted_close': bool(np.abs(hand).mean() > .6), 'predicted_hand_meanabs': float(np.abs(hand).mean()),
                                'hand_mse_at_target': float(np.square(hand - gt_hand[point['target_step']]).mean()),
                                'token_mse_at_target': float(np.square(token - gt_token[point['target_step']]).mean()),
                                'predicted_hand': hand.tolist(), 'predicted_token': token.tolist()})
                atomic(output / 'predictions.json', {'at': time.time(), 'records': records})
                atomic(work / 'workflow.json', {'heartbeat': time.time(), 'status': 'predicting', 'model': model, 'episode': episode_id, 'rows': len(records)})
                print(model, episode_id, 'rows', len(records), flush=True)
                del trajectory
                gc.collect()
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        summaries = {}
        for model in checkpoints:
            for phase in ['A', 'B']:
                for category in sorted({p['category'] for p in plan}):
                    rows = [r for r in records if r['model'] == model and r['pass'] == phase and r['category'] == category]
                    if not rows:
                        continue
                    summaries['/'.join([model, phase, category])] = {'draws': len(rows),
                        'episodes': len({r['episode'] for r in rows}), 'targets': len({(r['episode'], r['target_step']) for r in rows}),
                        'predicted_close_pct': 100 * sum(r['predicted_close'] for r in rows) / len(rows),
                        'expert_target_close': rows[0]['expert_target_close'],
                        'hand_mse_at_target': float(np.mean([r['hand_mse_at_target'] for r in rows])),
                        'token_mse_at_target': float(np.mean([r['token_mse_at_target'] for r in rows]))}
        atomic(output / 'results.json', {'at': time.time(), 'groups': summaries, 'rows': len(records), 'queries': len(hashes),
            'identical_observations_across_models_verified': True, 'no_input_or_decoder_changes': True,
            'limits': ['Small selected sample, four correlated inference draws per target, not a new closed-loop success measurement.',
                'Saved sibling prefixes omit the discarded failed passA trajectories and do not cover failed student control after reference grasp.']})
        atomic(work / 'done.json', {'at': time.time(), 'rows': len(records), 'results': str(output / 'results.json')})
        atomic(work / 'workflow.json', {'heartbeat': time.time(), 'status': 'complete', 'rows': len(records)})
    except BaseException as exc:
        atomic(work / 'abort.json', {'at': time.time(), 'error': repr(exc)})
        raise
    finally:
        if reserved:
            subprocess.run([sys.executable, str(repo / 'cluster/gpu_slots.py'), 'release', '--owner', str(os.getpid()), '--gpu', str(review['gpu'])], check=True)


if __name__ == '__main__':
    main()
