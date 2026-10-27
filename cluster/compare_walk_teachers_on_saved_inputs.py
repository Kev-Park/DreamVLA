"""Compare unchanged residual expert means on identical saved student inputs.

CPU-only diagnostic. Validate the original byte-level task observations and all
ten reference-joint blocks before querying either model. These selected points
do not estimate closed-loop success or identify a causal performance mechanism.
"""
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from audit_walk_base_token_reconstruction import definitions, reference_feature_library


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import numpy as np
    import torch
    from torch import nn
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    torch.set_num_threads(1)
    base = Path.home() / 'kevin'; repo = Path(__file__).resolve().parent.parent
    work = base / 'eval_videos/walk_teacher_reconstruction_validation_20261027'
    output = work / 'conditional_expert_means.json'; assert not output.exists()
    source = work / 'captured_numeric_inputs.json'
    manifest = json.loads((work / 'reference_identity_manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['numeric_export_sha256']
    data = json.loads(source.read_text()); rows = data['rows']; assert len(rows) == 154
    contract = manifest['actual_input_contract']; assert contract['captured_actual_task_shape'] == 951
    task = torch.tensor([r['task_policy'][0] for r in rows], dtype=torch.float32)
    assert task.shape == (154, 951) and torch.isfinite(task).all()
    for i, row in enumerate(rows):
        assert hashlib.sha256(task[i:i+1].numpy().tobytes()).hexdigest() == row['task_policy_sha256']
        assert row['metadata']['decoder_contract'] == 'previous_full_fsq'
    payloads = manifest['reference_payloads']
    library, by_source, ref_names, reference_proof = reference_feature_library(base, repo, payloads)
    ids = torch.tensor([by_source[str(r['metadata']['motion_id'])] for r in rows])
    times = torch.tensor([r['metadata']['motion_time'] for r in rows], dtype=torch.float32)
    root_quat = torch.tensor([r['scene']['root_quat'][0] for r in rows], dtype=torch.float32)
    assert root_quat.shape == (154, 4) and torch.isfinite(root_quat).all()
    offsets = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9]
    terms = ['target_ref_curr', 'target_ref_next', 'target_ref_next_next',
             *['target_ref_slim_0' + str(i) for i in range(3, 10)]]
    reference_errors = {}
    for name, offset in zip(terms, offsets):
        start = contract['actual_feature_slices'][name]['start']
        expected = library.get_motion_state(ids, times + offset)['dof_pos']
        error = (expected - task[:, start:start+29]).abs()
        reference_errors[name] = {'max_abs_error': float(error.max()),
            'worst_point_index': int(torch.argmax(error).item() // 29)}
    shifted = library.get_motion_state(ids, times + .05)['dof_pos']
    start = contract['actual_feature_slices']['target_ref_curr']['start']
    shifted_error = float((shifted - task[:, start:start+29]).abs().max())
    validation = {'at': time.time(), 'points': 154, 'task_observation_bytes_preserved': True,
        'reference_joint_blocks': reference_errors, 'absolute_tolerance': 1e-5,
        'shifted_reference_clock_negative_control_max_abs_error': shifted_error,
        'reference_payloads': reference_proof, 'real_teacher_queries_executed': False}
    validation['passed'] = all(v['max_abs_error'] < 1e-5 for v in reference_errors.values()) and shifted_error > 1e-3
    (work / 'conditional_teacher_input_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    assert validation['passed'], 'Held-out reference/input identity validation failed: ' + str(reference_errors)

    native = repo / 'Training/scripts/reinforcement_learning/rsl_rl/vla_sonic'
    math_file = repo / 'Training/source/isaaclab/isaaclab/utils/math.py'
    math_ns = {'torch': torch, '__name__': __name__}
    definitions(math_file, ['normalize', 'matrix_from_quat', 'quat_conjugate', 'quat_mul', 'yaw_quat'], math_ns)
    ns = {'torch': torch, 'nn': nn, 'math_utils': SimpleNamespace(**math_ns), 'N_FUTURE_FRAMES': 10,
          'FUTURE_DT': .1, 'N_BODY_JOINTS': 29, 'TOKEN_TOTAL_DIM': 64,
          'FSQ_MAX_NUM_TOKENS': 2, 'FSQ_NUM_LEVELS': 32, '__name__': __name__}
    definitions(native / 'token_action_wrapper.py', ['FSQ'], ns)
    original = next(n for n in ast.parse((native / 'token_adapter_wrapper.py').read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'TokenAdapterVecEnvWrapper')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in
               ('_update_base_token', '_append_base_token', 'compose', 'snap', 'expert_token')]
    assert len(methods) == 5
    container = ast.ClassDef(name='NativeEncoderMethodAudit', bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[container], type_ignores=[])),
                 str(native / 'token_adapter_wrapper.py'), 'exec'), ns)
    sys.path.insert(0, str(native.parent))
    from vla_sonic.sonic_pt import load_sonic_pt
    encoder, decoder = load_sonic_pt(str(base / 'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'), 'cpu', encoder='g1')
    del decoder; gc.collect()
    action_tree = ast.parse((native / 'token_action_wrapper.py').read_text())
    names = next(ast.literal_eval(n.value) for n in action_tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'UTM_29_JOINT_NAMES' for t in n.targets))
    obj = ns['NativeEncoderMethodAudit'](); obj.num_envs = len(rows); obj._dev = torch.device('cpu')
    obj.encoder = encoder; obj.fsq = ns['FSQ']([32] * 32); obj.pt_mode = True; obj.encoder_mode = 'g1'
    obj._future_offsets = (torch.arange(10, dtype=torch.float32) * .1).unsqueeze(0)
    obj._ref_to_sonic_perm = [names.index(n) for n in ref_names]; obj._parity_printed = True
    obj.residual_scale = .1; obj.residual_transform = 'additive_free'
    obj.unwrapped = SimpleNamespace(episode_length_buf=torch.zeros(len(rows), dtype=torch.long), step_dt=.02,
        start_motion_times=times, motion_ids=ids, motion_lib=library,
        scene={'robot': SimpleNamespace(data=SimpleNamespace(root_quat_w=root_quat))})
    with torch.inference_mode():
        obj._update_base_token()
        features = obj._append_base_token({'policy': task.clone()})['policy']
    assert features.shape == (154, 1015) and torch.equal(features[:, :951], task) and torch.isfinite(features).all()
    input_digest = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    obs = TensorDict({'policy': features}, batch_size=[len(rows)])
    predictions, tokens, models = {}, {}, {}
    for label, name in [('historical', 'T20_max_model_4999.pt'), ('current', 'T20_max_v3_model_4500.pt')]:
        checkpoint = base / 'residuals' / name
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == manifest['expert_checkpoint_sha256'][label]
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
        model = ActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 65,
            actor_hidden_dims=[256, 128], critic_hidden_dims=[512, 256, 256], activation='elu',
            actor_obs_normalization=False, critic_obs_normalization=False)
        model.load_state_dict(state, strict=True); model.eval()
        assert set(model.state_dict()) == set(state) and all(torch.equal(model.state_dict()[k], state[k]) for k in state)
        with torch.inference_mode():
            predictions[label] = model.act_inference(obs).clone()
            tokens[label] = obj.expert_token(predictions[label]).clone()
        assert predictions[label].shape == (154, 65) and torch.isfinite(predictions[label]).all()
        assert hashlib.sha256(features.numpy().tobytes()).hexdigest() == input_digest
        models[label] = {'checkpoint': name, 'sha256': digest, 'normalizer': type(model.actor_obs_normalizer).__name__,
                         'strict_weights_equal': True, 'common_input_sha256': input_digest}
        del model
    result_rows = []
    for i, row in enumerate(rows):
        student_hand = np.asarray(row['student_executed_hand_predictions'], dtype=np.float32)
        student_close = np.abs(student_hand).mean(axis=1) > .6
        ref_close_offset = contract['actual_feature_slices']['right_hand_state_target_val']['start']
        result_rows.append({'point_index': i, 'capture_model': row['capture_model'], 'sample_plan': row['sample_plan'],
            'metadata': row['metadata'], 'capture_sha256': row['capture_sha256'],
            'full_actor_input_sha256': hashlib.sha256(features[i:i+1].numpy().tobytes()).hexdigest(),
            'base_token': obj._base_token[i].tolist(),
            'historical_actor_mean': predictions['historical'][i].tolist(),
            'current_actor_mean': predictions['current'][i].tolist(),
            'historical_final_body_token': tokens['historical'][i].tolist(),
            'current_final_body_token': tokens['current'][i].tolist(),
            'historical_commands_close': bool(predictions['historical'][i, 64] < 0),
            'current_commands_close': bool(predictions['current'][i, 64] < 0),
            'student_first_command_close': bool(student_close[0]), 'student_any_executed_command_close': bool(student_close.any()),
            'reference_closed_observation': float(task[i, ref_close_offset]),
            'expert_body_token_l2': float(torch.linalg.vector_norm(tokens['historical'][i] - tokens['current'][i]))})
    def summarize(selected):
        n = len(selected)
        return {'points': n, 'references': len({r['metadata']['motion_id'] for r in selected}),
            'historical_close': sum(r['historical_commands_close'] for r in selected),
            'current_close': sum(r['current_commands_close'] for r in selected),
            'historical_close_current_open': sum(r['historical_commands_close'] and not r['current_commands_close'] for r in selected),
            'current_close_historical_open': sum(r['current_commands_close'] and not r['historical_commands_close'] for r in selected),
            'student_first_command_close': sum(r['student_first_command_close'] for r in selected),
            'mean_expert_body_token_l2': sum(r['expert_body_token_l2'] for r in selected) / n if n else None,
            'near_zero_finger_mean_counts': {label: sum(abs(r[label + '_actor_mean'][64]) < 1e-4 for r in selected)
                                             for label in ('historical', 'current')}}
    groups = {}
    for label in ('current_r2', 'historical_r2'):
        selected = [r for r in result_rows if r['capture_model'] == label]
        reasons = sorted({reason for r in selected for reason in r['sample_plan']['reasons']})
        groups[label] = {'all': summarize(selected), 'by_reason': {reason: summarize([r for r in selected if reason in r['sample_plan']['reasons']]) for reason in reasons}}
    result = {'at': time.time(), 'points': len(rows), 'teacher_predictions': 2 * len(rows),
        'models': models, 'groups': groups, 'per_point': result_rows,
        'input_export_sha256': manifest['numeric_export_sha256'], 'validated_reference_joint_blocks': reference_errors,
        'CPU_affinity': sorted(os.sched_getaffinity(0)), 'GPU_used': False, 'accepted_decoder_changed': False,
        'limits': ['Conditional deterministic expert means, not closed-loop rescue rollouts or success estimates.',
                  'Selected78/76query points on10reference replicas per student; repeated points and overlapping reason groups are not independent trials.',
                  'The missing64features are reconstructed with validated native source methods; direct base-token captures were unavailable.',
                  'CPUfloat32 inference can differ slightly from native GPU numerics; near-zero finger means are reported.',
                  'Teacher differences on identical observations do not prove which teacher would rescue the state or cause the VLA performance gap.']}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'points': len(rows), 'groups': groups}), flush=True)


if __name__ == '__main__':
    main()
