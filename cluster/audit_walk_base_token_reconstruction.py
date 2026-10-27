"""Check native encoder/token composition against saved expert labels on CPU.

Pure mathematical definitions are read unchanged from the checked-out sources;
this does not import or replace the Isaac runtime, install packages, or decode
actions. Recorded reference windows deliberately precede the harder validation
of reference-file interpolation for saved VLA evaluation inputs.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def definitions(path, names, namespace):
    """Execute selected original, dependency-free definitions without edits."""
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in selected}


def reference_feature_library(base, repo, payloads):
    """Use native interpolation for the two reference fields the encoder reads.

    Auxiliary spatial buffers satisfy the native method's unused return fields.
    The public result exposes only joints and root rotation so those buffers
    cannot accidentally be used to reconstruct other task observations.
    """
    import joblib
    import numpy as np
    import torch
    import isaac_utils.rotations as rotations
    path = repo / 'Training/source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib/motion_lib_base.py'
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and any(isinstance(m, ast.FunctionDef) and m.name == 'get_motion_state' for m in n.body))
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ('get_motion_state', '_calc_frame_blend')]
    assert len(methods) == 2
    container = ast.ClassDef(name='NativeMotionReferenceAudit', bases=[], keywords=[], body=methods, decorator_list=[])
    ns = {'torch': torch, 'slerp': rotations.slerp}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[container], type_ignores=[])), str(path), 'exec'), ns)
    native_names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'JointNamesOrder' for t in n.targets))
    joints, quats, proof, by_source = [], [], [], {}
    for source, identity in payloads.items():
        assert identity['pool'] in ('StitchMAX200v3', 'StitchMAX100v3_eval')
        assert Path(identity['reference_name']).name == identity['reference_name']
        file = base / 'ref_motions' / identity['pool'] / identity['reference_name']
        assert file.resolve().is_relative_to(base.resolve())
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest == identity['sha256'], 'Reference payload differs: ' + str(file)
        data = joblib.load(file)
        assert not any(k in data for k in ('punch_pos_real', 'kick_pos_real', 'squat_pos_real'))
        p = torch.as_tensor(np.array(data['joints']), dtype=torch.float32).clone()
        if p.shape[1] == 27:
            p = torch.cat([p[:, :13], p.new_zeros((len(p), 2)), p[:, 13:]], dim=1)
        q = torch.as_tensor(np.array(data['global_pose'].rotation().wxyz), dtype=torch.float32).clone()
        assert p.shape[1] == 29 and q.shape == (len(p), 4)
        by_source[source] = len(joints)
        joints.append(p); quats.append(q)
        proof.append({'source': source, **identity, 'verified_sha256': digest, 'frames': len(p), 'native_dt': .05})
    count, frames = len(joints), max(map(len, joints))
    obj = ns['NativeMotionReferenceAudit']()
    obj._motion_num_frames = torch.tensor([len(p) for p in joints], dtype=torch.int32)
    obj._motion_dt = torch.full((count,), .05, dtype=torch.float32)
    obj._motion_lengths = obj._motion_num_frames * obj._motion_dt
    def padded(values):
        return torch.stack([torch.cat([v, v[-1:].expand(frames-len(v), *v.shape[1:])]) for v in values])
    obj.dof_pos = padded(joints); obj.quats = padded(quats)
    obj.transl = torch.zeros(count, frames, 3)
    obj.local_keypts = torch.zeros(count, frames, 1, 3); obj.global_keypts = torch.zeros_like(obj.local_keypts)
    obj.offsets = torch.zeros(count, 3); obj.grab_pos = torch.zeros(count, 3)
    obj.switch_idxs = torch.zeros(count, dtype=torch.long); obj._contact_lead = 0; obj.object_poses = None
    class ReferenceFeaturesOnly:
        def get_motion_state(self, ids, times):
            state = obj.get_motion_state(ids, times)
            return {k: state[k] for k in ('dof_pos', 'root_rot')}
    return ReferenceFeaturesOnly(), by_source, native_names, {
        'references': proof, 'motion_lib_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rotation_source_sha256': hashlib.sha256(Path(rotations.__file__).read_bytes()).hexdigest(),
        'native_interpolation_ast_sha256': {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in methods}}


def main():
    base = Path.home() / 'kevin'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=base / 'eval_videos/walk_teacher_reconstruction_validation_20261027/numeric_windows.json')
    parser.add_argument('--output', type=Path, default=base / 'eval_videos/walk_teacher_reconstruction_validation_20261027/token_parity.json')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve the previous audit before another run'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This audit is CPU-only'
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import torch
    from torch import nn
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parent.parent
    native = repo / 'Training/scripts/reinforcement_learning/rsl_rl/vla_sonic'
    math_file = repo / 'Training/source/isaaclab/isaaclab/utils/math.py'
    action_file = native / 'token_action_wrapper.py'
    adapter_file = native / 'token_adapter_wrapper.py'
    data = json.loads(args.input.read_text())
    assert data['format'] == 'expert_token_validation_windows_v1'
    rows = data['samples']; sources = {Path(r['relative']).name: r for r in data['sources']}
    assert len(rows) == 24 and len(sources) == 3
    assert all(r['metadata']['encoder_mode'] == 'g1' and r['metadata']['residual_transform'] == 'additive_free'
               and r['metadata']['residual_scale'] == .1 for r in sources.values())
    names = next(ast.literal_eval(n.value) for n in ast.parse(action_file.read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'UTM_29_JOINT_NAMES' for t in n.targets))
    ref_names = next(iter(sources.values()))['ref_joint_names']
    assert all(r['ref_joint_names'] == ref_names for r in sources.values())
    assert len(names) == len(ref_names) == len(set(names)) == 29 and set(names) == set(ref_names)

    math_ns = {'torch': torch, '__name__': __name__}
    ast_hashes = definitions(math_file, ['normalize', 'matrix_from_quat', 'quat_conjugate', 'quat_mul', 'yaw_quat'], math_ns)
    ns = {'torch': torch, 'nn': nn, 'math_utils': SimpleNamespace(**math_ns),
          'N_FUTURE_FRAMES': 10, 'FUTURE_DT': .1, 'N_BODY_JOINTS': 29,
          'TOKEN_TOTAL_DIM': 64, 'FSQ_MAX_NUM_TOKENS': 2, 'FSQ_NUM_LEVELS': 32, '__name__': __name__}
    ast_hashes.update(definitions(action_file, ['FSQ'], ns))
    original = next(n for n in ast.parse(adapter_file.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'TokenAdapterVecEnvWrapper')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_update_base_token', 'compose', 'snap', 'expert_token')]
    assert len(methods) == 4
    for node in methods: ast_hashes[node.name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
    container = ast.ClassDef(name='NativeEncoderMethodAudit', bases=[], keywords=[], body=methods, decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[container], type_ignores=[]))
    exec(compile(tree, str(adapter_file), 'exec'), ns)
    sys.path.insert(0, str(native.parent))
    from vla_sonic.sonic_pt import load_sonic_pt
    sonic = base / 'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'
    encoder, decoder = load_sonic_pt(str(sonic), 'cpu', encoder='g1')
    del decoder
    gc.collect()

    def tensor(key): return torch.tensor([r[key] for r in rows], dtype=torch.float32)
    pos, quat = tensor('reference_dof_pos'), tensor('reference_root_quat_w')
    robot_quat, latent, target = tensor('robot_root_quat_w'), tensor('expert_latent'), tensor('expert_token')
    assert pos.shape == (24, 10, 29) and quat.shape == (24, 10, 4)
    assert latent.shape == (24, 65) and target.shape == (24, 64)
    assert all(torch.isfinite(t).all() for t in (pos, quat, robot_quat, latent, target))

    class RecordedReferenceWindow:
        def get_motion_state(self, ids, times):
            assert torch.equal(ids, torch.arange(24).repeat_interleave(10))
            assert times.shape == (240,)
            return {'dof_pos': pos.reshape(240, 29), 'root_rot': quat.reshape(240, 4)}

    obj = ns['NativeEncoderMethodAudit']()
    obj.num_envs = 24; obj._dev = torch.device('cpu'); obj.pt_mode = True; obj.encoder_mode = 'g1'
    obj.encoder = encoder; obj.fsq = ns['FSQ']([32] * 32)
    obj._future_offsets = (torch.arange(10, dtype=torch.float32) * .1).unsqueeze(0)
    obj._ref_to_sonic_perm = [names.index(n) for n in ref_names]
    obj._parity_printed = True  # Skip the simulator-only logging branch, after the token is calculated.
    obj.residual_scale = .1; obj.residual_transform = 'additive_free'
    obj.unwrapped = SimpleNamespace(episode_length_buf=torch.tensor([r['frame'] for r in rows]),
        step_dt=.02, start_motion_times=torch.zeros(24), motion_ids=torch.arange(24),
        motion_lib=RecordedReferenceWindow(), scene={'robot': SimpleNamespace(data=SimpleNamespace(root_quat_w=robot_quat))})
    with torch.inference_mode():
        obj._update_base_token()
        reconstructed = obj.expert_token(latent)
        zero_base_control = obj.snap(.1 * latent[:, :64])
    error = (reconstructed - target).abs()
    exact = (error == 0).all(dim=1)
    result = {'at': time.time(), 'status': 'exact_label_parity' if bool(exact.all()) else 'label_parity_mismatch',
        'samples': 24, 'source_recordings': 3, 'exact_complete_token_matches': int(exact.sum()),
        'exact_coordinate_matches': int((error == 0).sum()), 'coordinates': error.numel(),
        'max_abs_error': float(error.max()),
        'zero_base_negative_control_complete_token_matches': int((zero_base_control == target).all(dim=1).sum()),
        'pure_native_definition_ast_sha256': ast_hashes,
        'source_files_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (math_file, action_file, adapter_file, native / 'sonic_pt.py')},
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'gpu_used': False,
        'models_or_decoders_changed': False,
        'per_sample': [{'source': r['source_name'], 'frame': r['frame'], 'exact': bool(exact[i]),
                        'base_token': obj._base_token[i].tolist(), 'reconstructed_token': reconstructed[i].tolist(),
                        'target_token': target[i].tolist(), 'max_abs_error': float(error[i].max())}
                       for i, r in enumerate(rows)],
        'limits': ['Reference lookahead here comes from saved recording rows, not a separately loaded reference pickle.',
                   'Small floating-clock differences between frame+5*k and t+0.1*k remain a possible mismatch source.',
                   'Final token equality can mask base-token differences in saturated coordinates; it does not certify every base coordinate.',
                   'This is feature/token validation only: neither residual expert has been queried on saved VLA evaluation inputs.',
                   'Held-out reference identity, interpolation, input layout and native normalization still require validation before conditional expert comparisons.']}
    if 'reference_payloads' in data:
        previous_base = obj._base_token.clone()
        library, by_source, loaded_names, proof = reference_feature_library(base, repo, data['reference_payloads'])
        assert loaded_names == ref_names, 'Reference joint-name layout differs'
        ids = torch.tensor([by_source[r['source_name']] for r in rows])
        times = (obj.unwrapped.episode_length_buf * .02).unsqueeze(1) + obj._future_offsets
        windows = library.get_motion_state(ids.repeat_interleave(10), times.flatten())
        # The recording samples (frame+5*k)*dt, while the encoder uses
        # frame*dt+k*.1. Compare geometry at the recording's exact float clock;
        # retain the native encoder clock for the base/label parity checks.
        row_times = (obj.unwrapped.episode_length_buf.unsqueeze(1) + torch.arange(10) * 5) * .02
        recorded_windows = library.get_motion_state(ids.repeat_interleave(10), row_times.flatten())
        pos_error = (recorded_windows['dof_pos'].reshape_as(pos) - pos).abs().max()
        quat_error = (recorded_windows['root_rot'].reshape_as(quat) - quat).abs().max()
        shifted = library.get_motion_state(ids.repeat_interleave(10), row_times.flatten() + .05)
        shifted_error = (shifted['dof_pos'].reshape_as(pos) - pos).abs().max()
        assert shifted_error > 1e-3, 'Reference-clock negative control lacks excitation'
        obj.unwrapped.motion_ids = ids; obj.unwrapped.motion_lib = library
        with torch.inference_mode():
            obj._update_base_token()
            final = obj.expert_token(latent)
        base_exact = (obj._base_token == previous_base).all(dim=1)
        label_exact = (final == target).all(dim=1)
        passed = bool(pos_error < 1e-5 and quat_error < 1e-5 and base_exact.all() and label_exact.all())
        result['reference_pickle_validation'] = {**proof, 'passed': passed,
            'reference_joint_max_abs_error': float(pos_error), 'reference_quat_max_abs_error': float(quat_error),
            'geometry_validation_clock': '(recorded frame+5*k)*step_dt; unchanged1e-5absolute tolerance',
            'native_encoder_clock_reference_quat_vs_recorded_rows_max_abs_error': float((windows['root_rot'].reshape_as(quat) - quat).abs().max()),
            'max_encoder_vs_recorded_row_time_difference_seconds': float((times-row_times).abs().max()),
            'shifted_one_reference_frame_control_joint_max_abs_error': float(shifted_error),
            'complete_base_tokens_equal_recorded_window_reconstruction': int(base_exact.sum()),
            'complete_final_tokens_equal_recorded_expert_labels': int(label_exact.sum()),
            'max_base_token_abs_error': float((obj._base_token - previous_base).abs().max()),
            'max_final_token_abs_error': float((final - target).abs().max())}
        if not passed: result['status'] = 'reference_pickle_validation_mismatch'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'samples', 'exact_complete_token_matches', 'exact_coordinate_matches', 'coordinates', 'max_abs_error')}))


if __name__ == '__main__':
    main()
