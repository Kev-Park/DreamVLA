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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'samples', 'exact_complete_token_matches', 'exact_coordinate_matches', 'coordinates', 'max_abs_error')}))


if __name__ == '__main__':
    main()
