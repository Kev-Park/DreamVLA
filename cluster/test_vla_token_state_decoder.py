"""CPU-only token-contract tests; no Isaac app, model weights, or rollouts.

Compile the actual methods against a capture-only decoder. This tests the
quantization boundary while leaving environment integration to the GPU eval.
"""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

torch.set_num_threads(1)
RSL = Path(__file__).resolve().parents[1] / 'Training/scripts/reinforcement_learning/rsl_rl'
SOURCE = ast.parse((RSL / 'vla_sonic/token_action_wrapper.py').read_text())
NS = {'torch': torch, 'nn': torch.nn, 'TOKEN_TOTAL_DIM': 64,
      'FSQ_MAX_NUM_TOKENS': 2, 'FSQ_NUM_LEVELS': 32, 'N_BODY_JOINTS': 29}


def execute(node):
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-wrapper-method>', 'exec'), NS)


execute(next(n for n in SOURCE.body if isinstance(n, ast.ClassDef) and n.name == 'FSQ'))
base = next(n for n in SOURCE.body if isinstance(n, ast.ClassDef) and n.name == 'TokenActionDecoderVecEnvWrapper')
execute(next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_body_29'))
NS['TokenActionDecoderVecEnvWrapper'] = type('NativeLatentDecoder', (), {'_decode_body_29': NS['_decode_body_29']})
execute(next(n for n in SOURCE.body if isinstance(n, ast.ClassDef) and n.name == 'TokenStateDecoderVecEnvWrapper'))
spec = importlib.util.spec_from_file_location('fsq_contract_helper', RSL / 'vla_sonic/fsq.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
eval_source = ast.parse((RSL / 'eval_vla_sonic.py').read_text())
NS.update(np=np, fsq_lattice_snap=helper.fsq_lattice_snap)
execute(next(n for n in eval_source.body if isinstance(n, ast.FunctionDef) and n.name == 'fsq_snap_token'))


def decoded_token_state(wrapper_class, token):
    obj = object.__new__(wrapper_class)
    obj.num_envs = len(token)
    obj.fsq = NS['FSQ']([32] * 32)

    def build(state):
        return torch.cat([state, torch.full((len(state), 930), 42.0)], dim=1)

    def decoder(obs):
        obj.decoder_input = obs.clone()
        return torch.zeros((len(obs), 29))

    obj._build_obs_dict = build
    obj.decoder = decoder
    obj._decode_body_29(token)
    return obj.decoder_input


class TokenStateContractTests(unittest.TestCase):
    def setUp(self):
        self.grid = (torch.arange(-16, 16, dtype=torch.float32) / 16).reshape(32, 1).repeat(1, 64)

    def test_perfect_vla_predictions_preserve_every_grid_level(self):
        snapped = NS['fsq_snap_token'](self.grid.numpy())
        obs = decoded_token_state(NS['TokenStateDecoderVecEnvWrapper'], torch.from_numpy(snapped))
        self.assertTrue(torch.equal(obs[:, :64], self.grid))
        self.assertTrue(torch.equal(obs[:, 64:], torch.full((32, 930), 42.0)))

    def test_near_grid_predictions_snap_to_supervision_targets(self):
        snapped = NS['fsq_snap_token'](self.grid.numpy() + np.float32(0.01))
        obs = decoded_token_state(NS['TokenStateDecoderVecEnvWrapper'], torch.from_numpy(snapped))
        self.assertTrue(torch.equal(obs[:, :64], self.grid))

    def test_unsnapped_token_states_are_not_squashed(self):
        continuous = self.grid + 0.01
        obs = decoded_token_state(NS['TokenStateDecoderVecEnvWrapper'], continuous)
        self.assertTrue(torch.equal(obs[:, :64], continuous))

    def test_native_unbounded_policy_retains_full_fsq(self):
        obs = decoded_token_state(NS['TokenActionDecoderVecEnvWrapper'], self.grid)
        self.assertEqual(int((obs[:, 0] != self.grid[:, 0]).sum()), 20)
        self.assertEqual(float((obs[:, :64] - self.grid).abs().max()), 0.25)


if __name__ == '__main__':
    unittest.main()
