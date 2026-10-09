"""CPU-only contract test for the ablation's actual decoder input (no Isaac import)."""
import ast
from pathlib import Path
import unittest
import torch

source = Path(__file__).resolve().parents[1] / 'Training/scripts/reinforcement_learning/rsl_rl/vla_sonic/direct_expert_eval.py'
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))


class Base:
    @staticmethod
    def snap(x):
        return (x * 16).round().div(16).clamp(-1, 15 / 16)


scope = dict(torch=torch, TokenAdapterVecEnvWrapper=Base)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
Wrapper = scope['DirectExpertEvalWrapper']


class DirectTokenContract(unittest.TestCase):
    def run_mode(self, mode):
        w = Wrapper.__new__(Wrapper)
        w._dev = 'cpu'
        w.num_envs = 1
        w.token_mode = mode
        w._base_token = torch.full((1, 64), 42.)
        w.residual_scale = 0.1
        seen = []
        w._build_obs_dict = lambda token: token
        w.decoder = lambda obs: seen.append(obs.clone()) or torch.zeros((1, 29))
        latent = torch.linspace(-3, 3, 65).reshape(1, 65)
        composed = w.compose(latent)
        torch.testing.assert_close(composed, latent, rtol=0, atol=0)
        w._decode_body_29(composed[:, :64])
        return latent, seen[0]

    def test_raw_output_reaches_decoder_exactly(self):
        latent, decoded = self.run_mode('raw')
        torch.testing.assert_close(decoded, latent[:, :64], rtol=0, atol=0)

    def test_snap_only_when_requested(self):
        latent, decoded = self.run_mode('snap')
        torch.testing.assert_close(decoded, Base.snap(latent[:, :64]), rtol=0, atol=0)

    def test_nonfinite_is_failure_not_scored_episode(self):
        w = Wrapper.__new__(Wrapper)
        w.token_mode = 'raw'
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            w._decode_body_29(torch.full((1, 64), float('nan')))


if __name__ == '__main__':
    unittest.main()
