"""Serialized policy output/state contracts; run remotely in the target CPU env."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'vendored_policy_exporter',
    ROOT/'Training/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py')
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class PolicyExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def feedforward(self, normalizer):
        actor = torch.nn.Sequential(torch.nn.Linear(12, 16), torch.nn.ELU(),
                                    torch.nn.Linear(16, 4))
        policy = SimpleNamespace(actor=actor, is_recurrent=False)
        x = torch.sin(torch.arange(96, dtype=torch.float32).reshape(8, 12)*.013)
        with TemporaryDirectory() as directory:
            EXPORTER.export_policy_as_jit(policy, normalizer, directory)
            loaded = torch.jit.load(str(Path(directory)/'policy.pt')).eval()
            with torch.inference_mode():
                expected = actor(x if normalizer is None else normalizer(x))
                torch.testing.assert_close(loaded(x), expected, rtol=1e-5, atol=1e-6)

    def test_feedforward(self):
        self.feedforward(None)

    def test_feedforward_normalizer(self):
        self.feedforward(torch.nn.LayerNorm(12))

    def recurrent(self, rnn_type):
        rnn = rnn_type(12, 16, num_layers=2)
        actor = torch.nn.Sequential(torch.nn.Linear(16, 4), torch.nn.Tanh())
        policy = SimpleNamespace(actor=actor, is_recurrent=True,
                                 memory_a=SimpleNamespace(rnn=rnn))
        sequence = torch.sin(torch.arange(60, dtype=torch.float32).reshape(5, 1, 12)*.013)
        with TemporaryDirectory() as directory:
            EXPORTER.export_policy_as_jit(policy, None, directory)
            loaded = torch.jit.load(str(Path(directory)/'policy.pt')).eval()
            with torch.inference_mode():
                reference, _ = rnn(sequence)
                expected = actor(reference)
                actual = torch.stack([loaded(x) for x in sequence])
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                # Reset must restore the first action, not retain the previous
                # episode's hidden/cell state or merely expose a no-op method.
                loaded.reset()
                torch.testing.assert_close(loaded(sequence[0]), expected[0],
                                           rtol=1e-5, atol=1e-6)

    def test_lstm_state_and_reset(self):
        self.recurrent(torch.nn.LSTM)

    def test_gru_state_and_reset(self):
        self.recurrent(torch.nn.GRU)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()

