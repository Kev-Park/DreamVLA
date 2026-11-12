"""ONNX serialization contracts in the isolated project runtime, CPU provider only."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

from test_policy_exporter import EXPORTER


class OnnxPolicyExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def session(self, path):
        onnx.checker.check_model(str(path), full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        self.assertEqual(session.get_providers(), ['CPUExecutionProvider'])
        return session

    def feedforward(self, normalizer):
        actor = torch.nn.Sequential(torch.nn.Linear(12, 16), torch.nn.ELU(),
                                    torch.nn.Linear(16, 4))
        policy = SimpleNamespace(actor=actor, is_recurrent=False)
        x = torch.sin(torch.arange(96, dtype=torch.float32).reshape(8, 12)*.013)
        with TemporaryDirectory() as directory:
            EXPORTER.export_policy_as_onnx(policy, directory, normalizer)
            session = self.session(Path(directory)/'policy.onnx')
            with torch.inference_mode():
                expected = actor(x if normalizer is None else normalizer(x)).numpy()
            actual = np.concatenate([session.run(None, {'obs': row[None].numpy()})[0]
                                     for row in x])
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

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
            EXPORTER.export_policy_as_onnx(policy, directory)
            session = self.session(Path(directory)/'policy.onnx')
            with torch.inference_mode():
                reference, final_state = rnn(sequence)
                expected = actor(reference).numpy()
            inputs = {'h_in': np.zeros((2, 1, 16), dtype=np.float32)}
            if rnn_type is torch.nn.LSTM:
                inputs['c_in'] = np.zeros((2, 1, 16), dtype=np.float32)
            actions = []
            for row in sequence:
                values = session.run(None, dict(inputs, obs=row.numpy()))
                outputs = dict(zip([value.name for value in session.get_outputs()], values))
                actions.append(outputs['actions'])
                inputs['h_in'] = outputs['h_out']
                if 'c_in' in inputs:
                    inputs['c_in'] = outputs['c_out']
            np.testing.assert_allclose(np.stack(actions), expected, rtol=1e-4, atol=1e-5)
            hidden = final_state[0] if rnn_type is torch.nn.LSTM else final_state
            np.testing.assert_allclose(inputs['h_in'], hidden.numpy(), rtol=1e-4, atol=1e-5)
            if 'c_in' in inputs:
                np.testing.assert_allclose(inputs['c_in'], final_state[1].numpy(),
                                           rtol=1e-4, atol=1e-5)

    def test_lstm_state(self):
        self.recurrent(torch.nn.LSTM)

    def test_gru_state(self):
        self.recurrent(torch.nn.GRU)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
