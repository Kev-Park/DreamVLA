"""Short CPU optimizer/save/resume gate for the actual residual actor class.

Run in the reviewed cluster environment, not on the local workstation.
This does not substitute for the simulator-connected PPO training smoke test.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from tensordict import TensorDict

SCRIPTS = Path(__file__).resolve().parents[1]/'Training/scripts/reinforcement_learning/rsl_rl'
sys.path.insert(0, str(SCRIPTS))
from vla_sonic.adapter_actor_critic import AdapterActorCritic


class AdapterCheckpoint(unittest.TestCase):
    def test_zero_initialization_optimizer_update_and_exact_resume(self):
        torch.manual_seed(17)
        obs = TensorDict({'policy': torch.randn(8, 32), 'critic': torch.randn(8, 48)},
                         batch_size=[8])
        groups = {'policy': ['policy'], 'critic': ['critic']}

        def make():
            model = AdapterActorCritic(obs, groups, 65, actor_hidden_dims=[256, 128],
                                       critic_hidden_dims=[512, 256, 256], activation='elu')
            return model, torch.optim.Adam(model.parameters(), lr=1e-4)

        def update(model, optimizer):
            prediction = model.act_inference(obs)
            value = model.evaluate(obs)
            loss = (prediction-.1).square().mean()+(value-.2).square().mean()
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()

        model, optimizer = make()
        torch.testing.assert_close(model.act_inference(obs), torch.zeros(8, 65), rtol=0, atol=0)
        self.assertEqual(tuple(model.act(obs).shape), (8, 65))
        for _ in range(3):
            update(model, optimizer)
        self.assertGreater(float(model.act_inference(obs).abs().max()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'iter': 3}, path)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            restored, restored_optimizer = make()
            restored.load_state_dict(saved['model_state_dict'], strict=True)
            restored_optimizer.load_state_dict(saved['optimizer_state_dict'])
        torch.testing.assert_close(model.act_inference(obs), restored.act_inference(obs), rtol=0, atol=0)
        update(model, optimizer)
        update(restored, restored_optimizer)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
