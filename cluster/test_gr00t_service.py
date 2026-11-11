"""CPU loopback tests for RNG, diagnostic and action parity across processes."""
import ast
import multiprocessing
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/'Training/scripts/reinforcement_learning/rsl_rl'
sys.path.insert(0, str(SCRIPTS))
from vla_sonic.gr00t_service import PolicyService, PolicyServiceClient, capture_rng, restore_rng


class FixturePolicy:
    def __init__(self):
        self.model = SimpleNamespace(backbone=torch.nn.Identity())

    def get_action(self, observation, options=None):
        image = observation['video']['ego_view']
        state = observation['state']['joint_pos']
        assert image.dtype == np.uint8 and image.shape == (1, 1, 4, 4, 3)
        assert state.dtype == np.float32 and state.shape == (1, 1, 4)
        assert image.flags.writeable and state.flags.writeable
        assert observation['language']['task'] == [['pick']]
        features = self.model.backbone(torch.from_numpy(state.copy()))
        noise = torch.randn(1, 8, 4)
        mixed = noise+float(np.random.rand())+random.random()+features
        return {'motion_token': mixed.detach().numpy()}, {'options': options}

    def reset(self, options=None):
        return {'reset': True, 'options': options, 'sample': float(torch.rand(()))}


def serve_fixture(checkpoint, queue):
    torch.set_num_threads(1)
    PolicyService(FixturePolicy(), checkpoint, 'cpu', 'fixture').serve(queue.put)


class InferenceServiceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.directory.name)
        (cls.checkpoint/'config.json').write_text('{"model_type":"fixture"}')
        context = multiprocessing.get_context('spawn')
        cls.queue = context.Queue()
        cls.process = context.Process(target=serve_fixture, args=(str(cls.checkpoint), cls.queue))
        cls.process.start()
        try:
            cls.descriptor = cls.queue.get(timeout=30)
        except BaseException:
            if cls.process.is_alive():
                cls.process.terminate()
            cls.process.join(timeout=10)
            cls.queue.close()
            cls.queue.join_thread()
            cls.directory.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            with cls.client() as client:
                client.shutdown_server()
            cls.process.join(timeout=10)
        finally:
            if cls.process.is_alive():
                cls.process.terminate()
                cls.process.join(timeout=10)
            cls.queue.close()
            cls.queue.join_thread()
            cls.directory.cleanup()

    @classmethod
    def client(cls, descriptor=None):
        return PolicyServiceClient(descriptor or cls.descriptor, cls.checkpoint, 'cpu', 'fixture', timeout_ms=3000)

    def observation(self):
        return {'video': {'ego_view': np.zeros((1, 1, 4, 4, 3), np.uint8)},
                'state': {'joint_pos': np.arange(4, dtype=np.float32).reshape(1, 1, 4)},
                'language': {'task': [['pick']]}}

    def test_action_and_all_rng_advancement_match_in_process(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        start = capture_rng('cpu')
        expected, expected_info = FixturePolicy().get_action(self.observation(), {'test': 1})
        expected_next = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(start, 'cpu')
        with self.client() as client:
            action, info = client.get_action(self.observation(), {'test': 1})
        np.testing.assert_array_equal(action['motion_token'], expected['motion_token'])
        self.assertTrue(action['motion_token'].flags.writeable)
        self.assertEqual(info, expected_info)
        self.assertEqual(random.random(), expected_next[0])
        self.assertEqual(np.random.rand(), expected_next[1])
        torch.testing.assert_close(torch.rand(3), expected_next[2], rtol=0, atol=0)

    def test_reseeding_reproduces_action_and_diagnostics_survive(self):
        with self.client() as client:
            client.enable_backbone_features()
            torch.manual_seed(15)
            start = capture_rng('cpu')
            first = client.get_action(self.observation())[0]
            feature = client.last_backbone_feature.copy()
            restore_rng(start, 'cpu')
            second = client.get_action(self.observation())[0]
        np.testing.assert_array_equal(first['motion_token'], second['motion_token'])
        np.testing.assert_array_equal(feature, np.arange(4, dtype=np.float16))
        self.assertEqual(feature.dtype, np.float16)

    def test_reset_preserves_contract_and_rng(self):
        torch.manual_seed(21)
        state = capture_rng('cpu')
        expected = FixturePolicy().reset({'reset_test': 2})
        next_sample = torch.rand(2)
        restore_rng(state, 'cpu')
        with self.client() as client:
            result = client.reset({'reset_test': 2})
        self.assertEqual(result, expected)
        torch.testing.assert_close(torch.rand(2), next_sample, rtol=0, atol=0)

    def test_actual_collector_query_receives_remote_backbone_features(self):
        source = SCRIPTS/'collect_sonic_adapter.py'
        original = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.ClassDef) and n.name == 'DaggerDriver')
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
                   and n.name in {'_install_backbone_hook', '_query_vla'}]
        node = ast.ClassDef(name='DriverQueries', bases=[], keywords=[], body=methods, decorator_list=[])
        scope = {'torch': torch, 'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(source), 'exec'), scope)
        with self.client() as client:
            driver = scope['DriverQueries']()
            driver.vla = client
            driver.obs_adapter = self.observation
            driver._feat = None
            driver._install_backbone_hook()
            action = driver._query_vla()
            self.assertIn('motion_token', action)
            np.testing.assert_array_equal(driver._feat, np.arange(4, dtype=np.float16))

    def test_wrong_checkpoint_is_rejected_before_inference(self):
        descriptor = {**self.descriptor, 'checkpoint': {**self.descriptor['checkpoint'], 'config_sha256': 'wrong'}}
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            self.client(descriptor)

    def test_wrong_session_token_cannot_use_model(self):
        descriptor = {**self.descriptor, 'api_token': 'wrong'}
        with self.assertRaisesRegex(RuntimeError, 'Unauthorized'):
            self.client(descriptor)

    def test_policy_failure_is_explicit_and_service_can_answer_next_request(self):
        with self.client() as client:
            with self.assertRaises(RuntimeError):
                client.get_action({})
            self.assertIn('motion_token', client.get_action(self.observation())[0])


if __name__ == '__main__':
    unittest.main()
