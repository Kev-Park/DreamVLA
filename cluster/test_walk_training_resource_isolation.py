import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from walk_late_takeover_bridge import observe_training_resources


class TrainingResourceIsolationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_held_berkeley_connection_reports_attention_without_aborting_training_observation(self):
        (self.root / 'biped.connection_hold.json').write_text('{}')
        probe = Mock(side_effect=RuntimeError('Connection held pending operator review'))
        observe_training_resources(probe, self.root)
        probe.assert_called_once_with()
        attention = json.loads((self.root / 'resource_attention.json').read_text())
        self.assertEqual(attention['held_hosts'], ['biped'])
        self.assertTrue(attention['alerts'])

    def test_unclassified_observation_error_propagates(self):
        probe = Mock(side_effect=RuntimeError('Unexpected failure'))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected failure'):
            observe_training_resources(probe, self.root)

    def test_adroit_connection_failure_is_never_downgraded(self):
        for host in ('biped', 'adroit'):
            (self.root / (host + '.connection_hold.json')).write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Connection failure'):
            observe_training_resources(Mock(side_effect=RuntimeError('Connection failure')), self.root)


if __name__ == '__main__':
    unittest.main()
