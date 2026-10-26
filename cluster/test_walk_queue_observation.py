"""Slurm lookup failures must not masquerade as terminal or transport failures."""
import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import patch

import walk_late_takeover_bridge as bridge


def execute_remote(host, program):
    assert host == 'adroit'
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            exec(program, {})
        except SystemExit as exc:
            assert exc.code == 0
    return json.loads(output.getvalue())


class QueueObservationTests(unittest.TestCase):
    def observe(self, replies):
        with patch.object(bridge, 'call', side_effect=execute_remote), \
                patch.object(subprocess, 'run', side_effect=replies) as run:
            result = bridge.job_state('3431651')
            commands = [args.args[0] for args in run.call_args_list]
            self.assertTrue(all(cmd[0] in ('squeue', 'sacct') for cmd in commands))
            self.assertTrue(all('3431651' in cmd for cmd in commands))
            self.assertTrue(all(args.kwargs['timeout'] == 20 for args in run.call_args_list))
            return result, commands

    def test_live_pending_job(self):
        state, commands = self.observe([subprocess.CompletedProcess([], 0, 'PENDING|(Priority)\n', '')])
        self.assertEqual(state['state'], 'PENDING')
        self.assertEqual(len(commands), 1)

    def test_queue_timeout_is_incomplete_observation(self):
        state, _ = self.observe([subprocess.TimeoutExpired('squeue', 20)])
        self.assertEqual(state['state'], 'OBSERVATION_TIMEOUT')
        self.assertEqual(state['command'], 'squeue')

    def test_accounting_timeout_is_incomplete_observation(self):
        state, _ = self.observe([subprocess.CompletedProcess([], 0, '', ''),
                                 subprocess.TimeoutExpired('sacct', 20)])
        self.assertEqual(state['state'], 'OBSERVATION_TIMEOUT')
        self.assertEqual(state['command'], 'sacct')

    def test_slurm_error_does_not_assert_job_failed(self):
        state, _ = self.observe([subprocess.CompletedProcess([], 1, '', 'controller unavailable')])
        self.assertEqual(state['state'], 'OBSERVATION_ERROR')

    def test_accounting_completion_preserves_exit_status(self):
        state, _ = self.observe([subprocess.CompletedProcess([], 0, '', ''),
                                 subprocess.CompletedProcess([], 0, 'COMPLETED|0:0\n', '')])
        self.assertEqual(state, {'state': 'COMPLETED', 'exitcode': '0:0'})

    def test_accounting_pending_is_not_failure(self):
        state, _ = self.observe([subprocess.CompletedProcess([], 0, '', ''),
                                 subprocess.CompletedProcess([], 0, '', '')])
        self.assertEqual(state['state'], 'ACCOUNTING_PENDING')

    def test_real_job_failure_remains_terminal(self):
        state, _ = self.observe([subprocess.CompletedProcess([], 0, '', ''),
                                 subprocess.CompletedProcess([], 0, 'FAILED|1:0\n', '')])
        self.assertEqual(state, {'state': 'FAILED', 'exitcode': '1:0'})


if __name__ == '__main__':
    unittest.main()
