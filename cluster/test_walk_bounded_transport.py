"""Isolation and legacy-hook checks for one-CPU, dedicated SSH transfers."""
import builtins
from types import SimpleNamespace
import shlex
import unittest
from unittest.mock import patch

import walk_adroit_bridge as bridge


class FakeProcesses:
    def __init__(self, parent_uid=1008, parent_comm='sshd'):
        self.parent_uid, self.parent_comm = parent_uid, parent_comm
        self.affinity = {100: {0, 1}, 200: {0, 1}, 201: {0, 1}}
        self.set_calls, self.exec_calls = [], []

    def path(self, *parts):
        owner = self
        text = '/'.join(str(part).strip('/') for part in parts)
        class Path:
            def stat(self):
                return SimpleNamespace(st_uid=owner.parent_uid)
            def __truediv__(self, suffix):
                return owner.path(text, suffix)
            def read_text(self):
                return owner.parent_comm
            def iterdir(self):
                pid = int(text.split('/')[1])
                return [SimpleNamespace(name=str(tid)) for tid in ((200, 201) if pid == 200 else (100,))]
        return Path()

    def set_affinity(self, pid, mask):
        self.set_calls.append((pid, mask))
        self.affinity[pid] = mask

    def execute_wrapper(self, command):
        fake_os = SimpleNamespace(getpid=lambda: 100, getppid=lambda: 200, getuid=lambda: 1008,
            sched_getaffinity=lambda pid: self.affinity[100 if pid == 0 else pid],
            sched_setaffinity=self.set_affinity,
            execvp=lambda program, args: self.exec_calls.append((program, args)))
        real_import = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'os':
                return fake_os
            if name == 'pathlib':
                return SimpleNamespace(Path=self.path)
            return real_import(name, *args, **kwargs)
        wrapper = shlex.split(bridge.bounded_transport_command(command))
        with patch('builtins.__import__', side_effect=imports):
            exec(compile(wrapper[2], '<dedicated-transport>', 'exec'), {})


class BoundedTransportTests(unittest.TestCase):
    def test_only_own_session_threads_and_command_are_pinned(self):
        fake = FakeProcesses()
        command = "tar -C ~/kevin -cf - -- 'dataset with spaces'"
        fake.execute_wrapper(command)
        self.assertEqual(fake.set_calls, [(200, {0}), (201, {0}), (100, {0})])
        self.assertEqual(fake.exec_calls, [('bash', ['bash', '-c', command])])

    def test_privileged_or_foreign_parent_is_untouched(self):
        for uid in (0, 9999):
            fake = FakeProcesses(parent_uid=uid)
            with self.assertRaisesRegex(AssertionError, 'privileged or foreign'):
                fake.execute_wrapper('tar -xf -')
            self.assertEqual(fake.set_calls, [])
            self.assertEqual(fake.exec_calls, [])

    def test_non_ssh_parent_is_untouched(self):
        fake = FakeProcesses(parent_comm='tmux')
        with self.assertRaisesRegex(AssertionError, 'Dedicated SSH session'):
            fake.execute_wrapper('tar -xf -')
        self.assertEqual(fake.set_calls, [])

    def test_bounded_connection_explicitly_disables_multiplexing(self):
        args = bridge.ssh('bluesclues', 'tar -xf -', single_cpu=True)
        self.assertIn('ControlMaster=no', args)
        self.assertIn('ControlPath=none', args)

    def test_shared_adroit_transport_cannot_be_bounded(self):
        with self.assertRaisesRegex(AssertionError, 'shared Adroit'):
            bridge.ssh('adroit', 'tar -xf -', single_cpu=True)

    def test_default_inventory_preserves_old_call_hook_signature(self):
        def old_hook(host, script, timeout=90):
            self.assertEqual((host, timeout), ('biped', 1800))
            return ['unchanged-hook']
        with patch.object(bridge, 'call', old_hook):
            self.assertEqual(bridge.inventory('biped', 'dataset'), ['unchanged-hook'])


if __name__ == '__main__':
    unittest.main()
