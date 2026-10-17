"""The waiting original controller must not mask a failed stage bookkeeper."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from watch_walk_gr00t import stage_only_controllers


class ControllerHealth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.work = root/'work'
        self.proc = root/'proc'
        self.work.mkdir()
        (self.proc/'sys/kernel/random').mkdir(parents=True)
        (self.proc/'sys/kernel/random/boot_id').write_text('boot-a')
        self.payload = 'print("one stage")'
        self.review = {'at': 100, 'boot_id': 'boot-a',
                       'payload_sha256': hashlib.sha256(self.payload.encode()).hexdigest()}
        self.write('round2_single_stage_operator_review.json', self.review)
        self.launch = {'pid': 123, 'run': 'study_r2', 'steps': 99,
                       'operator_review': self.review}
        self.write('round2_single_stage.launch.json', self.launch)
        (self.proc/'123').mkdir()
        self.cmdline = self.proc/'123/cmdline'
        self.cmdline.write_bytes(('\0'.join([sys.executable, '-u', '-c', self.payload])+'\0').encode())

    def write(self, name, value):
        (self.work/name).write_text(json.dumps(value))

    def check(self, now=110):
        return stage_only_controllers(self.work, self.proc, now)[0]

    def test_controller_exit_is_visible_while_other_processes_remain(self):
        self.assertTrue(self.check()['controller_live'])
        (self.proc/'999').mkdir()  # An original waiting chain is still alive.
        (self.proc/'999/cmdline').write_bytes(b'python\0waiting-chain\0')
        self.cmdline.unlink()
        self.assertFalse(self.check()['controller_live'])
        self.assertTrue(self.check()['alerts'])

    def test_recycled_pid_or_reboot_does_not_count_as_live(self):
        self.cmdline.write_bytes(b'python\0-u\0-c\0different job\0')
        self.assertFalse(self.check()['controller_live'])
        self.cmdline.write_bytes(('\0'.join([sys.executable, '-u', '-c', self.payload])+'\0').encode())
        (self.proc/'sys/kernel/random/boot_id').write_text('boot-b')
        self.assertFalse(self.check()['controller_live'])

    def test_reviewed_launch_cannot_disappear_silently(self):
        (self.work/'round2_single_stage.launch.json').unlink()
        self.assertFalse(self.check(now=110)['alerts'])
        self.assertTrue(self.check(now=230)['alerts'])

    def test_abort_and_uncorroborated_completion_are_reported(self):
        self.write('round2_single_stage.abort.json', {'error': 'trainer failed'})
        self.assertEqual(self.check()['status'], 'failed')
        (self.work/'round2_single_stage.abort.json').unlink()
        final = {'run': 'study_r2', 'steps': 99, 'checkpoint': '/saved/checkpoint-99'}
        self.write('round2_single_stage.done.json', final)
        self.assertTrue(self.check()['alerts'])
        self.write('study_r2.trained.json', final)
        self.assertEqual(self.check()['status'], 'completed')
        self.assertFalse(self.check()['alerts'])


if __name__ == '__main__':
    unittest.main()
