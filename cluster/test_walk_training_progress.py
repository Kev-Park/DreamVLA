"""Validate progress tracking across loading bars, updates and stalled logs."""
import tempfile
from pathlib import Path
import unittest

from walk_training_progress import parse_progress,read_progress,watch_progress


class TrainingProgressTests(unittest.TestCase):
    def test_training_bar_ignores_different_loading_total(self):
        text='298/109629 [01:32<8:06:56, 3.74it/s]\r2/2 [00:01<00:00, 2.00it/s]'
        value=parse_progress(text,109629)
        self.assertEqual(value['step'],298)
        self.assertEqual(value['reported_remaining_seconds'],29216)
        self.assertEqual(value['updates_per_second'],3.74)

    def test_seconds_per_update_is_inverted(self):
        self.assertEqual(parse_progress('1/10 [00:02<00:18, 2.00s/it]',10)['updates_per_second'],.5)

    def test_missing_progress_is_not_completion(self):
        self.assertEqual(parse_progress('loading pretrained weights',109629),{})

    def test_reading_log_is_bounded_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root,'run.log');payload=b'loading\n'*10000+b'400/109629 [02:00<8:00:00, 3.70it/s]'
            path.write_bytes(payload);value=read_progress(path,109629)
            self.assertEqual(value['step'],400);self.assertEqual(path.read_bytes(),payload)

    def test_advancing_updates_clear_stall_attention(self):
        old={'first_running_observed_at':0,'last_advanced_at':0,'step':100,'attention':'stalled'}
        new=watch_progress(old,{'step':101},2000)
        self.assertEqual(new['last_advanced_at'],2000);self.assertNotIn('attention',new)

    def test_stalled_steps_only_report_attention(self):
        new=watch_progress({'last_advanced_at':0,'step':100},{'step':100},1801)
        self.assertEqual(new['step'],100);self.assertIn('attention',new)

    def test_missing_initial_step_eventually_reports_attention(self):
        first=watch_progress(None,{},100)
        self.assertIn('attention',watch_progress(first,{},1901))


if __name__=='__main__':unittest.main()
