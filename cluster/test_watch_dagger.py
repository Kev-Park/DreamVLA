"""Recovery guards: never resume over live dependent work or an abort."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import watch_dagger as watcher


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state = {tag: {"env": {"TAG": tag, "NR": "5", "CK": "checkpoint",
                                   "REFS": "train", "EVREFS": "eval", "BASE": "collect"}}
                      for tag in watcher.LINES}

    def tearDown(self):
        self.temp.cleanup()

    def test_live_workers_prevent_chain_restart(self):
        worker = {"pid": 7, "start": "1", "ticks": 10, "io": 20,
                  "args": "python /kevin/datasets/af60v8f/converter.py"}
        with patch.object(watcher, "processes", return_value=[worker]), patch.object(watcher, "run") as run:
            for n in range(3):
                watcher.cluster_tick(self.root, self.state, 100 + n)
            self.assertEqual(self.state["af60v8f"]["missing"], 0)
            self.assertFalse(any("dline_v8_kevin" in str(c) for c in run.call_args_list))

    def test_abort_prevents_restart(self):
        for tag in watcher.LINES:
            p = self.root / "dline" / tag / "abort"
            p.parent.mkdir(parents=True)
            p.touch()
        with patch.object(watcher, "processes", return_value=[]), patch.object(watcher, "run") as run:
            for n in range(3):
                watcher.cluster_tick(self.root, self.state, 100 + n)
            run.assert_not_called()

    def test_restart_requires_two_observations_and_idle_pane(self):
        from subprocess import CompletedProcess
        with patch.object(watcher, "processes", return_value=[]), patch.object(
                watcher, "run", return_value=CompletedProcess([], 0, "bash\n", "")) as run:
            watcher.cluster_tick(self.root, self.state, 100)
            run.assert_not_called()
            report = watcher.cluster_tick(self.root, self.state, 160)
            self.assertEqual(sum("safe resume" in e for e in report["events"]), 2)
            self.assertEqual(sum(c.args[0][1] == "send-keys" for c in run.call_args_list), 2)

    def test_io_activity_resets_stall_timer(self):
        worker = {"pid": 7, "start": "1", "ticks": 10, "io": 20,
                  "args": "python /kevin/datasets/af60v8f/converter.py"}
        with patch.object(watcher, "processes", return_value=[worker]), patch.object(watcher, "run"):
            watcher.cluster_tick(self.root, self.state, 100)
            worker["io"] += 1
            report = watcher.cluster_tick(self.root, self.state, 2000)
            self.assertEqual(report["lines"]["af60v8f"]["inactive_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
