"""Two simultaneous startup batches must not allocate the same memory twice."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipIf(os.name == "nt", "Linux fcntl allocator")
class ReservationTests(unittest.TestCase):
    def test_pending_batches_share_capacity_without_oversubscription(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home/'kevin/dline').mkdir(parents=True)
            batches = []
            for _ in range(3):
                out = io.StringIO()
                with patch.object(Path,'home',return_value=home), patch.object(sys,'argv',
                    ['gpu_slots.py','reserve','--owner',str(os.getpid()),'--count','4']), patch(
                    'subprocess.check_output',return_value='0, 570, 49140\n1, 570, 49140\n2, 570, 49140\n'), contextlib.redirect_stdout(out):
                    runpy.run_path(str(Path(__file__).with_name('gpu_slots.py')),run_name='__main__')
                batches.append(out.getvalue().split())
            self.assertEqual([len(x) for x in batches],[4,4,0])
            entries=json.loads((home/'kevin/dline/gpu_slots.json').read_text())
            for gpu in range(3):
                self.assertLessEqual(sum(x['mb'] for x in entries if x['gpu']==gpu),49140-2570)


if __name__ == '__main__':
    unittest.main()
