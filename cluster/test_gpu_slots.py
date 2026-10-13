"""Memory reservations and GPU spreading must remain safe across batches."""
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
from gpu_slots import select_slots, reservation_paused


@unittest.skipIf(os.name == 'nt', 'Linux fcntl allocator')
class ReservationTests(unittest.TestCase):
    def test_pending_batches_share_capacity_without_oversubscription(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)
            (home/'kevin/dline').mkdir(parents=True)
            batches=[]
            for _ in range(3):
                out=io.StringIO()
                with patch.object(Path,'home',return_value=home), patch.object(sys,'argv',
                    ['gpu_slots.py','reserve','--owner',str(os.getpid()),'--count','4']), patch(
                    'subprocess.check_output',return_value='0, 570, 49140\n1, 570, 49140\n2, 570, 49140\n'), contextlib.redirect_stdout(out):
                    runpy.run_path(str(Path(__file__).with_name('gpu_slots.py')),run_name='__main__')
                batches.append(out.getvalue().split())
            self.assertEqual([len(x) for x in batches],[4,4,0])
            entries=json.loads((home/'kevin/dline/gpu_slots.json').read_text())
            for gpu in range(3):
                self.assertLessEqual(sum(x['mb'] for x in entries if x['gpu']==gpu),49140-2570)


class SelectionTests(unittest.TestCase):
    def test_pause_applies_only_to_exact_owner_generation_and_request(self):
        pauses=[{'owner':123,'owner_start':'456','count':1,'mb':45000}]
        self.assertTrue(reservation_paused(pauses,123,'456',1,45000))
        for owner,start,count,mb in [(124,'456',1,45000),(123,'457',1,45000),(123,'456',4,17000)]:
            self.assertFalse(reservation_paused(pauses,owner,start,count,mb))
    def test_separate_gpus_before_sharing(self):
        rows=[[i,1000,49140] for i in range(4)]
        self.assertEqual(select_slots(rows,[],4,17000),[0,1,2,3])

    def test_fallback_sharing_obeys_memory(self):
        self.assertEqual(select_slots([[0,1000,49140],[1,1000,49140]],[],4,17000),[0,1,0,1])
        self.assertEqual(select_slots([[0,1000,49140]],[],4,17000),[])

    def test_pending_training_reservation_is_respected(self):
        rows=[[0,1000,49140],[1,1000,49140]]
        pending=[{'gpu':0,'mb':45000}]
        self.assertEqual(select_slots(rows,pending,2,17000),[1,1])

    def test_legacy_at_least_four_shard_contract(self):
        rows=[[i,1000,49140] for i in range(2)]
        self.assertEqual(select_slots(rows,[],8,17000),[0,1,0,1])


if __name__=='__main__':unittest.main()
