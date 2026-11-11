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
from gpu_slots import select_slots, reservation_paused, reviewed_placement, placed_slots


def gpu_xml(rows):
    """Model the NVIDIA XML snapshot used by both allocation paths."""
    return '<nvidia_smi_log>' + ''.join(
        f'<gpu><fb_memory_usage><used>{used} MiB</used><total>{total} MiB</total>'
        f'</fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization>'
        f'<processes /></gpu>' for _, used, total in rows) + '</nvidia_smi_log>'


@unittest.skipIf(os.name == 'nt', 'Linux fcntl allocator')
class ReservationTests(unittest.TestCase):
    def test_explicit_job_placement_is_reserved_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);work=home/'kevin/dline';work.mkdir(parents=True)
            owner=os.getpid()
            start=Path(f'/proc/{owner}/stat').read_text().rsplit(')',1)[1].split()[19]
            (work/'gpu_reservation_placements.json').write_text(json.dumps([
                {'owner':owner,'owner_start':start,'count':4,'mb':17000,'gpus':[2,4,7,2]}]))
            outputs=[]
            for _ in range(2):
                out=io.StringIO()
                with patch.object(Path,'home',return_value=home), patch.object(sys,'argv',
                    ['gpu_slots.py','reserve','--owner',str(owner),'--count','4','--mb','17000']), patch(
                    'subprocess.check_output',return_value=gpu_xml([(i, 1000, 49140) for i in range(10)])), contextlib.redirect_stdout(out):
                    runpy.run_path(str(Path(__file__).with_name('gpu_slots.py')),run_name='__main__')
                outputs.append(out.getvalue().split())
            self.assertEqual(outputs,[['2','4','7','2'],[]])
            entries=json.loads((work/'gpu_slots.json').read_text())
            self.assertEqual([e['gpu'] for e in entries],[2,4,7,2])

    def test_pending_batches_share_capacity_without_oversubscription(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)
            (home/'kevin/dline').mkdir(parents=True)
            batches=[]
            for _ in range(3):
                out=io.StringIO()
                with patch.object(Path,'home',return_value=home), patch.object(sys,'argv',
                    ['gpu_slots.py','reserve','--owner',str(os.getpid()),'--count','4']), patch(
                    'subprocess.check_output',return_value=gpu_xml([(i, 570, 49140) for i in range(3)])), contextlib.redirect_stdout(out):
                    runpy.run_path(str(Path(__file__).with_name('gpu_slots.py')),run_name='__main__')
                batches.append(out.getvalue().split())
            self.assertEqual([len(x) for x in batches],[4,4,0])
            entries=json.loads((home/'kevin/dline/gpu_slots.json').read_text())
            for gpu in range(3):
                self.assertLessEqual(sum(x['mb'] for x in entries if x['gpu']==gpu),49140-2570)


class SelectionTests(unittest.TestCase):
    def test_reviewed_placement_matches_only_exact_job_request(self):
        placement = [{'owner':123,'owner_start':'456','count':4,'mb':17000,'gpus':[2,4,7,2]}]
        self.assertEqual(reviewed_placement(placement,123,'456',4,17000),[2,4,7,2])
        for owner,start,count,mb in [(124,'456',4,17000),(123,'457',4,17000),
                                     (123,'456',1,17000),(123,'456',4,45000)]:
            self.assertIsNone(reviewed_placement(placement,owner,start,count,mb))
        with self.assertRaises(ValueError):
            reviewed_placement(placement+placement,123,'456',4,17000)

    def test_reviewed_placement_packs_workers_and_checks_pending_memory(self):
        rows = [[gpu,1000,49140] for gpu in range(10)]
        self.assertEqual(placed_slots(rows,[],[2,4,7,2],17000),[2,4,7,2])
        self.assertEqual(placed_slots(rows,[{'gpu':2,'mb':20000}],[2,4,7,2],17000),[])
        with self.assertRaises(ValueError):
            placed_slots(rows,[],[10],17000)

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




"""Allocation invariants without a GPU or third-party packages."""
import unittest
from gpu_slots import choose_gpus, observed_reserved_memory, pending_reserved_mb


class Reservations(unittest.TestCase):
    def test_live_allocation_preserves_unused_headroom(self):
        entries = [dict(owner=10, gpu=4, mb=24000)]
        observed = observed_reserved_memory([(4, 11, 15000)], {11: 10}, entries)
        self.assertEqual(pending_reserved_mb(4, entries, observed), 9000)
        self.assertEqual(choose_gpus([[4, 15000, 48000, 25]], entries, 1, 20000, 4, observed), [4])
        self.assertEqual(choose_gpus([[4, 15000, 48000, 25]], entries, 1, 24000, 4, observed), [])

    def test_pending_startup_remains_fully_reserved(self):
        entries = [dict(owner=10, gpu=4, mb=24000), dict(owner=20, gpu=4, mb=20000)]
        observed = {(4, 10): 15000}
        self.assertEqual(pending_reserved_mb(4, entries, observed), 29000)
        self.assertEqual(choose_gpus([[4, 15000, 48000, 25]], entries, 1, 5000, 4, observed), [])

    def test_unknown_ancestry_is_conservative(self):
        entries = [dict(owner=10, gpu=4, mb=24000)]
        self.assertEqual(observed_reserved_memory([(4, 11, 15000)], {}, entries), {})
        self.assertEqual(pending_reserved_mb(4, entries, {}), 24000)

    def test_nested_owners_and_duplicate_reports_cannot_double_credit(self):
        entries = [dict(owner=10, gpu=4, mb=24000), dict(owner=11, gpu=4, mb=20000)]
        observed = observed_reserved_memory([(4, 12, 15000), (4, 12, 15000)], {12: 11, 11: 10}, entries)
        self.assertEqual(observed, {(4, 11): 15000})
        self.assertEqual(pending_reserved_mb(4, entries, observed), 29000)

    def test_multiple_reservations_for_one_owner_are_summed(self):
        entries = [dict(owner=10, gpu=4, mb=12000), dict(owner=10, gpu=4, mb=12000)]
        self.assertEqual(pending_reserved_mb(4, entries, {(4, 10): 15000}), 9000)

    def test_other_gpu_or_process_cannot_pay_for_reservation(self):
        entries = [dict(owner=10, gpu=4, mb=24000)]
        observed = observed_reserved_memory([(0, 11, 15000), (4, 99, 10000)], {11: 10}, entries)
        self.assertEqual(pending_reserved_mb(4, entries, observed), 24000)

    def test_growth_above_target_never_creates_a_negative_reservation(self):
        self.assertEqual(pending_reserved_mb(4, [dict(owner=10, gpu=4, mb=20000)], {(4, 10): 25000}), 0)


if __name__ == '__main__':
    unittest.main()
