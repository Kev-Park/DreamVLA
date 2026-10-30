"""Regression checks for exited processes and stale shared-GPU observations."""
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock, Mock
from types import SimpleNamespace

import pick_original_bluesclues_queue as queue
from resume_pick200_resource_recovery import acquire_slots, verify_completed_training, TAG


def snapshot(cpu=.10, free=(4,), occupied=(), complete=True):
    return dict(cpu_fraction=cpu, free=list(free), occupied=list(occupied),
                cpu_measurement_complete=complete)


class CapacityTests(unittest.TestCase):
    def test_gpu_probe_failure_cannot_authorize_allocation_or_abort_wait(self):
        for error in [subprocess.TimeoutExpired(['nvidia-smi', '-q', '-x'], 10),
                      subprocess.CalledProcessError(1, ['nvidia-smi']),
                      OSError('GPU driver unavailable')]:
            with self.subTest(error=type(error).__name__), \
                    patch.object(queue, 'account_cpu_snapshot', return_value={}), \
                    patch.object(queue, 'account_cpu_fraction', return_value=.01), \
                    patch.object(queue.time, 'sleep'), \
                    patch.object(queue.subprocess, 'check_output', side_effect=error):
                current = queue.capacity()
            self.assertEqual(current['free'], [])
            self.assertFalse(current['cpu_measurement_complete'])
            self.assertIn('gpu_observation_error', current)
            self.assertIsNone(queue.revalidated_capacity(snapshot(), current))
            # No filesystem/lock/GPU reservation is reached for an unknown sample.
            if os.name != 'nt':
                self.assertIsNone(queue.reserve(current))

    def test_process_exit_cannot_erase_busy_account_cpu(self):
        # A dead worker's lifetime CPU vanishes from the old live-PID sum.
        legacy_fraction = max(0, 500-100000) / 100 / 2 / 104
        self.assertEqual(legacy_fraction, 0)
        before = dict(usage_usec=1000000000, time=10., complete=True)
        after = dict(usage_usec=1052000000, time=12., complete=True)
        self.assertEqual(queue.account_cpu_fraction(before, after, 104), .25)

    def test_unknown_coverage_and_counter_reset_are_not_zero_cpu(self):
        before = dict(usage_usec=100, time=10., complete=True)
        for after in [dict(usage_usec=200, time=12., complete=False),
                      dict(usage_usec=None, time=12., complete=True),
                      dict(usage_usec=50, time=12., complete=True)]:
            self.assertIsNone(queue.account_cpu_fraction(before, after, 104))

    def test_other_user_claiming_gpu_invalidates_old_free_snapshot(self):
        current = queue.revalidated_capacity(snapshot(), snapshot(free=()))
        self.assertEqual(current['free'], [])
        # Another user's GPU is unavailable even though it doesn't count toward
        # this account's five-device allowance.
        self.assertEqual(current['occupied'], [])

    def test_recheck_preserves_account_occupancy_and_cpu_spike(self):
        current = queue.revalidated_capacity(snapshot(.10, (4, 5), (0,)),
                                              snapshot(.24, (5,), (3,)))
        self.assertEqual(current, snapshot(.24, (5,), (0, 3)))
        self.assertIsNone(queue.revalidated_capacity(snapshot(), snapshot(None, complete=False)))


class RecoveryLeaseTests(unittest.TestCase):
    def test_eval_recovery_requires_complete_training_and_every_model_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); work = root/'dline'; work.mkdir()
            for index in range(4):
                run = f'{TAG}_r{index}'
                checkpoint = root/'checkpoints'/run/'checkpoint-57'
                checkpoint.mkdir(parents=True)
                dataset = root/'datasets'/run
                (dataset/'meta').mkdir(parents=True)
                (dataset/'meta/info.json').write_text(json.dumps(dict(total_frames=160, total_episodes=2)))
                (checkpoint/'trainer_state.json').write_text(json.dumps(dict(global_step=57, max_steps=57)))
                (checkpoint/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'a':'model-1.safetensors'})))
                (checkpoint/'model-1.safetensors').write_bytes(b'fixture')
                receipt = dict(run=run, checkpoint=str(checkpoint), dataset=str(dataset),
                               frames=160, episodes=2, batch_size=16, steps=57)
                (work/f'{run}.trained.json').write_text(json.dumps(receipt))
                if index < 3:
                    (work/f'{run}.eval.json').write_text(json.dumps(dict(episodes=100)))
            self.assertEqual(len(verify_completed_training(root, work)), 4)
            (checkpoint/'model-1.safetensors').unlink()
            with self.assertRaises(FileNotFoundError):
                verify_completed_training(root, work)

    def test_finetune_uses_one_full_device_without_changing_memory_request(self):
        chain = SimpleNamespace(leases={}, condition=MagicMock(), log=Mock())
        allocator = SimpleNamespace(capacity=Mock(return_value=snapshot()), reserve=Mock(return_value=7))
        self.assertEqual(acquire_slots(chain, 1, 45000, allocator), [7])
        self.assertEqual(chain.leases, {7: [45000]})

    def test_evaluation_preserves_two_worker_slots_per_device(self):
        chain = SimpleNamespace(leases={}, condition=MagicMock(), log=Mock())
        allocator = SimpleNamespace(capacity=Mock(return_value=snapshot()), reserve=Mock(side_effect=[4, 6]))
        self.assertEqual(acquire_slots(chain, 4, 17000, allocator), [4, 4, 6, 6])
        self.assertEqual(chain.leases, {4: [17000, 17000], 6: [17000, 17000]})


@unittest.skipIf(os.name == 'nt', 'Linux fcntl reservation integration')
class ReservationTests(unittest.TestCase):
    def test_fresh_gpu_collision_does_not_write_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'dline').mkdir()
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'capacity',
                    return_value=snapshot(free=())), patch.object(queue.os, 'cpu_count', return_value=104):
                self.assertIsNone(queue.reserve(snapshot()))
            self.assertFalse((root/'dline/gpu_slots.json').exists())

    def test_successful_reservation_records_only_rechecked_free_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'dline').mkdir()
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'capacity',
                    return_value=snapshot(free=(5,))), patch.object(queue.os, 'cpu_count', return_value=104):
                self.assertEqual(queue.reserve(snapshot(free=(4, 5))), 5)
            entries = json.loads((root/'dline/gpu_slots.json').read_text())
            self.assertEqual(entries, [dict(owner=os.getpid(), gpu=5, mb=45000)])


if __name__ == '__main__':
    unittest.main()
