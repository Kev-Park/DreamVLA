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
