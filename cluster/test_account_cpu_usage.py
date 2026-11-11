"""The account cap must not be assessed from an unlabelled partial snapshot."""
import unittest

from account_cpu_usage import cpu_result


class CpuAccounting(unittest.TestCase):
    def check(self, before=None, after=None, first=None, last=None, errors=()):
        before = before if before is not None else {10: {'ticks': 100, 'born': '1', 'scope': '/user.slice/user-1008.slice/session-1.scope'}}
        after = after if after is not None else {10: {'ticks': 200, 'born': '1', 'scope': '/user.slice/user-1008.slice/session-1.scope'}}
        first = first if first is not None else {'usage_usec': 1000000, 'identity': (1, 2)}
        last = last if last is not None else {'usage_usec': 5000000, 'identity': (1, 2)}
        return cpu_result(1008, before, after, first, last, 1, 100, 100, errors)

    def test_kernel_counts_work_missing_from_endpoint_processes(self):
        result = self.check()
        self.assertEqual(result['process_snapshot_lower_bound_pct'], 1)
        self.assertEqual(result['account_cpu_pct'], 4)
        self.assertTrue(result['cpu_measurement_complete'])

    def test_account_process_outside_slice_requires_review(self):
        row = {11: {'ticks': 200, 'born': '2', 'scope': '/system.slice/cron.service'}}
        result = self.check(after=row)
        self.assertFalse(result['cpu_measurement_complete'])
        self.assertIn('outside', result['cpu_snapshot_error'])

    def test_similar_uid_prefix_does_not_count_as_covered(self):
        row = {10: {'ticks': 200, 'born': '1', 'scope': '/user.slice/user-1008.slice-other/session-1.scope'}}
        self.assertFalse(self.check(after=row)['cpu_measurement_complete'])

    def test_counter_reset_and_permission_failures_are_explicit(self):
        self.assertFalse(self.check(last={'usage_usec': 1, 'identity': (1, 2)})['cpu_measurement_complete'])
        self.assertFalse(self.check(last={'usage_usec': 5000000, 'identity': (1, 3)})['cpu_measurement_complete'])
        self.assertIn('permission', self.check(errors=['permission'])['cpu_snapshot_error'])

    def test_pid_reuse_does_not_inflate_process_delta(self):
        row = {10: {'ticks': 999999, 'born': '99', 'scope': '/user.slice/user-1008.slice'}}
        self.assertEqual(self.check(after=row)['process_snapshot_lower_bound_pct'], 0)

    def test_missing_kernel_counter_cannot_claim_complete_account_measurement(self):
        row = {10: {'ticks': 100, 'born': '1', 'scope': '/user.slice/user-1008.slice'}}
        result = cpu_result(1008, row, row, None, None, 1, 100, 100)
        self.assertFalse(result['cpu_measurement_complete'])
        self.assertIn('unavailable', result['cpu_snapshot_error'])

    def test_watchdog_surfaces_partial_cpu_coverage(self):
        from walk_study_watchdog import assess
        data = {'account_resources': {'cpu_snapshot_error': 'outside user slice'},
                'disk_free_gib': 100, 'workflow': {'status': 'complete'}, 'results': []}
        self.assertIn('Account resource snapshot incomplete: outside user slice',
                      assess('biped', data, 100, {}))


if __name__ == '__main__':
    unittest.main()
