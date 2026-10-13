import unittest
from pick_budget_biped_eval import validate_report, POOL


class EvaluationGateTests(unittest.TestCase):
    def report(self):
        return {'pool': POOL, 'episodes': 100, 'successes': 20,
                'world_mpjpe_cm': 5., 'root_aligned_mpjpe_cm': 4.,
                'per_episode': [{'motion_id': i, 'success': i < 20,
                                 'world_mpjpe_cm': 5., 'root_aligned_mpjpe_cm': 4.}
                                for i in range(100)]}

    def test_requires_full_unique_holdout_and_finite_consistent_metrics(self):
        validate_report(self.report())
        invalid = []
        d = self.report(); d['per_episode'].pop(); invalid.append(d)
        d = self.report(); d['per_episode'][1]['motion_id'] = 0; invalid.append(d)
        d = self.report(); d['successes'] = 21; invalid.append(d)
        d = self.report(); d['world_mpjpe_cm'] = float('nan'); invalid.append(d)
        d = self.report(); d['per_episode'][0]['root_aligned_mpjpe_cm'] = 10.; invalid.append(d)
        d = self.report(); d['pool'] = 'training_pool'; invalid.append(d)
        for report in invalid:
            with self.assertRaises(AssertionError):
                validate_report(report)


if __name__ == '__main__':
    unittest.main()
