"""Lightweight schedule checks, without importing Isaac or loading a model."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
    'Training/scripts/reinforcement_learning/rsl_rl'))
from vla_sonic.rescue_sampling import takeover_bounds


class RescueSamplingTest(unittest.TestCase):
    def test_existing_bounds_are_unchanged(self):
        for grab in (1, 17, 278, 600):
            for fraction in (0.1, 0.5, 1.0):
                self.assertEqual(takeover_bounds(grab, k_frac=fraction),
                                 (0, max(1, int(round(fraction * grab)))))

    def test_final_second_uses_simulation_rate(self):
        self.assertEqual(takeover_bounds(278, window_s=1.0, step_dt=0.02), (228, 278))
        self.assertEqual(takeover_bounds(278, window_s=0.5, step_dt=0.02), (253, 278))
        self.assertEqual(takeover_bounds(278, window_s=1.0, step_dt=0.01), (178, 278))

    def test_short_trajectory_clips_at_start(self):
        self.assertEqual(takeover_bounds(20, window_s=1.0), (0, 20))
        self.assertEqual(takeover_bounds(1, window_s=0.001), (0, 1))

    def test_invalid_or_conflicting_window_rejected(self):
        for window in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                takeover_bounds(278, window_s=window)
        with self.assertRaises(ValueError):
            takeover_bounds(278, window_s=1.0, k_frac=0.5)
        with self.assertRaises(ValueError):
            takeover_bounds(0, window_s=1.0)


if __name__ == '__main__':
    unittest.main()
