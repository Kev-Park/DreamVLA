import unittest
from transition_finetune_epochs import step_budget


class StepBudgetTests(unittest.TestCase):
    def test_target_is_not_truncated_by_old_cap(self):
        self.assertEqual(step_budget(354630, 5.7), 127000)
        self.assertGreaterEqual(step_budget(354630, 5.7) * 16 / 354630, 5.7)

    def test_rounds_up_and_preserves_minimum(self):
        self.assertEqual(step_budget(211881, 5.7), 76000)
        self.assertEqual(step_budget(100, 5.7), 10000)

    def test_invalid_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            step_budget(0, 5.7)


if __name__ == "__main__":
    unittest.main()
