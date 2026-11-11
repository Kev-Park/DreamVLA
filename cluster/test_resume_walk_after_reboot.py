import unittest
from resume_walk_after_reboot import progress


class RecoveryCoverage(unittest.TestCase):
    def test_reuse_saved_and_valid_skip_but_reissue_interrupted(self):
        text = ('[INFO] motion 001_r0/200\n'
                '[rescue] motion 1: pass A 4-rule PASS\n'
                '[rescue] WROTE student success 1 -> sonic_adapter__Pick_sim2__motion_001_r0.hdf5\n'
                '[INFO] motion 002_r0/200\n'
                '[rescue] motion 2: pass A 4-rule FAIL\n'
                '[rescue] SKIP motion 2: no reference grasp frame (grab_idx=-1)\n'
                '[INFO] motion 003_r0/200\n'
                '[rescue] motion 3: pass A 4-rule FAIL\n')
        result = progress([text], [1, 2, 3, 4])
        self.assertEqual(result['completed_ids'], [1, 2])
        self.assertEqual(result['remaining_ids'], [3, 4])
        self.assertEqual(result['infrastructure_interrupted_ids'], [3])

    def test_duplicate_saved_attempts_rejected(self):
        text = ('[INFO] motion 001_r0/200\n[rescue] motion 1: pass A 4-rule PASS\n'
                '[rescue] WROTE rescue 1 -> sonic_adapter__Pick_sim2__motion_001_r0.hdf5\n')
        with self.assertRaises(AssertionError):
            progress([text, text], [1])

    def test_unpaired_finalization_rejected(self):
        with self.assertRaises(AssertionError):
            progress(['[rescue] SKIP motion 1: no reference grasp frame\n'], [1])

    def test_unselected_reference_rejected(self):
        with self.assertRaises(AssertionError):
            progress(['[INFO] motion 010_r0/200\n'], [1])


if __name__ == '__main__':
    unittest.main()
