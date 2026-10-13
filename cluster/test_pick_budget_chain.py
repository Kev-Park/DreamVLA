import unittest
from pick_budget_chain import steps_for, verify_collection_logs


class BudgetTests(unittest.TestCase):
    def test_known_and_small_dataset_steps(self):
        self.assertEqual(steps_for(4990),1778)
        self.assertEqual(steps_for(24950),8889)
        for frames in [1,4990,24950,9980,99801]:
            steps=steps_for(frames)
            self.assertGreaterEqual(steps*16,frames*5.7)
            self.assertLess((steps-1)*16,frames*5.7)

    def log(self,ids,errors=0):
        return '\n'.join(f'[INFO] motion {i}_r0/400\n[rescue] motion {i}: pass A 4-rule FAIL' for i in ids)+f'\n[INFO] Collection finished. (errored={errors})'

    def test_shards_must_cover_each_reference_once(self):
        ids=list(range(10))
        self.assertEqual(verify_collection_logs([self.log(ids[::2]),self.log(ids[1::2])],ids)['student_attempts'],10)
        for logs in [[self.log(ids[:-1])],[self.log(ids+[0])],[self.log(ids,1)],['']]:
            with self.assertRaises(ValueError):verify_collection_logs(logs,ids)


if __name__=='__main__':unittest.main()
