import unittest
from gpu_slots import select_slots


class SelectionTests(unittest.TestCase):
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
