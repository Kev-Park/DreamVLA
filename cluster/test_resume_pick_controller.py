import json
from pathlib import Path
import tempfile
import unittest

from resume_pick_controller import ready, controller_command


class HandoffTests(unittest.TestCase):
    def test_controller_identity_accepts_unbuffered_python(self):
        for flags in [[],['-u'],['-B','-u']]:
            controller_command(['python',*flags,'/kevin/cluster/pick_budget_chain.py','--budget','50'],50)
        for command in [['python','other_script.py','--budget','50'],
                        ['python','pick_budget_chain.py','--budget','10'],
                        ['python','-c','pick_budget_chain.py','--budget','50']]:
            with self.assertRaises(AssertionError):controller_command(command,50)

    def test_requires_idle_workers_and_verified_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);work=root/'work';work.mkdir()
            health={'abort':None,'controller_live':True,'active_workers':[]}
            self.assertFalse(ready(root,work,50,health))
            ids=list(range(50));(work/'recipe.json').write_text(json.dumps({'motion_ids':ids}))
            (work/'collection_r3.json').write_text(json.dumps({'student_attempts':50,'motion_ids':ids}))
            dataset=root/'datasets/pick50_dagger57_20261013_r3';meta=dataset/'lerobot/ds/meta';meta.mkdir(parents=True)
            (dataset/'built.json').write_text(json.dumps({'episodes':185,'new_attempts':50}))
            (meta/'info.json').write_text(json.dumps({'total_episodes':185,'total_frames':90000}))
            self.assertTrue(ready(root,work,50,health))
            for changed in [{'active_workers':[{'pid':123}]},{'abort':{'error':'failed'}},{'controller_live':False}]:
                self.assertFalse(ready(root,work,50,dict(health,**changed)))
            (work/'collection_r3.json').write_text(json.dumps({'student_attempts':50,'motion_ids':ids[:-1]+[0]}))
            with self.assertRaises(AssertionError):ready(root,work,50,health)


if __name__=='__main__':unittest.main()
