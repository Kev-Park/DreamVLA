"""A submitted run resumes its existing job rather than restaging mutable data."""
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import walk_late_takeover_bridge as bridge


class SubmittedResumeTests(unittest.TestCase):
    def state(self):
        return {'completed_batches':[f'batch_{i:03d}' for i in range(20)],
                'dataset':{'dataset':'~/kevin/datasets/example','new_attempts':200,'steps':109629},
                'training_submission':{'job_id':'3431651','stage':'finetune','command':['sbatch','existing']}}

    def test_new_run_does_not_require_remote_submission(self):
        with patch.object(bridge,'read_remote') as read:
            self.assertIsNone(bridge.submitted_training_state({}))
            read.assert_not_called()

    def test_missing_receipt_cannot_create_another_job(self):
        with patch.object(bridge,'read_remote',return_value=None),patch.object(bridge,'submit') as submit:
            with self.assertRaisesRegex(AssertionError,'never resubmit'):
                bridge.submitted_training_state(self.state())
            submit.assert_not_called()

    def test_changed_dataset_requires_reconciliation(self):
        state=self.state()
        with patch.object(bridge,'read_remote',side_effect=[state['training_submission'],{'steps':1}]):
            with self.assertRaisesRegex(AssertionError,'dataset identity changed'):
                bridge.submitted_training_state(state)

    def test_driver_skips_recopy_and_submission_for_existing_job(self):
        state=self.state()
        resources={'cpu_measurement_complete':True,'account_cpu_pct':0,
                   'account_gpu_devices':[],'account_gpu_processes':[],'live_gpu_reservations':[]}
        health={'status':'collection_complete','completed_attempts':200,'pid_live':False}
        with tempfile.TemporaryDirectory() as directory,contextlib.ExitStack() as stack:
            Path(directory,'workflow.json').write_text(json.dumps(state))
            stack.enter_context(patch('sys.argv',['bridge','--state-dir',directory]))
            stack.enter_context(patch.object(bridge,'configure'))
            stack.enter_context(patch.object(bridge,'read_remote',side_effect=[state['training_submission'],state['dataset']]))
            stack.enter_context(patch.object(bridge,'collector_health',return_value=health))
            stack.enter_context(patch.object(bridge,'resources',return_value=resources))
            job=stack.enter_context(patch.object(bridge,'job_state',return_value={'state':'FAILED','exitcode':'1:0'}))
            forbidden={name:stack.enter_context(patch.object(bridge,name))
                       for name in ('inventory','checked_transfer','start_assembly','submit')}
            with self.assertRaises(AssertionError):
                bridge.main()
            job.assert_called_once_with('3431651')
            for function in forbidden.values():function.assert_not_called()
            recorded=json.loads(Path(directory,'workflow.json').read_text())
            self.assertTrue(recorded['resumed_existing_submission'])
            self.assertEqual(recorded['training_submission'],state['training_submission'])


if __name__=='__main__':
    unittest.main()
