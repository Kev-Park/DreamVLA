import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pick_budget_biped_eval import validate_report, slurm_worker_gpus, run_worker, FINAL_RUN, POOL


class EvaluationGateTests(unittest.TestCase):
    def test_adroit_dispatch_preserves_all_holdout_shards_and_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);root=home/'kevin';work=root/'dline/eval';work.mkdir(parents=True)
            runtime=root/'wt/pinned';runtime.mkdir(parents=True)
            (runtime/'marker').write_text('pinned runtime')
            refs=root/'ref_motions'/POOL;refs.mkdir(parents=True)
            for i in range(100):(refs/f'pick_{i}.pkl').touch()
            request=work/'request.json'
            request.write_text(json.dumps({'host':'adroit','run':FINAL_RUN,'pool':POOL,'episodes':100,
                'runtime_relative':'wt/pinned','runtime_files':{'marker':hashlib.sha256(b'pinned runtime').hexdigest()},
                'checkpoint_relative':'checkpoints/final','output_relative':'eval_videos/final',
                'box':{'lo':[0,0,0],'hi':[1,1,1]},'eval_environment':{'HS_EVAL_NO_EE_TERM':'1'}}))
            calls=[]
            def spawn(args,**kwargs):
                calls.append((args,kwargs['env']['CUDA_VISIBLE_DEVICES'],kwargs['cwd']))
                return SimpleNamespace(pid=123,wait=lambda:0)
            def score(args,**kwargs):
                (root/'eval_videos/final/results.json').write_text(json.dumps(self.report()))
            with patch.object(Path,'home',return_value=home), patch.dict('os.environ',
                 {'SLURM_JOB_ID':'123','SLURM_GPUS_ON_NODE':'1','CUDA_VISIBLE_DEVICES':'3'}), \
                 patch('subprocess.Popen',side_effect=spawn), patch('subprocess.run',side_effect=score), \
                 patch('subprocess.check_output',side_effect=AssertionError('Must not discover unallocated GPUs')):
                run_worker(request)
            self.assertEqual(sorted(int(args[args.index('--seed')+1]) for args,_,_ in calls),[100,101,102,103])
            self.assertTrue(all(gpu=='3' and cwd==runtime/'Training' for _,gpu,cwd in calls))
            ids=[int(i) for f in work.glob('eval_ids*.txt') for i in f.read_text().split()]
            self.assertEqual(sorted(ids),list(range(100)))
            self.assertEqual(json.loads((work/'state.json').read_text())['phase'],'complete')

    def test_adroit_workers_keep_the_assigned_device(self):
        for device in ['0', '3', 'GPU-assigned-uuid']:
            self.assertEqual(slurm_worker_gpus({'SLURM_JOB_ID':'123',
                             'CUDA_VISIBLE_DEVICES':device,'SLURM_GPUS_ON_NODE':'1'}),
                             [device,device])
        for environment in [{}, {'CUDA_VISIBLE_DEVICES':'0'},
                            {'SLURM_JOB_ID':'123','CUDA_VISIBLE_DEVICES':'0,1'},
                            {'SLURM_JOB_ID':'123','CUDA_VISIBLE_DEVICES':'0','SLURM_GPUS_ON_NODE':'2'}]:
            with self.assertRaises(ValueError):
                slurm_worker_gpus(environment)

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
