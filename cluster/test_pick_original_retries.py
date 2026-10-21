"""Exercise receipt replay and continued retries without Isaac or GPU access."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Mock the Linux allocation boundary, rather than importing fcntl on Windows.
with patch.dict(sys.modules, {'pick_original_bluesclues_queue': SimpleNamespace(capacity=None,reserve=None,release=None)}):
    spec=importlib.util.spec_from_file_location('original_retries_under_test',Path(__file__).with_name('pick_original_chain.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class OriginalRetryTests(unittest.TestCase):
    def request(self):
        return dict(run='pick200_original_dagger57_v1',seed=42,retry_failed_originals_authorized=True,
                    runtime_hashes={'cluster/pick_original_chain.py':'a'*64,'collector.py':'b'*64},
                    references=[dict(reference_file=f'pick_{i}.pkl',source_motion_id=i,
                       existing_original_demo=f'/old/{i}.hdf5' if i<=173 else None) for i in range(1,201)])

    def test_only_controller_pin_may_change_in_completed_receipt(self):
        expected=self.request();prior=copy.deepcopy(expected)
        prior['runtime_hashes']['cluster/pick_original_chain.py']='c'*64
        self.assertTrue(module.completed_retry_request_matches(prior,expected))
        for changed in ['seed','collector','reference','approval']:
            modified=copy.deepcopy(prior)
            if changed=='seed':modified['seed']+=1
            elif changed=='collector':modified['runtime_hashes']['collector.py']='d'*64
            elif changed=='reference':modified['references'][0]['reference_file']='augmented.pkl'
            else:modified['retry_failed_originals_authorized']=False
            self.assertFalse(module.completed_retry_request_matches(modified,expected),changed)

    def chain(self,root):
        chain=module.OriginalChain.__new__(module.OriginalChain)
        chain.request=self.request();chain.tag=chain.request['run'];chain.work=root
        initial=[dict(reference_file=f'pick_{i}.pkl',path=f'/new/{i}.hdf5',passing=True,bytes=100) for i in range(174,200)]
        (root/'collection.result.json').write_text(json.dumps(dict(recordings=initial)))
        chain.log=lambda message:None;chain.slots=lambda *args:[1];chain.release=lambda gpu:None
        def execute(args,label,gpu,cwd):
            request=Path(args[-1]);value=json.loads(request.read_text());passed=value['seed']==46
            rows=[dict(reference_file='pick_200.pkl',path='/new/200.hdf5',passing=True)] if passed else []
            request.with_name('collection.result.json').write_text(json.dumps(dict(state='attempts_completed',recordings=rows)))
        chain.execute=execute
        return chain

    def test_three_zero_gain_sweeps_do_not_prevent_fourth_sweep_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chain=self.chain(root)
            with patch.object(module.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)):
                chain.complete_original_coverage()
            result=json.loads((root/'original_coverage.result.json').read_text())
            self.assertEqual(result['passing'],27)
            self.assertEqual(len(module.original_sources(chain.request,result)),200)
            self.assertTrue((root/'original_retry_plateau.json').exists())

    def test_insufficient_disk_does_not_launch_or_drop_accepted_demos(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chain=self.chain(root)
            chain.execute=lambda *args: self.fail('Must not launch without headroom')
            with patch.object(module.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError,'Insufficient retry-collection headroom'):
                    chain.complete_original_coverage()
            self.assertFalse((root/'original_coverage.result.json').exists())
            self.assertEqual(len(json.loads((root/'collection.result.json').read_text())['recordings']),26)

    def historical_request(self,chain):
        prior=copy.deepcopy(chain.request)
        prior.update(run=chain.tag+'_retry01',seed=43,expected_attempts=1)
        for row in prior['references']:
            row['requires_original_demo_collection']=row['reference_file']=='pick_200.pkl'
        prior['runtime_hashes']['cluster/pick_original_chain.py']='c'*64
        directory=chain.work/'original_retries/01';directory.mkdir(parents=True)
        request=directory/'collection.request.json';request.write_text(json.dumps(prior))
        return request

    def test_completed_historical_request_is_preserved_and_not_recollected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chain=self.chain(root);request=self.historical_request(chain)
            before=request.read_bytes()
            request.with_name('collection.result.json').write_text(json.dumps(dict(state='attempts_completed',recordings=[])))
            executed=[];original=chain.execute
            def execute(args,*rest):
                executed.append(json.loads(Path(args[-1]).read_text())['seed']);original(args,*rest)
            chain.execute=execute
            with patch.object(module.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)):
                chain.complete_original_coverage()
            self.assertEqual(executed,[44,45,46])
            self.assertEqual(request.read_bytes(),before)

    def test_incomplete_old_request_requires_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chain=self.chain(root);self.historical_request(chain)
            chain.execute=lambda *args: self.fail('Must not launch an old mismatched request')
            with self.assertRaisesRegex(AssertionError,'Retry request changed'):
                chain.complete_original_coverage()


if __name__=='__main__':unittest.main()
