"""Validate the approved spare substitution without Isaac or remote compute."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

with patch.dict(sys.modules,{'fcntl':SimpleNamespace(),
                            'pick_original_bluesclues_queue':SimpleNamespace(capacity=None,reserve=None,release=None)}):
    import pick_spare_chain as module
    spec=importlib.util.spec_from_file_location('spare_worker_under_test',Path(__file__).with_name('pick_original_base_collect.py'))
    worker=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)


def row(source):
    return dict(source_motion_id=source,reference_file=f'pick_{source}.pkl',reference_sha256='a'*64,
                existing_original_demo=None,requires_original_demo_collection=True)


class SpareTests(unittest.TestCase):
    def test_fixed_candidate_order_skips_accepted_and_wraps(self):
        candidates=[row(i) for i in range(19)]
        batch,cursor=module.candidate_batch(candidates,{'pick_0.pkl':{}},0,7)
        self.assertEqual([r['source_motion_id'] for r in batch],list(range(1,8)))
        batch,cursor=module.candidate_batch(candidates,{'pick_0.pkl':{}},18,3)
        self.assertEqual([r['source_motion_id'] for r in batch],[18,1,2])
        self.assertEqual(cursor,22)

    def test_only_exactly_200_distinct_originals_are_assembled(self):
        retained=[row(i) for i in range(193)]
        replacement=[row(i) for i in range(201,208)]
        missing=[f'pick_{i}.pkl' for i in range(193,200)]
        self.assertEqual(len(module.assemble_references(retained,replacement,missing)),200)
        for invalid in [replacement[:-1],replacement[:-1]+[row(0)],replacement[:-1]+[row(193)]]:
            with self.assertRaises(AssertionError):module.assemble_references(retained,invalid,missing)
        augmented=copy.deepcopy(replacement);augmented[0]['reference_file']='pick_201_aug.pkl'
        with self.assertRaises(AssertionError):module.assemble_references(retained,augmented,missing)

    def test_screening_requires_approval_and_pinned_candidate_membership(self):
        candidates=[row(i) for i in range(201,220)]
        request=dict(purpose='approved_spare_screening',spare_replacements_authorized=True,
                     references=copy.deepcopy(candidates[:7]),eligible_spare_references=candidates,
                     reference_pool='Holosoma_Pick_29_hf60v3')
        self.assertEqual(worker.reference_request(request)[1],'Holosoma_Pick_29_hf60v3')
        for field,value in [('spare_replacements_authorized',False),('reference_pool','heldout')]:
            changed=copy.deepcopy(request);changed[field]=value
            with self.assertRaises(AssertionError):worker.reference_request(changed)
        changed=copy.deepcopy(request);changed['references'][0]['reference_sha256']='b'*64
        with self.assertRaises(AssertionError):worker.reference_request(changed)
        with self.assertRaises(AssertionError):worker.reference_request(dict(references=candidates[:7]))

    def chain(self,root):
        chain=module.SpareChain.__new__(module.SpareChain)
        chain.tag='pick200_original_spares57_v2';chain.work=root/'dline'/chain.tag
        chain.work.mkdir(parents=True);chain.reference_pool='Holosoma_Pick_29_original200_spares_v2'
        (root/'ref_motions').mkdir();source=root/'sources';source.mkdir()
        retained=[row(i) for i in range(1,194)];candidates=[row(i) for i in range(201,220)]
        for r in retained+candidates:
            path=source/r['reference_file'];path.write_bytes(r['reference_file'].encode())
            r['reference_source_path']=str(path);r['reference_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            if r['source_motion_id']<=173:r['existing_original_demo']=f'/old/{r["source_motion_id"]}.hdf5'
        reused=[dict(reference_file=f'pick_{i}.pkl',path=f'/new/{i}.hdf5',passing=True,collection_seed=45) for i in range(174,194)]
        chain.request=dict(run=chain.tag,spare_replacements_authorized=True,references=retained,spare_candidates=candidates,
                           reused_original_recordings=reused,replaced_reference_names=[f'pick_{i}.pkl' for i in range(194,201)],
                           eval_reference_manifest=dict(files=[dict(name='pick_9999.pkl')]),seed=42,runtime_hashes={})
        chain.verify_reused=lambda:None;chain.verify_eval_pool=lambda:None
        chain.slots=lambda *args:[1];chain.release=lambda gpu:None
        calls=[]
        def execute(args,label,gpu,cwd):
            path=Path(args[-1]);request=json.loads(path.read_text());refs=request['references'];calls.append(request)
            passed=refs[:4] if len(calls)==1 else refs
            result=dict(state='attempts_completed',attempted_motion_ids=list(range(len(refs))),
                        recordings=[dict(reference_file=r['reference_file'],path=f'/spare/{r["source_motion_id"]}.hdf5',passing=True) for r in passed])
            module.atomic(path.with_name('collection.result.json'),result)
        chain.execute=execute
        return chain,calls

    def test_failed_candidates_are_not_counted_and_only_deficit_is_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chain,calls=self.chain(root);request=chain.work/'chain.request.json'
            module.atomic(request,chain.request)
            # Windows does not grant symlink privilege; mock only that Linux
            # filesystem boundary, keeping request replay and selection real.
            with patch.object(module,'ROOT',root),patch.object(module.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)),patch.object(module,'link_reference',side_effect=lambda destination,source:destination.write_bytes(source.read_bytes())):
                chain.complete_spares(request)
                # A restart after assembly reuses the committed request and receipts.
                chain.complete_spares(request)
            self.assertEqual([len(r['references']) for r in calls],[7,3])
            coverage=module.read(chain.work/'original_coverage.result.json')
            self.assertEqual(len(module.original_sources(chain.request,coverage)),200)
            self.assertEqual(len(coverage['recordings']),27)
            self.assertEqual([r['collection_seed'] for r in coverage['recordings'][:20]],[45]*20)
            self.assertEqual(len(list((root/'ref_motions'/chain.reference_pool).glob('*.pkl'))),200)
            self.assertFalse({r['reference_file'] for r in chain.request['references']} & {'pick_205.pkl','pick_206.pkl','pick_207.pkl'})

    def test_low_disk_preserves_reused_data_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chain,calls=self.chain(root);request=chain.work/'chain.request.json'
            with patch.object(module,'ROOT',root),patch.object(module.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(AssertionError,'headroom'):chain.complete_spares(request)
            self.assertEqual(calls,[])
            self.assertEqual(len(chain.request['reused_original_recordings']),20)
            self.assertFalse((chain.work/'original_coverage.result.json').exists())


if __name__=='__main__':unittest.main()
