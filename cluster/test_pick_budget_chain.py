import unittest
import concurrent.futures
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import pick_budget_chain
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

    def test_external_wait_does_not_block_local_evaluation(self):
        remote_started=threading.Event();release_remote=threading.Event();last_local=threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);work=root/'work';work.mkdir();(root/'eval_videos').mkdir()
            (work/'box.json').write_text('{}')
            report=root/'eval_videos/r2';report.mkdir()
            (report/'results.json').write_text(json.dumps({'pool':pick_budget_chain.EVAL_REFS,
                'episodes':100,'per_episode':[{'motion_id':i} for i in range(100)]}))
            chain=pick_budget_chain.Chain.__new__(pick_budget_chain.Chain)
            chain.work=work;chain.budget=50;chain.local_eval_lock=threading.Lock()
            chain.prepare=lambda:None;chain.collect=lambda *args:None;chain.dataset=lambda *args:None
            chain.finetune=lambda data,index:(f'r{index}',root/f'checkpoint{index}')
            chain.log=lambda message:None
            active=0;maximum=0
            def local(run, checkpoint):
                nonlocal active,maximum
                active+=1;maximum=max(maximum,active)
                time.sleep(0.05)
                if run=='r3':last_local.set()
                active-=1
            def external(phase,run,payload):
                if run!='r2':return False
                remote_started.set()
                if not release_remote.wait(5):raise TimeoutError('Remote release missing')
                return True
            chain.evaluate_local=local;chain.external=external
            fake_fcntl=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
            with patch.object(pick_budget_chain,'ROOT',root),patch.object(pick_budget_chain,'fcntl',fake_fcntl,create=True):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    running=executor.submit(chain.run)
                    try:
                        self.assertTrue(remote_started.wait(2))
                        self.assertTrue(last_local.wait(2),'Ready local evaluation waited for remote host')
                    finally:
                        release_remote.set()
                        running.result(timeout=3)
            self.assertEqual(maximum,1,'Local evaluations overlapped')


if __name__=='__main__':unittest.main()
