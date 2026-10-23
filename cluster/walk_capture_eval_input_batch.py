"""Capture ten selected standard-eval replicas without changing policy outputs."""
import argparse,concurrent.futures,hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
from gr00t_budget_chain import atomic,read
from walk_gr00t_chain import WalkChain
from watch_walk_gr00t import process_argv,stage_command_matches,output_activity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-model',choices=['current_r2','historical_r2'],default='current_r2');args=parser.parse_args()
    root=Path.home()/'kevin';repo=Path(__file__).resolve().parent.parent
    name='walk_gap_eval_input_capture20261022' if args.capture_model=='current_r2' else 'walk_gap_hist_eval_input_capture20261022'
    work=root/'dline'/name;out=root/'eval_videos'/work.name
    review=read(work/'operator_review.json')
    assert review.get('capture_model','current_r2')==args.capture_model
    checkpoint={'current_r2':'walk200_v3_dagger57_20261013_r2/checkpoint-92967','historical_r2':'T20maxf_dagger2_run01/checkpoint-24000'}[args.capture_model]
    assert socket.gethostname().startswith('bluesclues') and review['standard_evaluation_logging_only']
    assert review['account_resources_manually_reviewed'] and review['other_agents_pending_allocations_manually_reviewed']
    assert 0<=time.time()-review['at']<120 and set(os.sched_getaffinity(0))==set(review['cpus'])
    assert len(review['cpus'])<=18 and review['workers']==2
    assert len(review['motion_ids'])==len(set(review['motion_ids']))==10 and set(review['motion_ids'])<=set(range(100))
    for relative,sha in review['source_sha256'].items():assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==sha,relative
    assert not (work/'launch.json').exists() and not out.exists()
    out.mkdir();atomic(work/'launch.json',{'at':time.time(),'pid':os.getpid(),'review':review})
    slots=[]
    try:
        slots=list(map(int,subprocess.check_output([sys.executable,str(repo/'cluster/gpu_slots.py'),'reserve','--owner',str(os.getpid()),
            '--gpu',str(review['gpu']),'--count','2','--mb','17000'],text=True).split()))
        assert slots==[review['gpu']]*2,'Reviewed capacity changed'
        chain=WalkChain(200);chain.tag=work.name;chain.work=work;chain.sim_task_threads=4;chain.env['PXR_WORK_THREAD_LIMIT']='4'
        jobs=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[]
            for k in range(2):
                ids=review['motion_ids'][k::2];idsfile=work/f'ids{k}.txt';idsfile.write_text(' '.join(map(str,ids))+'\n')
                label=f'capture_shard{k}';jobs.append({'label':label,'ids':ids,'seed':200+k})
                args=['env','HS_EVAL_NO_EE_TERM=1',sys.executable,repo/'cluster/walk_capture_vla_eval_inputs.py',
                    '--capture-root',out/f'inputs_b{k}','--headless','--sonic-pt',chain.pt,'--ref-motions-path','StitchMAX100v3_eval',
                    '--vla-checkpoint',root/'checkpoints'/checkpoint,
                    '--waist-dof','29','--motions-from',idsfile,'--num-episodes',str(len(ids)),'--seed',str(200+k),
                    '--traj-dump',out/f'b{k}','--grasp-dump',out/f'grasp_b{k}.pkl','--max-steps-per-episode','1200',
                    '--kit_args=--/plugins/carb.tasking.plugin/threadCount=4']
                futures.append(pool.submit(chain.execute,args,label,review['gpu'],chain.training_cwd))
            atomic(work/'jobs.json',{'jobs':jobs})
            while not all(f.done() for f in futures):
                rows=[];alerts=[]
                for job in jobs:
                    handle=work/(job['label']+'.process.json')
                    if not handle.exists():continue
                    h=read(handle);live=stage_command_matches(h['command'],process_argv(h['pid']))
                    row={'label':job['label'],'pid':h['pid'],'live':live}
                    if live:
                        activity,stalled=output_activity(h['pid'],work/(job['label']+'.log'),time.time());row.update(activity);alerts.extend(stalled)
                    rows.append(row)
                atomic(work/'workflow.json',{'heartbeat':time.time(),'pid':os.getpid(),'status':'capturing','workers':rows,
                    'capture_files':len(list(out.glob('inputs_b*/*.npz'))),'alerts':alerts,'gpu':review['gpu']})
                concurrent.futures.wait(futures,timeout=30)
            for f in futures:f.result()
        receipts=[read(out/f'inputs_b{k}/capture_complete.json') for k in range(2)]
        assert all(r['logging_does_not_advance_rng'] and r['policy_calls_per_query']==1 and r['compression']=='none' for r in receipts)
        ids=sorted(int(re.search(r'_m(\d+)_traj.npz$',p.name)[1]) for p in out.glob('b*_traj.npz'))
        assert ids==sorted(review['motion_ids'])
        atomic(work/'done.json',{'at':time.time(),'selected_reference_replicas':10,'capture_queries':sum(r['queries'] for r in receipts),
            'capture_bytes':sum(r['bytes'] for r in receipts),'receipts':receipts,
            'limits':'Selected diagnostic references with new seeds200/201, not an unbiased success estimate or exact original-baseline replay.'})
        atomic(work/'workflow.json',{'heartbeat':time.time(),'pid':os.getpid(),'status':'complete','alerts':[],
            'capture_files':sum(r['queries'] for r in receipts),'gpu':review['gpu']})
    except BaseException as exc:
        atomic(work/'abort.json',{'at':time.time(),'error':repr(exc)})
        atomic(work/'workflow.json',{'heartbeat':time.time(),'pid':os.getpid(),'status':'failed_for_review','alerts':[repr(exc)]});raise
    finally:
        for gpu in slots:subprocess.run([sys.executable,str(repo/'cluster/gpu_slots.py'),'release','--owner',str(os.getpid()),'--gpu',str(gpu)],check=True)


if __name__=='__main__':main()
