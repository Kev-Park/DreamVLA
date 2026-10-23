"""Compare unchanged checkpoints on exact saved evaluation inputs and RNG states."""
import gc, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
from walk_gap_offline_predictions import atomic


def main():
    root=Path.home()/'kevin';repo=Path(__file__).resolve().parent.parent
    work=root/'dline/walk_gap_eval_input_replay20261022'
    output=root/'eval_videos'/work.name
    source=root/'eval_videos/walk_gap_eval_input_capture20261022'
    review=json.loads((work/'operator_review.json').read_text())
    assert review['existing_environment_approved'] and review['account_resources_manually_reviewed']
    assert review['other_agents_pending_allocations_manually_reviewed']
    assert 0<=time.time()-review['at']<120
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(review['gpu'])
    assert set(os.sched_getaffinity(0))<=set(review['cpus'])
    assert Path(sys.prefix)==root/'wt/locomotion/.venv'
    assert not (work/'launch.json').exists() and not output.exists()
    for filename,key in [('capture_independent_audit.json','audit_sha256'),('replay_sample_plan.json','plan_sha256')]:
        assert hashlib.sha256((source/filename).read_bytes()).hexdigest()==review[key]
    audit=json.loads((source/'capture_independent_audit.json').read_text())
    assert audit['all_queries_complete'] and audit['all_executed_hand_commands_match']
    points=json.loads((source/'replay_sample_plan.json').read_text())['points']
    inventory={r['path']:r for r in audit['inventory']}
    output.mkdir();atomic(work/'launch.json',{'at':time.time(),'pid':os.getpid(),'review':review})
    reserved=False
    try:
        selected=subprocess.check_output([sys.executable,str(repo/'cluster/gpu_slots.py'),'reserve','--owner',str(os.getpid()),
            '--gpu',str(review['gpu']),'--count','1','--mb','15000'],text=True).strip()
        assert selected==str(review['gpu']);reserved=True
        import numpy as np
        import torch
        sys.path.insert(0,str(root/'Isaac-GR00T'))
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        torch.set_num_threads(2)
        def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value
        models={
            'current_r2':root/'checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967',
            'historical_r2':root/'checkpoints/T20maxf_dagger2_run01/checkpoint-24000',
            'current_r3':root/'checkpoints/walk200_v3_dagger57_20261013_r3/checkpoint-118933'}
        runtime={'python':sys.version,'prefix':sys.prefix,'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),'grad_enabled':torch.is_grad_enabled(),
            'inference_mode':torch.is_inference_mode_enabled(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn':torch.backends.cudnn.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision()}
        atomic(output/'runtime.json',runtime)
        records=[];arrays={};reproduction=[]
        for label,checkpoint in models.items():
            atomic(work/'workflow.json',{'heartbeat':time.time(),'pid':os.getpid(),'status':'loading_model','model':label,'rows':len(records)})
            policy=Gr00tPolicy(embodiment_tag='unitree_g1_sonic',model_path=str(checkpoint),device='cuda:0')
            for index,point in enumerate(points):
                file=source/point['path'];assert hashlib.sha256(file.read_bytes()).hexdigest()==inventory[point['path']]['sha256']
                with np.load(file,allow_pickle=False) as z:
                    md=json.loads(str(z['metadata_json']));obs={}
                    for key in z.files:
                        if not key.startswith('observation/'):continue
                        _,modality,name=key.split('/',2);value=z[key].copy()
                        obs.setdefault(modality,{})[name]=value.tolist() if modality=='language' else value
                    numpy=md['numpy_rng_before']
                    torch.set_rng_state(torch.from_numpy(z['rng/torch_cpu_before'].copy()))
                    torch.cuda.set_rng_state(torch.from_numpy(z['rng/torch_cuda_before'].copy()),0)
                    np.random.set_state((numpy[0],z['rng/numpy_keys_before'].copy(),*numpy[1:]))
                    random.setstate(tuples(md['python_rng_before']))
                    # Match the actual evaluator: no additional outer autocast/inference context.
                    result=policy.get_action(obs);prediction=result[0] if isinstance(result,tuple) else result
                    token=np.asarray(prediction['motion_token']);hand=np.asarray(prediction['right_hand_joints'])
                    assert token.shape==(1,40,64) and hand.shape==(1,40,7)
                    assert np.isfinite(token).all() and np.isfinite(hand).all()
                    count=point['executed_steps'];meanq=np.abs(hand[0,:count].astype(np.float32)).mean(axis=1)
                    close=meanq>.6
                    if label=='current_r2':
                        token_ref=z['prediction/motion_token'];hand_ref=z['prediction/right_hand_joints']
                        token_match=bool(np.allclose(token,token_ref,rtol=1e-4,atol=1e-5))
                        hand_match=bool(np.allclose(hand,hand_ref,rtol=1e-4,atol=1e-5))
                        close_match=bool(np.array_equal(close,np.abs(hand_ref[0,:count].astype(np.float32)).mean(axis=1)>.6))
                        reproduction.append({'path':point['path'],'token_allclose':token_match,'hand_allclose':hand_match,
                            'executed_close_exact':close_match,'token_maxabs':float(np.abs(token-token_ref).max()),
                            'hand_maxabs':float(np.abs(hand-hand_ref).max())})
                        atomic(output/'reproduction.json',{'at':time.time(),'tolerance':{'rtol':1e-4,'atol':1e-5},'records':reproduction})
                        assert token_match and hand_match and close_match,('Recorded-current-R2 reproduction failed; stop before comparing other models',reproduction[-1])
                    prefix=f'{label}_{index:03d}'
                    arrays[prefix+'/motion_token']=token;arrays[prefix+'/right_hand_joints']=hand
                    records.append({**point,'model':label,'capture_sha256':inventory[point['path']]['sha256'],
                        'motion_time':md['motion_time'],'executed_close':close.tolist(),'executed_hand_meanabs':meanq.tolist(),
                        'predicted_close_frames':int(close.sum()),'prediction_array_prefix':prefix})
                if index%8==0:
                    atomic(output/'predictions.json',{'at':time.time(),'records':records})
                    atomic(work/'workflow.json',{'heartbeat':time.time(),'pid':os.getpid(),'status':'predicting','model':label,'rows':len(records),'sample_index':index})
                    print(label,index+1,'/',len(points),flush=True)
            del policy;gc.collect();torch.cuda.empty_cache()
        assert len(reproduction)==len(points) and len(records)==3*len(points)
        np.savez(output/'prediction_arrays.npz',**arrays)
        atomic(output/'predictions.json',{'at':time.time(),'records':records})
        atomic(output/'results.json',{'at':time.time(),'points':len(points),'predictions':len(records),
            'current_r2_full_chunk_reproduction_passed':True,'identical_actual_evaluation_observations_and_rng':True,
            'accepted_previous_full_fsq_trajectories':True,'decoder_changed':False,
            'limits':'Conditional predictions on ten selected current-R2 rollout states; not other models closed-loop trajectories, unbiased success, or causal attribution.'})
        atomic(work/'done.json',{'at':time.time(),'points':len(points),'predictions':len(records)})
        atomic(work/'workflow.json',{'heartbeat':time.time(),'pid':os.getpid(),'status':'complete','rows':len(records)})
    except BaseException as exc:
        atomic(work/'abort.json',{'at':time.time(),'error':repr(exc)})
        atomic(work/'workflow.json',{'heartbeat':time.time(),'pid':os.getpid(),'status':'failed_for_review','alerts':[repr(exc)]});raise
    finally:
        if reserved:subprocess.run([sys.executable,str(repo/'cluster/gpu_slots.py'),'release','--owner',str(os.getpid()),'--gpu',str(review['gpu'])],check=True)


if __name__=='__main__':main()
