"""Check whether completed R3 training learned the saved R3 correction targets."""
import collections,copy,gc,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
from walk_gap_offline_predictions import atomic


def main():
    root=Path.home()/'kevin';repo=Path(__file__).resolve().parent.parent
    work=root/'dline/walk_gap_rescue_prefix_r3_20261022'
    output=root/'eval_videos/walk_gap_rescue_prefix_r3_20261022'
    prior=root/'eval_videos/walk_gap_rescue_prefix20261022'
    review=json.loads((work/'operator_review.json').read_text())
    assert review['existing_gr00t_environment_approved'] and review['account_resources_manually_reviewed']
    assert 0<=time.time()-review['at']<120 and sys.prefix==str(root/'Isaac-GR00T/.venv')
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(review['gpu']) and set(os.sched_getaffinity(0))<=set(review['cpus'])
    assert not (work/'launch.json').exists() and not output.exists()
    output.mkdir();atomic(work/'launch.json',{'at':time.time(),'pid':os.getpid(),'review':review})
    reserved=False
    try:
        selected=subprocess.check_output([sys.executable,str(repo/'cluster/gpu_slots.py'),'reserve','--owner',str(os.getpid()),
            '--gpu',str(review['gpu']),'--count','1','--mb','15000'],text=True).strip()
        assert selected==str(review['gpu']);reserved=True
        import numpy as np
        import torch
        from gr00t.data.utils import parse_modality_configs
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.eval.open_loop_eval import parse_observation_gr00t
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        torch.set_num_threads(2)
        assert hashlib.sha256((prior/'sample_plan.json').read_bytes()).hexdigest()==review['prior_sample_plan_sha256']
        plan=json.loads((prior/'sample_plan.json').read_text());old=json.loads((prior/'predictions.json').read_text())['records']
        hashes={(r['episode'],r['query_step']):r['observation_sha256'] for r in old}
        checkpoint=root/'checkpoints/walk200_v3_dagger57_20261013_r3/checkpoint-118933'
        config=json.loads((checkpoint/'processor_config.json').read_text())['processor_kwargs']
        modality={k:v for k,v in parse_modality_configs(config['modality_configs'])['unitree_g1_sonic'].items() if k!='rl_info'}
        loader=LeRobotEpisodeLoader(dataset_path=str(root/'datasets/walk200_v3_dagger57_20261013_r3/lerobot/ds'),
            modality_configs=modality,video_backend='torchcodec',video_backend_kwargs=None)
        obs_modality=copy.deepcopy(modality);obs_modality.pop('action')
        grouped=collections.defaultdict(list)
        for point in plan['points']:grouped[point['episode']].append(point)
        atomic(work/'workflow.json',{'heartbeat':time.time(),'status':'loading_model','rows':0})
        policy=Gr00tPolicy(embodiment_tag='unitree_g1_sonic',model_path=str(checkpoint),device='cuda:0')
        records=[]
        for episode,points in grouped.items():
            trajectory=loader[episode];gt_hand=np.stack(trajectory['action.right_hand_joints']);gt_token=np.stack(trajectory['action.motion_token'])
            queries=collections.defaultdict(list)
            for point in points:queries[point['query_step']].append(point)
            for query_step,targets in sorted(queries.items()):
                data=extract_step_data(trajectory,query_step,obs_modality,EmbodimentTag.resolve('unitree_g1_sonic'))
                obs={**{f'state.{k}':v for k,v in data.states.items()},**{f'video.{k}':np.array(v) for k,v in data.images.items()}}
                for key in modality['language'].modality_keys:obs[key]=data.text
                digest=hashlib.sha256()
                for key,value in sorted(obs.items()):
                    digest.update(key.encode())
                    if isinstance(value,str):digest.update(value.encode())
                    else:
                        array=np.ascontiguousarray(value);digest.update(str((array.shape,array.dtype)).encode());digest.update(array.tobytes())
                assert digest.hexdigest()==hashes[episode,query_step],'Observation differs from paired R2 comparison'
                parsed=parse_observation_gr00t(obs,modality)
                for seed in [0,1,2,3]:
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    with torch.inference_mode():prediction,_=policy.get_action(parsed)
                    for point in targets:
                        offset=point['chunk_offset'];hand=np.asarray(prediction['right_hand_joints'])[0,offset];token=np.asarray(prediction['motion_token'])[0,offset]
                        assert hand.shape==(7,) and token.shape==(64,) and np.isfinite(hand).all() and np.isfinite(token).all()
                        records.append({**point,'model':'current_r3','seed':seed,'observation_sha256':digest.hexdigest(),
                            'predicted_close':bool(np.abs(hand).mean()>.6),'predicted_hand_meanabs':float(np.abs(hand).mean()),
                            'hand_mse_at_target':float(np.square(hand-gt_hand[point['target_step']]).mean()),
                            'token_mse_at_target':float(np.square(token-gt_token[point['target_step']]).mean()),
                            'predicted_hand':hand.tolist(),'predicted_token':token.tolist()})
            atomic(output/'predictions.json',{'at':time.time(),'records':records})
            atomic(work/'workflow.json',{'heartbeat':time.time(),'status':'predicting','episode':episode,'rows':len(records)})
            print(episode,'rows',len(records),flush=True);del trajectory;gc.collect()
        groups={}
        for phase in ['A','B']:
            for category in sorted({r['category'] for r in records}):
                rows=[r for r in records if r['pass']==phase and r['category']==category]
                if rows:groups[phase+'/'+category]={'draws':len(rows),'predicted_close_draws':sum(r['predicted_close'] for r in rows),
                    'predicted_close_pct':100*sum(r['predicted_close'] for r in rows)/len(rows)}
        atomic(output/'results.json',{'at':time.time(),'rows':len(records),'groups':groups,
            'observations_identical_to_r2_comparison':True,'in_sample_for_r3_but_unseen_during_r2_training':True,
            'no_input_output_or_decoder_changes':True,'limits':'Training-data prediction check only; not held-out closed-loop success or proof of causality.'})
        atomic(work/'done.json',{'at':time.time(),'rows':len(records)})
        atomic(work/'workflow.json',{'heartbeat':time.time(),'status':'complete','rows':len(records)})
    except BaseException as exc:
        atomic(work/'abort.json',{'at':time.time(),'error':repr(exc)});raise
    finally:
        if reserved:subprocess.run([sys.executable,str(repo/'cluster/gpu_slots.py'),'release','--owner',str(os.getpid()),'--gpu',str(review['gpu'])],check=True)


if __name__=='__main__':main()
