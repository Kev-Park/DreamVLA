"""Independently audit captured policy queries against their executed rollouts."""
import argparse, collections, hashlib, json, os, re, time, zipfile
from pathlib import Path


def main():
    import numpy as np
    parser=argparse.ArgumentParser();parser.add_argument('--capture-model',choices=['current_r2','historical_r2'],default='current_r2');args=parser.parse_args()
    name='walk_gap_eval_input_capture20261022' if args.capture_model=='current_r2' else 'walk_gap_hist_eval_input_capture20261022'
    root=Path.home()/'kevin';work=root/'dline'/name
    output=root/'eval_videos'/work.name
    launch=json.loads((work/'operator_review.json').read_text())
    controller=(work/'controller.log').read_text()
    assert all(f'EXIT capture_shard{k} code=0' in controller for k in range(2))
    inventory=[];episodes=[];plan=[];total=0;max_scene_error=0.;max_time_error=0.
    for shard in range(2):
        folder=output/f'inputs_b{shard}'
        assert not list(folder.glob('*.partial')) and not (folder/'capture_abort.json').exists()
        files=sorted(folder.glob('*.npz'));grouped=collections.defaultdict(list)
        for file in files:
            match=re.fullmatch(r'm(\d+)_ep(\d+)_s(\d+).npz',file.name);assert match,file
            mid,episode,step=map(int,match.groups());grouped[episode].append((step,mid,file))
        assert set(grouped)==set(range(5))
        grasp_path=output/f'grasp_b{shard}.pkl.npz'
        with np.load(grasp_path,allow_pickle=False) as grasp:
            assert {int(k.split('_')[0][2:]) for k in grasp.files}==set(range(5))
            for episode,queries in sorted(grouped.items()):
                queries.sort();prefix=f'ep{episode}_';mid=int(grasp[prefix+'motion'])
                end=int(grasp[prefix+'end_step']);grab=int(grasp[prefix+'grab_step'])
                close=grasp[prefix+'cmd_close'];assert len(close)==end
                assert [q[0] for q in queries]==list(range(0,end,8))
                assert all(q[1]==mid for q in queries)
                trajectory_path=output/f'b{shard}_ep{episode}_m{mid}_traj.npz'
                with np.load(trajectory_path,allow_pickle=False) as trajectory:
                    assert len(trajectory['motion_time'])==end-1
                    for step,_,file in queries:
                        with zipfile.ZipFile(file) as archive:assert all(i.compress_type==zipfile.ZIP_STORED for i in archive.infolist())
                        with np.load(file,allow_pickle=False) as z:
                            md=json.loads(str(z['metadata_json']))
                            assert (md['episode'],md['motion_id'],md['step'])==(episode,mid,step)
                            assert md['decoder_contract']=='previous_full_fsq'
                            assert md['eval_source_sha256']==launch['source_sha256']['Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py']
                            assert md['decoder_source_sha256']==launch['source_sha256']['Training/scripts/reinforcement_learning/rsl_rl/vla_sonic/token_action_wrapper.py']
                            assert z['observation/video/ego_view'].shape==(1,1,480,640,3)
                            assert z['observation/video/ego_view'].dtype==np.uint8 and z['observation/video/ego_view'].max()>0
                            assert z['prediction/motion_token'].shape==(1,40,64)
                            assert z['prediction/right_hand_joints'].shape==(1,40,7)
                            assert z['rng/torch_cpu_before'].dtype==z['rng/torch_cuda_before'].dtype==np.uint8
                            for key in z.files:
                                array=z[key];assert array.dtype.kind!='O',key
                                if array.dtype.kind in 'biufc':assert np.isfinite(array).all(),key
                            count=min(8,end-step)
                            hand=np.asarray(z['prediction/right_hand_joints'],dtype=np.float32)[0,:count]
                            predicted=np.abs(hand).mean(axis=1)
                            assert np.array_equal(predicted>.6,close[step:step+count]),(mid,step)
                            assert np.allclose(predicted,grasp[prefix+'pred_meanq'][step:step+count],atol=1e-7,rtol=1e-6)
                            if step:
                                # Capture is before this step; trajectory rows are after each step.
                                for scene,logged in [('root_pos','root_pos'),('root_quat','root_quat'),('object_pos','obj_pos'),('object_quat','obj_quat')]:
                                    error=float(np.max(np.abs(z['scene/'+scene][0]-trajectory[logged][step-1])))
                                    max_scene_error=max(max_scene_error,error);assert error<1e-6,(mid,step,scene,error)
                                error=abs(md['motion_time']-float(trajectory['motion_time'][step-1]))
                                max_time_error=max(max_time_error,error);assert error<1e-6,(mid,step,error)
                        inventory.append({'path':str(file.relative_to(output)),'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
                firstclose=int(np.flatnonzero(close)[0]) if close.any() else None
                reasons=collections.defaultdict(list)
                for label,target in [('start',0),('pre_grab_48',grab-48),('pre_grab_24',grab-24),('grab',grab),('post_grab_24',grab+24),('post_grab_48',grab+48),('last',end-1),('first_close',firstclose)]:
                    if target is not None and 0<=target<end:reasons[(target//8)*8].append(label)
                for step,labels in sorted(reasons.items()):
                    file=folder/f'm{mid:03d}_ep{episode:03d}_s{step:05d}.npz'
                    plan.append({'path':str(file.relative_to(output)),'shard':shard,'episode':episode,'motion_id':mid,'query_step':step,'grab_step':grab,'end_step':end,'first_close_step':firstclose,'reasons':labels,'executed_steps':min(8,end-step)})
                episodes.append({'shard':shard,'episode':episode,'motion_id':mid,'end_step':end,'grab_step':grab,'queries':len(queries),'first_close_step':firstclose,'close_frames':int(close.sum())})
        total+=len(files)
    assert sorted(r['motion_id'] for r in episodes)==sorted(launch['motion_ids'])
    result={'at':time.time(),'capture_queries':total,'capture_bytes':sum(r['bytes'] for r in inventory),'episodes':episodes,
        'all_queries_complete':True,'all_executed_hand_commands_match':True,'all_capture_arrays_finite':True,
        'all_records_uncompressed':True,'max_pre_step_scene_abs_error':max_scene_error,'max_motion_time_error':max_time_error,
        'normal_worker_exit_codes':[0,0],'input_output_decoder_changed':False,'inventory':inventory,
        'receipt_status':'independently_verified_data_audit',
        'capture_model':args.capture_model,
        'original_abort_preserved':json.loads((work/'abort.json').read_text()) if (work/'abort.json').exists() else None,
        'limits':'Ten outcome-stratified diagnostic replicas, not an unbiased success estimate. No population rate inferred.'}
    (output/'capture_independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'replay_sample_plan.json').write_text(json.dumps({'at':time.time(),'points':plan,'selection':'Query0, near reference grasp, final query, first actual close; deduplicated per episode.'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'}))


if __name__=='__main__':main()
