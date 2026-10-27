"""Three exact-recipe updates for memory capacity only; never publish weights."""
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time
import traceback


class CapacityComplete(BaseException):
    pass


def main():
    root=Path.home()/'kevin'
    work=root/'eval_videos/walk_late_biped_capacity_probe_20261027'
    work.mkdir(parents=True,exist_ok=True)
    result_path=work/'result.json'
    assert not result_path.exists(),'Inspect the existing probe before another launch'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert len(os.sched_getaffinity(0))==8
    repo=root/'Isaac-GR00T'
    dataset=root/'datasets/walk200_v3_late1s_20261024_r3/lerobot/ds'
    info=json.loads((dataset/'meta/info.json').read_text())
    assert info['total_episodes']==493 and info['total_frames']==307730
    revision=Path.home()/'.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/refs/main'
    assert revision.read_text().strip()=='2fc962b973bccdd5d8ce4f67cc63b264d6886495'
    slot=root/'wt/locomotion/DreamVLA/cluster/gpu_slots.py'
    reserved=subprocess.check_output(['python3',str(slot),'reserve','--owner',str(os.getpid()),
        '--count','1','--mb','30000','--gpu','0'],text=True).strip()
    assert reserved=='0','Capacity probe has no reviewed GPU reservation'
    result={'pid':os.getpid(),'started':time.time(),'purpose':'memory capacity only; not a study fine-tune',
        'batch_size':16,'gradient_accumulation_steps':1,'schedule_max_steps':109629,
        'learning_rate':0.0001,'steps_to_measure':3,'dataset':str(dataset),
        'pretrained_revision':revision.read_text().strip(),'checkpoint_saving':'forbidden',
        'cpu_affinity':sorted(os.sched_getaffinity(0))}
    torch=None
    try:
        sys.path.insert(0,str(repo))
        os.chdir(repo)
        import torch
        from transformers import Trainer,TrainerCallback
        torch.cuda.set_device(0)
        result['gpu']=torch.cuda.get_device_name(0)
        result['gpu_total_bytes']=torch.cuda.get_device_properties(0).total_memory
        timings=[]

        class ProbeCallback(TrainerCallback):
            def on_train_begin(self,args,state,control,**kwargs):
                assert state.global_step==0,'Do not resume old weights in a capacity probe'
                result['actual_training_args']={key:getattr(args,key) for key in
                    ['per_device_train_batch_size','gradient_accumulation_steps','max_steps',
                     'learning_rate','bf16','tf32','gradient_checkpointing','dataloader_num_workers']}

            def on_step_begin(self,args,state,control,**kwargs):
                torch.cuda.synchronize();self.begin=time.monotonic()

            def on_step_end(self,args,state,control,**kwargs):
                torch.cuda.synchronize();timings.append(time.monotonic()-self.begin)
                result['completed_steps']=state.global_step
                if state.global_step>=3:raise CapacityComplete()

        original_train=Trainer.train

        def forbid_save(*args,**kwargs):
            raise AssertionError('Capacity probe must not write checkpoint weights or optimizer state')

        def train(trainer,*args,**kwargs):
            assert trainer.args.per_device_train_batch_size==16
            assert trainer.args.gradient_accumulation_steps==1
            assert trainer.args.max_steps==109629 and trainer.args.learning_rate==0.0001
            trainer.save_model=forbid_save
            trainer._save_checkpoint=forbid_save
            trainer.add_callback(ProbeCallback())
            torch.cuda.reset_peak_memory_stats()
            return original_train(trainer,*args,**kwargs)

        Trainer.train=train
        sys.argv=[str(repo/'gr00t/experiment/launch_finetune.py'),
            '--base-model-path','nvidia/GR00T-N1.7-3B','--dataset-path',str(dataset),
            '--embodiment-tag','unitree_g1_sonic','--num-gpus','1','--global-batch-size','16',
            '--gradient-accumulation-steps','1','--learning-rate','0.0001','--dataloader-num-workers','4',
            '--output-dir',str(work),'--experiment-name','unsaved_capacity_probe',
            '--max-steps','109629','--save-steps','5000','--save-total-limit','1']
        try:
            runpy.run_path(str(repo/'gr00t/experiment/launch_finetune.py'),run_name='__main__')
        except CapacityComplete:
            result.update(status='capacity_fits',completed_steps=3,
                measured_step_seconds=timings,steady_step_seconds=statistics.median(timings[1:]))
        except torch.cuda.OutOfMemoryError as error:
            result.update(status='out_of_memory',capacity_fits=False,
                out_of_memory_message=str(error),out_of_memory_traceback=traceback.format_exc())
        else:
            raise AssertionError('Official trainer returned without the three-step capacity callback')
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error)[-1600:])
        raise
    finally:
        if torch is not None and torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result['completed_at']=time.time()
        result['checkpoint_files_written']=[str(p.relative_to(work)) for pattern in
            ('*.safetensors','optimizer.pt','pytorch_model*.bin') for p in work.rglob(pattern)]
        temporary=result_path.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(result_path)
        for child in multiprocessing.active_children():
            child.terminate();child.join(timeout=5)
            if child.is_alive():child.kill();child.join(timeout=5)
        subprocess.run(['python3',str(slot),'release','--owner',str(os.getpid()),'--gpu','0'],check=True)
        print(json.dumps(result),flush=True)
    assert not result['checkpoint_files_written']


if __name__=='__main__':main()
