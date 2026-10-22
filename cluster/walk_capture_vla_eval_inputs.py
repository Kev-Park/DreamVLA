"""Trace unchanged accepted-eval policy calls, inputs, outputs and RNG state."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

EVAL_SHA='30a4cc7a7e15bb6df724f4a55cb1d14134f03715f64758c64e6a57f40b59f63c'
DECODER_SHA='19a4288b0a61094b4711163ce6ee648cb060f2745445e6a6a182cc53279b5dbe'


def build_eval_ast(source):
    original=ast.parse(source)
    class Trace(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='policy.get_action' and len(node.args)==1 and isinstance(node.args[0],ast.Name) and node.args[0].id=='vla_obs':
                assert not node.keywords
                self.count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_capture_policy_action',ctx=ast.Load()),
                    args=[ast.Name(id=name,ctx=ast.Load()) for name in ['policy','vla_obs','ep','mid','step','unw','obs']],keywords=[]),node)
            return node
    trace=Trace();tree=trace.visit(ast.parse(source));assert trace.count==1
    class Restore(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_capture_policy_action':
                return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='policy',ctx=ast.Load()),attr='get_action',ctx=ast.Load()),
                    args=[ast.Name(id='vla_obs',ctx=ast.Load())],keywords=[]),node)
            return self.generic_visit(node)
    restored=Restore().visit(ast.parse(ast.unparse(tree)))
    assert ast.dump(original,include_attributes=False)==ast.dump(restored,include_attributes=False)
    return ast.fix_missing_locations(tree)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--capture-root',type=Path,required=True)
    capture_args,remaining=parser.parse_known_args()
    repo=Path(__file__).resolve().parent.parent
    original=repo/'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'
    decoder=original.parent/'vla_sonic/token_action_wrapper.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==EVAL_SHA
    assert hashlib.sha256(decoder.read_bytes()).hexdigest()==DECODER_SHA
    capture=capture_args.capture_root.resolve()
    assert capture.is_relative_to((Path.home()/'kevin/eval_videos').resolve()) and not capture.exists()
    capture.mkdir()
    tree=build_eval_ast(original.read_text());state={'queries':0,'bytes':0}
    def get_action(policy,observation,episode,motion,step,unwrapped,task_observation):
        import random,time
        import numpy as np
        import torch
        arrays={}
        def pack(prefix,value):
            if hasattr(value,'items'):
                for key,child in value.items():pack(prefix+'/'+str(key),child)
            else:
                a=value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.array(value,copy=True)
                assert a.dtype.kind!='O',prefix
                if a.dtype.kind in 'biufc':assert np.isfinite(a).all(),prefix
                arrays[prefix]=a
        cpu_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state(0).clone()
        numpy_before=np.random.get_state();python_before=random.getstate()
        pack('observation',observation)
        result=policy.get_action(observation)  # Delegate exactly once, with the original object.
        cpu_after=torch.get_rng_state().clone();cuda_after=torch.cuda.get_rng_state(0).clone()
        numpy_after=np.random.get_state();python_after=random.getstate()
        pack('prediction',result[0] if isinstance(result,tuple) else result)
        pack('task_observation',task_observation)
        pack('rng/torch_cpu_before',cpu_before);pack('rng/torch_cuda_before',cuda_before)
        arrays['rng/numpy_keys_before']=numpy_before[1]
        robot=unwrapped.scene['robot'];obj=unwrapped.scene['object']
        for name,value in [('root_pos',robot.data.root_pos_w),('root_quat',robot.data.root_quat_w),
            ('joint_pos',robot.data.joint_pos),('joint_vel',robot.data.joint_vel),('object_pos',obj.data.root_pos_w),('object_quat',obj.data.root_quat_w)]:pack('scene/'+name,value)
        current=unwrapped.episode_length_buf*unwrapped.step_dt+unwrapped.start_motion_times.to(device=unwrapped.device,dtype=torch.float32)
        metadata={'episode':int(episode),'motion_id':int(motion),'step':int(step),'motion_time':float(current[0].item()),
            'step_dt':float(unwrapped.step_dt),'joint_names':list(robot.joint_names),'eval_source_sha256':EVAL_SHA,
            'decoder_source_sha256':DECODER_SHA,'decoder_contract':'previous_full_fsq',
            'numpy_rng_before':[numpy_before[0],int(numpy_before[2]),int(numpy_before[3]),float(numpy_before[4])],
            'python_rng_before':python_before,'recorded_at':time.time()}
        arrays['metadata_json']=np.asarray(json.dumps(metadata))
        target=capture/f'm{int(motion):03d}_ep{int(episode):03d}_s{int(step):05d}.npz'
        assert not target.exists()
        temporary=target.with_suffix('.npz.partial')
        with temporary.open('wb') as stream:np.savez(stream,**arrays)  # Uncompressed ZIP_STORED records.
        temporary.replace(target)
        assert torch.equal(cpu_after,torch.get_rng_state()) and torch.equal(cuda_after,torch.cuda.get_rng_state(0))
        after=np.random.get_state()
        assert numpy_after[0]==after[0] and np.array_equal(numpy_after[1],after[1]) and numpy_after[2:]==after[2:]
        assert python_after==random.getstate()
        state['queries']+=1;state['bytes']+=target.stat().st_size
        if state['queries']%16==0:print('[input-capture] '+json.dumps(state),flush=True)
        return result
    module=types.ModuleType('__main__');module.__file__=str(original)
    module.__dict__['_capture_policy_action']=get_action
    sys.modules['__main__']=module;sys.argv=[str(original),*remaining];sys.path.insert(0,str(original.parent))
    try:
        try:exec(compile(tree,str(original),'exec'),module.__dict__)
        except SystemExit as exc:
            if exc.code not in [0,None]:raise
        assert state['queries']>0
        (capture/'capture_complete.json').write_text(json.dumps({**state,'original_eval_sha256':EVAL_SHA,
            'decoder_sha256':DECODER_SHA,'logging_does_not_advance_rng':True,'compression':'none','policy_calls_per_query':1},indent=2)+'\n')
    except BaseException as exc:
        (capture/'capture_abort.json').write_text(json.dumps({**state,'error':repr(exc)},indent=2)+'\n');raise


if __name__=='__main__':main()
