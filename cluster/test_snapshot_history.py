"""Snapshot real observation rings, restore after reset, then continue appending."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'Training/scripts/reinforcement_learning/rsl_rl/collect_sonic_adapter.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
names = {'_wrapper_owning', 'rollout_snapshot', 'rollout_restore'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
nodes += [n for n in tree.body if isinstance(n, ast.Assign) and
          any(isinstance(t, ast.Name) and t.id in ('_SNAP_WRAPPER_TENSORS','_SNAP_ENV_TENSORS') for t in n.targets)]
scope = {'torch':torch}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
spec = importlib.util.spec_from_file_location('real_ring', ROOT/'Training/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class SnapshotHistory(unittest.TestCase):
    def test_continue_after_restore(self):
        for pushes in (0,2,7):
            with self.subTest(pushes=pushes):
                ring = module.CircularBuffer(3,2,'cpu')
                control = module.CircularBuffer(3,2,'cpu')
                for i in range(pushes):
                    value = torch.full((2,2),float(i))
                    ring.append(value);control.append(value)
                if pushes:
                    ring.reset([1]);control.reset([1])
                manager = SimpleNamespace(_group_obs_term_history_buffer={'policy':{'joint':ring}},
                                          compute=lambda **kw: {})
                scene = SimpleNamespace(get_state=lambda **kw:{},articulations={},
                                        reset_to=lambda *a,**kw:None,write_data_to_sim=lambda:None)
                env = SimpleNamespace(scene=scene,sim=SimpleNamespace(forward=lambda:None),
                                      num_envs=2,device='cpu',observation_manager=manager)
                env.unwrapped = env
                for name in scope['_SNAP_WRAPPER_TENSORS']:
                    setattr(env,name,torch.zeros((2,1)))
                saved = scope['rollout_snapshot'](env)
                for i in range(5):ring.append(torch.full((2,2),-100.))
                ring.reset()
                scope['rollout_restore'](env,saved)
                for i in range(4):
                    value = torch.full((2,2),float(100+i))
                    ring.append(value);control.append(value)
                    torch.testing.assert_close(ring.buffer,control.buffer,rtol=0,atol=0)
                    torch.testing.assert_close(ring.current_length,control.current_length,rtol=0,atol=0)
                    torch.testing.assert_close(ring[torch.tensor([0,1])],control[torch.tensor([0,1])],rtol=0,atol=0)

if __name__ == '__main__':unittest.main()
