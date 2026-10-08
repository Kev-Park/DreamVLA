"""Production launcher with the already exercised action-head compiler backend.

Keep the original model, optimizer, parameter names, CLI and checkpoint format.
No benchmark, device repartitioning, or asynchronous data transformation.
"""
import runpy
from pathlib import Path

import torch
import gr00t.experiment.experiment as experiment


class CompiledTrainer(experiment.Gr00tTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve ATen random operations and the original state_dict keys.
        torch._inductor.config.fallback_random = True
        self.model.action_head.forward = torch.compile(
            self.model.action_head.forward, mode="reduce-overhead")
        print("Production action-head compilation enabled; training recipe unchanged", flush=True)


experiment.Gr00tTrainer = CompiledTrainer
runpy.run_path(
    str(Path(experiment.__file__).with_name("launch_finetune.py")),
    run_name="__main__",
)
