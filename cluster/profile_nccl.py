"""Small communication-only diagnostic, independent of training model state."""
import json
import os
import time
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
try:
    for mib in (4, 64, 256):
        tensor = torch.ones(mib*1024*1024//4, device='cuda', dtype=torch.float32)
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        seconds = (time.perf_counter()-start)/30
        if dist.get_rank() == 0:
            print(json.dumps({'mib':mib, 'seconds':seconds, 'bus_GB_per_sec':mib*1024**2/seconds/1e9}), flush=True)
finally:
    dist.destroy_process_group()
