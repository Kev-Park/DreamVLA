# Running DreamVLA on Adroit (Princeton RC)

Adroit is RHEL 9 (glibc 2.34) with no internet on compute nodes. Isaac Sim 5.1 needs glibc 2.35, so
everything Isaac runs inside an Apptainer container that provides only the Ubuntu 22.04 userland;
`dreamcontrol_51` itself is a normal conda env (pip-installed Isaac) living on scratch and is
identical to biped's. The `~/kevin` contract from `WORKTREES.md` holds unchanged.

## Files here → where they live on the cluster

| file | install location | role |
|---|---|---|
| `isaac` | `~/kevin/bin/isaac` (on `PATH`) | enter the container; opens the proxy tunnel on compute nodes |
| `isaac.rc` | `~/kevin/bin/isaac.rc` | sourced in every container shell: conda, proxy, Vulkan ICD, `DREAMCONTROL_PY` |
| `isaac-base.def` | `~/kevin/sif/isaac-base.def` → `apptainer build --fakeroot isaac-base.sif` | the Ubuntu 22.04 + Vulkan/GL runtime layer (~250 MB) |
| `nvidia_icd.json` | `~/kevin/etc/vulkan/icd.d/nvidia_icd.json` | Vulkan ICD manifest; `--nv` binds the driver libs but not this |
| `adroit.sbatch` | run from the checkout | batch template |

After `git pull`, keep the cluster copies as symlinks into the checkout so there is one source:
`ln -sf ~/kevin/DreamVLA/cluster/adroit/{isaac,isaac.rc} ~/kevin/bin/` etc.

## Layout (all on `/scratch/network/$USER`, home is 10 GB)

```
~/kevin -> /scratch/network/$USER/kevin        DreamVLA, holosoma, GR00T-WholeBodyControl, sonic/, ref_motions/, wt/, bin/, sif/, etc/
~/.cache/ov, ~/.local/share/ov, ~/.nv, ~/.cache/huggingface, ~/.holosoma_deps -> /scratch/network/$USER/...
conda: RC module anaconda3/2025.6 (bound read-only into the container); envs in ~/kevin/conda via ~/.condarc envs_dirs
```

## Every session

```bash
# once per login node boot: the HTTP proxy compute nodes tunnel through (tmux `conn` on adroit5)
python3 -m proxy --port 8899 --hostname 127.0.0.1 --num-acceptors 1 --num-workers 1 --threadless

salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gres=gpu:1 --constraint=gpu80 --time=4:00:00
isaac                       # (isaac) prompt = inside the container
conda activate dreamcontrol_51
cd ~/kevin/DreamVLA/Training && python scripts/... 
```

- `--constraint=gpu80` is required: plain `--gres=gpu:1` hands out a MIG slice, which has no
  Vulkan/PhysX-GPU. Never pass `--partition`.
- host → `tmux` → `isaac`, never `tmux` inside the container.
- `pip`/`conda`/`python` for `dreamcontrol_51` only ever run inside `isaac` (glibc).
- Assets stream from NVIDIA's S3 through the proxy (omni.client honours HTTP proxies, not SOCKS)
  and are cached in `~/.cache/ov`; the 140 GB offline asset pack is not needed.

## Env install notes (`dreamcontrol_51` from `new_requirements.txt`)

`pip install --no-deps -r` (drop the `-e git+` lines, `decoupled_wbc`, `jaxls`, `pyroki`; install
those `-e`/at the frozen commits afterwards); `flatdict` needs `setuptools<80 --no-build-isolation`;
then `./isaaclab.sh --install`, which **downgrades `nvidia-cudnn-cu12`** to torch's pin and breaks
JAX (`DNN library initialization failed`) — restore with
`pip install --no-deps nvidia-cudnn-cu12==9.21.0.82`. After any pip install in this env:

```bash
diff <(grep -E '^nvidia-|^torch|^jax|^numpy' new_requirements.txt | sort) <(pip freeze | grep -E '^nvidia-|^torch|^jax|^numpy' | sort)
```

`hsretargeting`: `holosoma/scripts/setup_retargeting.sh` then `pip install --no-deps -r cluster/hsretargeting_freeze.txt`
(biped's freeze, so retargeted datasets stay solver-comparable: cvxpy 1.8.1 / mujoco 3.4.0).
