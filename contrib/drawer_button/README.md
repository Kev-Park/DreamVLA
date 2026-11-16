# Drawer and standing-button research pipeline

An isolated contribution for the G1 drawer-opening and standing button-press
tasks: kinematic references → SONIC residual PPO expert → successful ego-camera
demonstrations → LeRobot v2.1 → GR00T → closed-loop evaluation.

This package does not replace DreamVLA's existing `vla_sonic` implementation or
register over its Gym task IDs. It contains source, configuration examples,
documentation and synthetic tests only. No experimental results, datasets,
videos, pretrained weights, simulator installation or credentials are included.

## Status and scope

This is a **research release candidate**, not a new validated benchmark result.
The task/controller/recorder sources were extracted from the drawer/button
workspace. CPU contracts and CLI checks are provided; the reorganized package
still needs an allocated-GPU integration smoke test against a clean checkout.
Do not attribute historical experiment scores to this refactor without rerunning
it. See [limitations](docs/limitations.md) before using DAgger or comparing runs.

The fine-tuning launcher uses NVIDIA's native online trainer. It is **not** a
bitwise reproduction of the historical cached-feature/action-head experiments.
The package includes the separately saved head-loading utility, but does not
silently apply action-encoder rescaling or a task-trained warm start.

## Installation

Use separate existing environments for Isaac, OmniControl and GR00T; keep a
single Isaac Sim installation. Follow the upstream projects' installation and
license requirements first. Do not install the GR00T stack into Isaac's Python.
See [setup and external assets](docs/setup.md) and the exact source revisions in
[dependencies.json](configs/dependencies.json).

From the repository root, install this small package into each compatible
Python environment that will run its stages:

```bash
python -m pip install --no-deps -e ./contrib/drawer_button
python -m dreamvla_drawer_button --help
```

NumPy and PyYAML are the core dependencies; simulation/model/export dependencies
come from their respective upstream environments. `--help` does not load Kit or
model weights. Legacy OmniControl/Python 3.7 must use `PYTHONPATH` to run
`sample-human` from source rather than installing the Python ≥3.10 package.

## Workflow

Examples use an ignored runtime directory and explicit paths. Activate the
environment named in each step. Replace reference/base-model paths with assets
you obtained separately. These commands do not run automatically on install.

```bash
export DREAMVLA_ROOT="$PWD"
export DREAMVLA_RUN="$PWD/contrib/drawer_button/.runtime/drawer"
export DREAMVLA_ASSET_ROOT="$PWD/contrib/drawer_button/.runtime/assets"
```

1. Prepare references in the CPU IK environment. `prepare-references` creates a
   seeded, nested target plan and hashes the generators/base motions. It does
   not generate successful physical demonstrations. The drawer mixes standing
   and squatting references; the button task is standing-only.

```bash
python -m dreamvla_drawer_button prepare-references --task drawer --count 200 \
  --base-stand "$DREAMVLA_ROOT/TrajGen/sample/Open_Drawer_sim/stand_still.pkl" \
  --base-squat "$DREAMVLA_ROOT/TrajGen/sample/Open_Drawer_sim/squat.pkl" \
  --output "$DREAMVLA_RUN/generated" --plan "$DREAMVLA_RUN/reference_plan.json"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -m dreamvla_drawer_button \
  refs-drawer-stand --point-plan "$DREAMVLA_RUN/reference_plan.json"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -m dreamvla_drawer_button \
  refs-drawer-squat --point-plan "$DREAMVLA_RUN/reference_plan.json"
```

   The outputs are in `generated/stand` and `generated/squat`. Place the selected
   reference files in one training directory and generate a separate held-out
   library with a different target seed. Reference pickles are trusted inputs;
   do not load untrusted files. Button planning uses `--task standing_button`
   and `--base-button PATH`; run `refs-button --plan PLAN` afterward. Button IK
   currently refuses an existing output rather than claiming resumability.

2. In the Isaac environment, prepare assets and check dependencies. Run simulation
   commands from `Training`, as required by DreamControl's common motion code.
   Choose an allocated GPU explicitly; physical GPU indices are never hardcoded.

```bash
cd "$DREAMVLA_ROOT/Training"
python -m dreamvla_drawer_button doctor --stage isaac
CUDA_VISIBLE_DEVICES=0 python -m dreamvla_drawer_button train-ppo \
  --task drawer --references /path/to/train_references --output "$DREAMVLA_RUN/ppo" \
  --iterations 9000 --num-envs 2048 --device cuda:0
```

   Set `--iterations 3000` to request that paper budget; it is not equivalent
   to 9,000 iterations. Start with `--iterations 2 --num-envs 2` in a different
   output for a smoke test. Task-trained checkpoint weights are only loaded
   with explicit `--resume` and an unchanged run contract.

3. Create one immutable evaluation bank, qualify the expert, then collect.
   Replace the checkpoint filename with the actual saved file. The preparation
   step rejects train/eval reference-content overlap.

```bash
python -m dreamvla_drawer_button prepare-run --task drawer \
  --train-references /path/to/train_references --eval-references /path/to/heldout_references \
  --teacher-checkpoint "$DREAMVLA_RUN/ppo/model_8999.pt" --output "$DREAMVLA_RUN/config"
CUDA_VISIBLE_DEVICES=0 python -m dreamvla_drawer_button record \
  --config-dir "$DREAMVLA_RUN/config" --mode teacher_eval --episodes 100 \
  --track repeatability --output "$DREAMVLA_RUN/expert_eval" --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -m dreamvla_drawer_button record \
  --config-dir "$DREAMVLA_RUN/config" --mode collect --episodes 200 --max-attempts 2000 \
  --output "$DREAMVLA_RUN/raw" --device cuda:0
```

   Collection retries toward a success quota within the explicit attempt cap.
   Evaluation always runs exactly the requested bank prefix, including failures.
   Review expert quality before collection; there is no hidden 80% gate or
   automatic continuation into GR00T training.

4. In SONIC's data-collection environment, convert only accepted, matching-scene
   episodes. Video/array/provenance checks run before opening the output dataset.

```bash
python -m dreamvla_drawer_button convert "$DREAMVLA_RUN/raw" "$DREAMVLA_RUN/lerobot"
```

5. In the GR00T environment, inspect the frame-based budget before training.
   This recipe freezes the VLM, fits state/action normalization from the selected
   data, trains the action components, and predicts absolute full SONIC tokens.

```bash
python -m dreamvla_drawer_button finetune --dataset "$DREAMVLA_RUN/lerobot" \
  --base-model /path/to/GR00T-N1.7-3B --output "$DREAMVLA_RUN/student" --dry-run
python -m dreamvla_drawer_button finetune --dataset "$DREAMVLA_RUN/lerobot" \
  --base-model /path/to/GR00T-N1.7-3B --output "$DREAMVLA_RUN/student" \
  --epochs 5.7 --batch-size 16 --learning-rate 1e-4 --scheduler constant
```

   Multi-GPU uses `torchrun --standalone --nproc-per-node N -m
   dreamvla_drawer_button.finetune ...`; batch size remains the effective batch
   across ranks and gradient accumulation. No scheduler, warm start or extra
   initialization is inferred from historical experiment names.

6. Serve the student in the GR00T environment, then evaluate in the Isaac
   environment. Use separate outputs and ports/server processes for concurrent
   evaluation tracks. Both processes must run on the same host (loopback bridge).

```bash
CUDA_VISIBLE_DEVICES=0 python -m dreamvla_drawer_button serve \
  --model-path "$DREAMVLA_RUN/student" --port 5777 --device cuda:0
# Separate terminal, Isaac environment, cwd Training:
CUDA_VISIBLE_DEVICES=1 python -m dreamvla_drawer_button record \
  --config-dir "$DREAMVLA_RUN/config" --mode eval --episodes 100 --server-port 5777 \
  --track repeatability --output "$DREAMVLA_RUN/student_repeatability" --device cuda:0
```

   Repeat with `--track fresh_start`. Read the track definitions in
   [the protocol](docs/protocol.md); do not describe seed-only variation as a
   new randomized-scene test. Model artifact hashes and bank hashes are recorded
   so resumed evaluation cannot silently combine different students.

## Tests and layout

```bash
python -m pip install -e './contrib/drawer_button[test]'
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest contrib/drawer_button/tests
ruff check contrib/drawer_button
ruff format --check contrib/drawer_button
```

The optional Torch tests run on CPU and skip when Torch is unavailable. They
check contracts, not task performance. No test launches training, downloads a
model, edits a dataset, or allocates a GPU.

`dreamvla_drawer_button/isaac/` holds isolated task/controller snapshots;
`contracts.py` defines the scientific data interface; `run_config.py` manages
reference/evaluation manifests; `rollout.py`, `convert.py`, `finetune.py` and
`serve.py` are the stage entrypoints. `dagger.py` and the history proxy are
experimental utilities, not a validated multi-round campaign launcher.

Licensing and upstream attribution are in [NOTICE.md](NOTICE.md) and `LICENSE`.
