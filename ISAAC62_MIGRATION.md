# g1 / locomotion consolidation and Isaac Sim 6.2

The target is one maintained implementation containing both branches' capabilities,
running on Isaac Sim 6.2 with preserved policy inputs, outputs, physics recipes,
data formats, collection semantics, metrics, and rendering. A successful import
or a file-surface audit is not completion.

## Recorded parents

- g1: `0505979bf6058b772edb1e7436bb757461aba88f`
- locomotion: `bb9d874aa8e528bc3f5d53c0421fca4a2d4ff9b1`
- Current integration branch: `g1`. No new branch was created.

## Preservation gates

| Capability | Required evidence |
| --- | --- |
| Registered tracking, pick, drawer, button, walking, camera and motion-only tasks | Compare task registrations to both parents; instantiate each relevant configuration and representative environment |
| Nominal SONIC, residual transforms, direct-token ablation, GR00T and recorded-action replay | Load representative saved checkpoints; finite observations/actions; preserve ordering, normalization, FSQ and finger commands |
| Residual training and GR00T fine-tuning | Short optimizer updates, checkpoint save/reload and resume; no full training run required |
| Expert collection and DAgger | Small passing/failing/recovery cases, broad and late uniform sampling, explicit motion selection, sampling retries, skip-start, snapshot/replay, saved expert labels and finger commands |
| HDF5 / LeRobot conversion and subset selection | Small fixtures through conversion, merge and normalization-statistics checks; existing data must remain reusable |
| Evaluation and metrics | Short evaluations for each policy path; numerical success criteria, world/root MPJPE, episode accounting and termination coverage |
| Motion/reference generation | Static and stitched own-motion references, alignment, foot-slip handling, augmentation and FK/replay checks |
| Rendering | Third-person and ego cameras, clean videos and overlays, tracking, reference playback/PD, motion-only scenes; verify nonempty files, frame counts, finite images, nonblack output and paired numerical state traces |
| Resource coordination and recovery | Both allocator APIs, manual placements/pauses, live-memory credit, explicit GPU selection, failure/continuation and archive-integrity tests; no account-wide enforcement |
| Scientific equivalence | Explicit joint/body/quaternion ordering, actuator/solver limits, dt/decimation, reset timing, camera timing and original decoder; short matched baseline evaluations and user review of clips |

## Merge choices requiring particular attention

- Keep g1's current default residual transform (`additive_free`) and retain all
  locomotion choices, including `direct`. Reproduction commands must specify the
  original transform explicitly; a CLI default is not checkpoint metadata.
- g1 places accepted object mass and hand gains directly in its pick environment.
  Its shared override helpers are opt-in. Retain this behavior so unrelated task
  assets are not silently modified; audit walking launch recipes against their
  effective legacy mass/gains before accepting runtime equivalence.
- Preserve both contact-point and palm-normal overlays, object and motion-only
  replay, and all existing diagnostics. `joint_pos` remains raw articulation order
  for torque diagnostics; `ref_order_joint_pos` stores the separate motion-library
  ordering. Walking replay checks accept both old and merged trace formats.
- The shared collector retains g1's manifest selection and expert-evaluation paths
  alongside locomotion's in-memory scorer, late uniform takeover, retry and recovery
  features. Matching float64 recorded/in-memory verdicts remain asserted.

## Simulator migration gates

1. Inspect the project's vendored framework changes before choosing the maintained
   Isaac 6.2 integration boundary. Do not replace custom task code with stock tasks.
2. Use an isolated Python 3.12 environment. Leave `dreamcontrol_51` unchanged.
   The official Isaac Lab 3.0 baseline targets Sim 6.2 and Torch 2.11; verify the
   exact GR00T/SONIC dependency set, including compiled extensions.
3. Retain Isaac Sim PhysX/RTX. Newton is not an equivalent replacement for these
   existing experiments. Preserve WXYZ at saved-data and learned-policy boundaries
   if the simulator/Lab API uses XYZW internally.
4. Port API ownership, tensor access, actuator limits and camera/render timing
   explicitly; avoid dependency monkeypatches or edits to live cluster checkouts.
5. Run a bounded empty-stage PhysX/RTX probe, then actual robot/task and camera
   probes, before short training/collection/evaluation smoke cases.
6. Bring rendered artifacts into the workspace `out/` for user review. Automated
   checks establish technical validity; the user assesses motion/task appearance.

## Evidence and status

`cluster/audit_capability_union.py` checks the project file, declared symbol,
class member and CLI-option union against both recorded parents. It deliberately does not claim
behavioral equivalence. Runtime, visual and optimization evidence is still required.

Initial merge audit artifacts and test receipts are under the EMBER workspace's
`out/isaac62_migration/`. Foundation physics/rendering now pass the bounded gate
below; project task and training validation remain incomplete. Do not resume the
existing full DAgger study or mark this migration complete from static checks.

The isolated foundation is now installed on Bluesclues and Biped; Biped also has
the project dependencies and five editable packages installed. The actual Biped
project environment passed 82 CPU compatibility contracts and ten cross-interpreter
service fixtures at `f54c1a0`. These do not establish simulation behavior.

The original T20_max_v3/model_4500, T20_max/model_4999 and af60v7/model_4999
residual checkpoints, plus both native SONIC encoder heads and the shared decoder,
also ran on Biped GPU 0 using Torch 2.11/cu128. Checkpoint hashes and synthetic
inputs exactly matched their earlier target CPU baselines; the largest output
difference was 5.72e-6, within the predefined rtol=1e-4/atol=1e-5 gate. During the
probe, only GPU 0 had account-owned contexts; all were released afterward.
Reports and arrays are in workspace
`out/isaac62_migration/isaac62_cuda_checkpoints_c082dde/`. The probes now accept
`--device cuda:0` only with one explicitly visible, separately reviewed GPU.
This is frozen inference compatibility, not closed-loop/task/training validation.

The optional `cluster/isaac62/setup_references.sh` profile adds CPU JAX/JAXLie
without changing any installed simulator/project package versions. At `a5ade21`,
three saved references (static pick, stitched training and held-out walking;
694 frames total) loaded in both the legacy and target environments. Joint,
root, object, hand and ankle geometry matched exactly. Independent XML/URDF
forward kinematics agreed with pytorch_kinematics within 1.81e-7; own-motion
alignment, translated-prefix footfall invariants and JAXLie serialization checks
also passed. Reports, source hashes and numerical arrays are in workspace
`out/isaac62_migration/isaac62_reference_a5ade21/`. This does not validate new
IK optimization, full retargeting or simulator replay. The legacy CPU run logged
its existing CUDA-plugin initialization warning with CUDA hidden; it nevertheless
selected the CPU backend and completed all checks. The new CPU-only JAX profile
did not emit that warning.

The real stitch optimizer also passed a bounded CPU check at `db3f6d7` in both
environments: 1,200 Adam/scheduler updates on three saved grasp-neighborhood
targets, initialized with a 0.03-radian upper-body perturbation. Recomputed final
hand-position error was at most 0.718 mm and orientation error at most 0.097
degrees. Against the legacy runtime, solved joint angles differed by at most
2.15e-6 radians and hand positions by 5.96e-8 meters. The near-identity angle-error
metrics differed by 0.00694 degrees; the full arrays and both logs remain available for inspection in workspace
`out/isaac62_migration/isaac62_reference_ik_db3f6d7/`. Input hashes and initial
arrays match exactly, inputs remain unchanged, and joint limits were checked.
This exercises the actual `solve_upper` implementation, not a substitute solver;
full retargeting, stitched sequence generation and simulator replay remain open.

A full 252-frame stitch/export/reload case ran in both environments at `a4f8523`,
using the saved walking prefix from `StitchMAX200v3/pick_14030.pkl` and its matching
static reference, the unchanged calibrated hand shift and 1,200 IK iterations.
Both standalone checks passed frame/grab arithmetic, unit quaternions, finite
geometry, preserved walking legs/root, held post-footfall lower body, serialization
and unchanged input hashes. This reuses an existing walking prefix; it does not
regenerate the original raw walking retarget.

The cross-runtime strict array-parity gate FAILED for joint angles at
rtol=1e-4/atol=1e-5. Maximum difference is 7.54e-5 radians (0.00432 degrees), in
upper-body joint index 24 at frame 99. Only waist/right-arm IK joints differ;
legs, left arm, root, object and ankle geometry are exact. Maximum hand-position
Euclidean difference is 1.87e-7 meters. Do not replace this failed gate with a
post-hoc relaxed tolerance or claim visual equivalence. Both outputs, full logs,
source hashes, arrays, comparison and joint diagnostics are in workspace
`out/isaac62_migration/isaac62_stitch_a4f8523/`; simulator replay/user review remains
required. Neither environment modified the input references.

The actual GR00T Trainer gate is prepared in
`cluster/isaac62/probe_gr00t_train_resume.py`: batch one, a two-update schedule,
one full model/optimizer checkpoint after update one, and a fresh-process resume.
It retains the real model, dataset loader, collator and AdamW implementation in
the existing GR00T environment. Its warmup-free two-step schedule is a validation
fixture, not a new research run. The production shard defaults must be retained;
an initial tiny-shard fixture produced empty shards before training.

The corrected `2d8c410` run loaded the original walking checkpoint and real saved
LeRobot data, reached forward/backward, then failed while allocating Adam state on
Biped's 32 GB GPU. The model has 1,620,515,968 trainable parameters and 12,576,064,000
bytes of parameter storage. FP32 model weights, trainable gradients and Adam
moments alone need about 32.02 GB, before activations/allocator/driver overhead.
No optimizer update/checkpoint/resume passed; do not count this gate as complete.
The process exited, its reservation was released and account GPU contexts returned
to zero. Logs and the failed receipt are in workspace
`out/isaac62_migration/isaac62_gr00t_training_2d8c410_online/`.
The probe now rejects inadequate free memory before Trainer startup. Keep the
recipe unchanged and use a larger reviewed allocation when available; no packages
or existing GR00T source files were modified.

The real foundation attempts started Isaac 6.2 but failed before valid rendered
artifacts. The raw-stage tensor view needed an explicit USD stage ID. SDK fast
shutdown could also bypass final receipts and return zero on failure; the probe
now persists its receipt first and supplies the intended shutdown exit code.
The prepared fixes still need a passing GPU run. Isaac additionally created a
6 MiB graphics-only context on Biped GPU 1 despite its single-GPU CUDA mask. The
failed probe was terminated and both contexts released. Further Biped Isaac
launches require the user's narrow graphics exception or verified device
isolation under the workspace's strict one-GPU rule. Docker is installed but the
account cannot access its daemon; no permission workaround was attempted.

## Isolated foundation gate

`cluster/isaac62/setup_environment.sh` creates a new environment only at
`~/kevin/envs/dreamcontrol_62`; it refuses to overwrite an existing environment.
It pins Python 3.12, Sim 6.2.0.0 and Torch 2.11/cu128, and does not install the
project's legacy packages or GR00T. The existing `dreamcontrol_51` is untouched.

After reviewing account-wide utilization and obtaining any required CPU exception,
run the setup in an identified tmux session under a bounded CPU allocation. Then
run `cluster/isaac62/probe_physics_render.py` on one manually selected GPU with an
external timeout. It checks GPU PhysX, reset through the public tensor API, fixed
physics-step accounting, rendering without extra physics steps, RGB camera output
and local asset authoring. Its output defaults to `~/kevin/eval_videos/` and must
be pulled to local `out/` for review. It does not load a robot or learned policy.

The foundation gate is prepared but has not run. Project imports, actual robot
physics, model compatibility and all capability gates above remain open.

The Linux x86_64/Python 3.12 foundation dependency set has been resolved against
the actual `manylinux_2_35` wheel platform. The installer uses the resulting
`requirements.sim.lock`, including explicit CUDA 12.8 Torch, TorchVision and
TorchAudio builds. It checks glibc >= 2.35 and at least 50 GiB free before
creating an environment. RHEL 9 hosts require an appropriate container rather
than a host pip installation. Dependency resolution proves package compatibility
only; it does not prove extension loading or runtime behavior.

On 2026-11-11, 47 CPU-only regression tests passed in Biped's existing
`dreamcontrol_51` environment using a Git-exported copy of commit `076f63f` and
one logical CPU, with CUDA disabled. These cover direct-token decoder contracts,
collector in-memory/HDF5 verdict agreement, both reservation interfaces, retry
recovery, checkpoint integrity and account CPU measurement. Receipts and logs are
in workspace `out/isaac62_migration/biped_cpu_contracts.json`. These tests do not
establish compatibility with the new Torch/Sim versions.

NumPy 2 removed `np.float_`, which broke the motion JSON encoder even for array
inputs. The encoder now uses NumPy's public scalar base classes. Local NumPy 2
round-trip tests cover scalar metadata, empty/scalar/multidimensional arrays,
dtype preservation and legacy ndarray JSON. Existing file format is unchanged.

The existing Isaac-GR00T fork pins Torch 2.7.1, NumPy 1.26.4 and compiled
FlashAttention/TorchCodec builds. Do not install it into the foundation environment
with ignored dependency checks. Validate a supported unified dependency set or use
its existing inference-service boundary, preserving diagnostic features as well as
actions. GR00T fine-tuning and teacher-forced offline checks remain separate from
the simulator migration.

## Vendored project port

This integration branch now targets Python 3.12, NumPy 2 and Torch 2.11 in its
vendored package metadata. Existing 5.1 studies must continue using their pinned
source and `dreamcontrol_51`; do not pull this branch into their live checkout.
The matching project dependency set (253 packages) resolves against the pinned
foundation, and the five editable package metadata builds also resolve. This is
not evidence that the packages import or run inside Kit.

`setup_project.sh` installs the vendored framework, custom motion libraries,
RSL-RL 3.0.1 and both SONIC loaders into the isolated foundation environment.
It does not install GR00T or change its model implementation. UV's cache and
managed Python installation are kept under `~/kevin` by default. An alternate
storage root can be selected with `DREAMVLA62_STORE` only after the operator has
reviewed and authorized that location; the normal environment entry point remains
`~/kevin/envs/dreamcontrol_62`. The installer does not move or delete existing data.

The old hand-retargeting dependency required NumPy <2; the maintained 0.5.0
release supports NumPy 2. The old `flatdict` dependency failed metadata building
with modern setuptools. Its sole runtime use was replaced with the project's
`flatten_dict`; tests check every authored rendering-preset key and retain scalar
types and array-valued settings. Retargeting runtime tests remain open.

Official 6.2 wheel source was inspected without installation using bounded HTTP
range reads (receipts under workspace `out/isaac62_migration/sdk_source`). The
public tensor classes now live in `omni.physics.tensors.api`. The SDK still
maintains a Torch simulation view separately from its new Warp-only accessor.
Do not mechanically replace `get_physics_sim_view()` with
`get_physics_simulation_view()`: our asset/controller interface must remain Torch.
Kinematic reset now calls the public view method instead of its private backend.

An eager annotation for the removed `SoftBodyView` type prevented importing the
asset package, including rigid tasks. Postponed annotation evaluation fixes that
import dependency. It does **not** port legacy deformable simulation: the old
nodal-state APIs were removed and remain an explicit compatibility gap requiring
separate treatment if used. No deformable runtime capability is claimed.

`probe_task.py` is the next gate after the foundation probe. Run it from the
project's `Training` directory with an explicitly reviewed GPU, task and reference
pool, for example:

```bash
~/kevin/envs/dreamcontrol_62/bin/python ../cluster/isaac62/probe_task.py \
  --task Isaac-Motion-Tracking-Pick-Cam-BinaryFingers-v0 \
  --ref-motions-path /path/to/reviewed/reference/pool --device cuda:0 \
  --output ~/kevin/eval_videos/isaac62_pick_smoke
```

This bounded gate checks actual project-task initialization, reset and stepping,
finite observations/rewards/states, normalized root quaternions, Torch tensors,
PhysX selection, and cameras that do not advance physics. It saves the resolved
configuration, GPU process inventory, state traces, third-person and available
sensor PNGs, plus an MP4 checked with ffprobe for frame count and resolution.
It uses zero actions; learned-policy, optimizer, collection and task-performance
gates still require their own tests and paired renders for user review. Neither
this real-task gate nor the foundation gate has run yet.

## GR00T environment boundary

The existing GR00T fork pins Torch 2.7.1, NumPy 1.26.4 and compiled extensions.
The new `serve_gr00t_policy.py` entry point is intended to run that unchanged
policy in its existing model environment, on the same host as the new simulator.
The simulator-side client does not import GR00T or load its compiled extensions.
No model architecture, weights, processor, attention backend or fine-tuning recipe
is changed by this service path.

Collection, evaluation and clean playback accept `--gr00t-service <ready.json>`
alongside the existing checkpoint argument. The client verifies the live session,
embodiment, resolved checkpoint directory and configuration digest. Requests use
loopback-only ZeroMQ and MessagePack/NumPy arrays; a token remains in an owner-only
descriptor file. The service owner must shut it down after its consumers finish.
All service and simulator processes count toward the normal account resource caps.

Each call explicitly transfers Python, NumPy, Torch CPU and the selected CUDA
generator state in both directions. Optional backbone diagnostics preserve the
existing pooling and float16 representation. Failed or timed-out requests raise
errors and are not automatically retried. CPU process-to-process tests cover the
wire contract; actual-model parity across the two Torch versions and GPU runtime
tests remain required before enabling this path in a study. Do not describe a
transport fixture test as learned-policy validation.

During integration, a duplicated collector setup block was found in the merge.
It loaded two GR00T models and replaced the first DAgger driver. The second block
was removed; a regression test executes the actual setup and checks a single
model/driver, reference ordering/bounds and the expert-only path. The test fails
on `67a8510` because that version loads twice. This fix does not change the
accepted takeover, labeling or decoder recipe.

## Dataset preservation checks

The merged LeRobot utility now rejects duplicate/noninteger source episode IDs,
inconsistent catalog totals, missing/duplicate episode statistics and mismatched
modality metadata before starting an output. Valid source ordering and explicit
subset ordering remain unchanged. Required parquet index columns are checked
explicitly before rewriting indices. These validations do not change labels,
filtering or normalization rules.

`test_merge_lerobot_catalog.py` covers metadata failure cases without the converter
environment. `test_merge_lerobot_roundtrip.py` uses the baseline converter's real
datasets/pyarrow/LeRobot libraries to compare contiguous shard merging and reordered
episode subsets against a directly constructed dataset, including all numeric
columns, embedded HF schema, per-episode statistics and video bytes. Its video
fixtures are opaque bytes, so this test does not establish video codec validity.
Cached shard-level global/relative quantiles must remain absent from merged output.

On 2026-11-11 all nine catalog and real parquet round-trip tests passed on
Bluesclues in the existing `.venv_data_collection`, with CUDA disabled and one
logical CPU. Numeric columns, HF schema metadata, per-episode statistics and
opaque video bytes matched for both shard concatenation and reordered subsets.
Reports were copied by native SCP to workspace
`out/isaac62_migration/blue_data_contracts_f1d9e66.json` and its `.log` companion.

Reviewing the actual 6.2 physics wheel also identified an error in the prepared
foundation probe: the SDK explicitly rejects NumPy tensor views for a GPU physics
pipeline. The probe now uses the same Torch/CUDA tensor frontend as project assets
for state reads and resets, converting to NumPy only when saving CPU artifacts.
This is source-verified; the foundation gate still needs an actual GPU run.

## Target frontend CPU checks

`setup_cpu_contracts.sh` prepares a separate, small Python 3.12/Torch 2.11 CPU
environment under `~/kevin/envs/dreamcontrol_62_cpu_contracts`. It has no Isaac or
CUDA packages and cannot validate GPU physics/rendering. The shared NumPy,
TensorDict, RSL-RL and serialization versions match the target runtime. The
CPU-only dependency lock is separate so it cannot replace the CUDA build in the
real simulator environment. Existing environments are not modified.

The service tests can use `DREAMVLA_TEST_SERVER_PYTHON` to run their fixture server
under the legacy interpreter while the client runs in the new frontend. These
remain transport fixtures, not real checkpoint inference. Readiness publication
is now atomic and exclusive: a descriptor is complete when it becomes visible,
and an existing descriptor is never overwritten.

On 2026-11-11, the isolated Biped CPU runtime passed 73 contracts on Python
3.12.13, Torch 2.11.0+cpu and NumPy 2.3.1. Ten additional service contracts passed
with the client in this runtime and the fixture server in the existing GR00T
Python 3.10.20/Torch 2.7.1 environment. Reports are in workspace
`out/isaac62_migration/target_4cd4cb8.json` and
`cross_interpreter_4cd4cb8.json`. These are CPU and fixture-service checks only.

`cluster/isaac62/probe_residual_checkpoint.py` also strictly loaded three existing
trained checkpoints into the merged `AdapterActorCritic`, using explicit ELU
activation and checkpoint-derived MLP dimensions, in both the legacy and target
CPU runtimes. For eight identical synthetic observations, the largest absolute
actor-output differences were 4.77e-7 for T20_max_v3/model_4500, 2.38e-7 for
T20_max/model_4999 and 4.77e-7 for af60v7/model_4999. Critic differences were at
most 1.91e-6. Inputs were identical and all outputs finite; the original weights
were not modified. The checkpoint hashes, per-runtime package versions and arrays
are in workspace `out/isaac62_migration/isaac62_residual_parity_9994e4b/`.
This verifies trained MLP weight compatibility for the stated inputs/activation;
it does not establish simulation behavior, CUDA parity, SONIC decoder compatibility
or task success. GPU physics, policy-in-simulation and rendering gates remain open.

The native SONIC loader was then exercised against the existing 1.1 GB
`model_step_100000.pt` checkpoint with both its `g1` and `teleop` encoder heads and
the shared decoder. The lean CPU profile initially failed because serialized
trainer metadata requires Transformers and Accelerate. Transformers was already
in the intended simulator profile; Accelerate was missing and is now explicitly
declared in both profiles. All other simulator lock versions were retained.
No checkpoint conversion, loader workaround or control convention change was used.
On four fixed synthetic inputs per module, both encoder outputs and the decoder
outputs matched exactly between the legacy and target CPU runtimes. Reports,
hashes and arrays are in workspace
`out/isaac62_migration/isaac62_sonic_parity_1f86c86/`; earlier dependency-failure
artifacts remain alongside them. This does not establish CUDA, FSQ/control-loop,
physics or rendering equivalence.
After the dependency update, the same 73 CPU contracts and ten cross-interpreter
service contracts passed again at `1f86c86`; the corresponding report names are
`target_1f86c86.json` and `cross_interpreter_1f86c86.json` in the workspace evidence
directory. No GPU was allocated by these checks.

The real GR00T CUDA/service gate subsequently passed on Biped GPU 0 using the
existing Python 3.10/Torch 2.7.1 model environment and the historical walking R2
`checkpoint-92967`. Direct inference and service inference matched exactly for
all 40 predicted motion tokens and both hand trajectories on one fixed synthetic
observation. Backbone features and Python/NumPy/Torch CPU/CUDA RNG advancement
also matched exactly. Peak model allocation was 6,352,286,208 bytes. Both owned
CUDA contexts were released afterward. Evidence copied by native SCP is in
workspace `out/isaac62_migration/isaac62_real_gr00t_4cda30d_online/`.
The first attempt's forced Hugging Face offline mode prevented tokenizer metadata
lookup; the successful retry used normal access without environment/package/model
changes. This validates a real checkpoint through the service, but both CUDA
processes used the existing model runtime: new-runtime CUDA/Isaac integration and
actual task behavior remain unverified.

A further NumPy 2 source audit found the star-terrain generator using removed
`np.math` aliases. It now calls the same standard-library `math` functions
directly. The actual mesh functions, with target NumPy/SciPy/Trimesh versions,
passed finite-geometry, origin and configured-bound checks for 4, 5 and 7 bars.
The complete target CPU contract count is now 74; all passed, along with ten
cross-interpreter service checks, at `972b8c7`. Local reports are
`out/isaac62_migration/target_972b8c7.json` and
`cross_interpreter_972b8c7.json`. Mesh construction does not substitute for the
outstanding terrain-in-simulation and rendering gates.

The real saved-data gate also passed through the existing GR00T loader and its
default TorchCodec backend. One complete episode from each of `latband60_v1`,
`pick50_dagger57_20261013_r2/lerobot/ds`, and
`walk200_v3_late1s_20261024_r3/lerobot/ds` supplied 499, 499 and 607 frames,
respectively. All decoded episode lengths matched parquet and catalog lengths;
every loaded state/action column, including the 64-dimensional motion token and
both finger targets, matched its original parquet slice exactly. Nine sampled
nonblack, nonconstant RGB frames and the report were pulled into workspace
`out/isaac62_migration/isaac62_real_dataset_17c0386_nested/`.
The first attempt correctly reported two parent directories without `meta/info.json`;
the rerun used their verified nested LeRobot roots. The check used one CPU core,
CUDA disabled, and the unchanged existing Torch 2.7.1/NumPy 1.26.4 model environment.
It validates these saved episodes and the existing codec path, not all datasets,
new-runtime normalization, Isaac 6.2 camera rendering or task behavior.

`cluster/isaac62/probe_train_resume.py` prepares the simulator-connected residual
training gate using the actual `train_sonic_adapter.py` entry point: two updates,
save, then one resumed update. Its four environments, 16-step rollouts and one
PPO epoch/minibatch are smoke settings, not a study recipe. It checks checkpoint
iteration counts, continued optimizer steps, finite model state and changed
parameters, retaining both trainer logs and checkpoints. It requires explicit
reference, SONIC checkpoint and asset-directory paths and a manually reviewed
GPU allocation. Its checkpoint schema was checked against installed RSL-RL
3.0.1 source; the simulator-connected gate subsequently passed at `556890e` (see below).

Serialized policy checks found a pre-existing recurrent export bug in both the
legacy and target Torch runtimes: assigning an undecorated bound reset method
removed `reset()` from the saved TorchScript interface. The exporter now keeps
the decorated reset method on the class and clears available hidden/cell buffers.
Four output/state tests pass in both runtimes at `8e21349`, covering feedforward,
normalized feedforward, GRU and LSTM policies. Recurrent sequence outputs match
the native modules, and resetting restores the first action. Evidence is in
workspace `out/isaac62_migration/isaac62_export_8e21349.json`; failing controls are
retained at `policy_export_b4bfd35.json` and `policy_export_legacy_b4bfd35.json`.
These use small fixtures, not trained-policy or simulator behavior. ONNX export
subsequently passed four real export/checker/ONNX Runtime cases in the full target
environment, alongside the four TorchScript cases, within the 82 CPU contracts
at `f54c1a0`.


## November 12 simulator progress

The bounded PhysX/RTX foundation probe passed at `e42b3db` in the isolated
Python 3.12 / Torch 2.11 / Isaac 6.2 environment on Biped. It advanced exactly
256 physics steps, captured 64 nonblack/nonconstant frames, checked GPU dynamics,
free-fall velocity and authored/reset XYZW quaternion interpretation. Independent
projection of the physics position matched the rendered cube centroid within the
predefined four-pixel endpoint tolerance: 3.753 pixels initially and 0.525 pixels
finally. This proves the stated endpoint gate, not perfect per-frame alignment.
Evidence is in workspace `out/isaac62_migration/isaac62_foundation_e42b3db/`.

A prior `d53d3a7` pixel-change check was insufficient: rotation changed the image
while translation remained stale. Stronger projection checks correctly failed
subsequent probes. The successful scheduling follows SDK Core rendering: keep the
timeline playing and disable automatic physics with `/app/player/playSimulations`
only during application updates. Pausing the timeline displayed authored poses
instead. Fabric scene delegation is enabled; an attempted extra GPU transform-read
setting did not fix this problem and was removed from the raw probe. All failed
artifacts remain available; none establish project-policy behavior.

Actual kitchen/G1 pick-task probes exposed and fixed two additional migration
issues: absent headless viewports, and a removed deformable Python schema class
in generic physics-material binding. Real USD material-relationship checks pass
for scenes, rigid colliders and the new deformable schema. This does not port the
remaining deformable tensor/stress APIs. The real task probes subsequently passed at `556890e`: pick-camera with static
references, pick-camera with stitched walking references, and motion-only. Each
completed reset, 32 zero-action physics steps, finite-state checks, video capture,
and final reset. Viewer, scene and ego camera paths produced nonblack/nonconstant
images. These are technical smoke checks; robot behavior requires user review.

The actual residual trainer also passed two PPO updates, checkpoint save and one
resumed update. All 15 optimizer states advanced from step 2 to 3 and 15 model
tensors changed with finite values. Actual nominal and saved residual evaluators
completed episodes and produced finite world/root-relative body MPJPE. Recorded
checkpoint recipes must be preserved: the saved static/walking experts expect
1015 actor inputs, while the non-reworked task exposes 988. An intentional wrong-
recipe test at `1b7cc5d` verifies nonzero exit and reservation release after this
mismatch. The evaluator now closes its environment on exceptions before Kit
teardown and prints the failure before SDK shutdown.

The real GR00T checkpoint `walk200_v3_dagger57_20261013_r2/checkpoint-92967`
completed a 32-step closed loop at `bed9ff4`: model inference remains in the
unchanged Python 3.10 / Torch 2.7.1 environment, while the simulator client uses
Python 3.12 / Torch 2.11 / Isaac 6.2. The real client package is installed normally
from the user fork (`81c7a50`) with dependency resolution and a constraint lock;
all 267 previously installed package versions were preserved. Body trajectories
and predicted hand targets were finite. This is not a scientific success-rate
evaluation. The real policy video entrypoint also passed at `1b7cc5d`: 32 frames each
for third-person (2560 x 1920) and ego (1280 x 960), both at 50 fps.
Basic image statistics pass; user appearance review remains pending.

Evidence is local under `out/isaac62_migration/`: task folders
`isaac62_{pick,walk,motiononly}_task_556890e_isolated`,
`isaac62_residual_training_556890e`, `isaac62_expert_eval_8bd0631`, and
`isaac62_vla_loop_bed9ff4` and `isaac62_vla_loop_1b7cc5d`.

Drawer's first step exposed an inherited 27-element reward mask applied to the
29-joint layout; button uses the same pattern. A name-based compatibility patch
is prepared but awaits user review because it affects reward evaluation. The
full GR00T Adam update/save/resume gate still needs more GPU memory than Biped's
32 GB. Other remaining gates include real collection/intervention labels and
conversion, paired reference replay and user appearance review, and removed
generic deformable tensor/stress APIs. The migration is not complete.

The user approved only the bounded migration probes' GPU-1 graphics context of
at most 6 MiB; simulation, rendering and CUDA remain on GPU 0, with the normal
20% aggregate account CPU limit. This exception is recorded in EMBER/AGENTS.md.
It does not authorize a second computation GPU. Probes release their reservations
on terminal success or failure; other researchers' contexts remain untouched.


## November 12 collector validation

Actual saved expert collection at `fe2460f` completed one attempted episode
(605 steps), reported one rejection and zero script errors, and released its
reservation. It retained no episode, so this case does not prove successful
expert-demo writing or estimate the expert's success rate.

The real GR00T DAgger label-writing probe at `b7d6bce` then retained two isolated
64-frame HDF5 episodes: one entirely student-driven and one entirely expert-driven.
Both contained finite 64-dimensional FSQ-grid expert token labels, 65-dimensional
actions, seven finger targets and matching driver masks. Neither segment contained
a close command. This gate proves the stated schema/driver checks, not closed-grasp
coverage, failure-conditioned rescue, or scientific policy performance. The eight
small finger/action template files were extracted without modification from archived
base demonstrations; they are explicitly marked incomplete fixtures, not training
examples. Source hashes and column hashes accompany them.

Strict snapshot and repeated-reset diagnostics remain failed. At 16 steps, two
seeded resets diverged by 0.01037 radians in joint position. Snapshot restoration
initially diverged by 0.2521 radians / 5.104 rad/s. Preserving PD position, velocity
and effort targets reduced those errors; preserving the omitted decoder
`_prev_body_29` state reduced joint-position error further to 0.002984 radians,
but joint-velocity error remained 1.732 rad/s. The 1e-6 strict gate is still failed;
no tolerance was relaxed. Historical Isaac 5.1 logs also show failures, but use
different checkpoints/settings and are not a matched baseline.

Importantly, production failure-conditioned rescue reruns the student prefix;
it does not call these snapshot functions. The snapshot changes affect the diagnostic
path. They do not establish that full rescue collection passes migration validation.

A malformed smoke-test motion-ID argument exposed a separate collector teardown
gap before the rollout loop. Environment and SDK shutdown now use an exit stack
covering setup failures too; explicit motion-ID manifests replace the malformed
argument. The corrected two-phase collector exits normally and releases resources.
Local evidence: `out/isaac62_migration/isaac62_collection_{fe2460f,c6c401d,f7c056d}`
and `isaac62_dagger_labels_b7d6bce`. New-recording conversion is a separate gate:
the simulator profile lacks the exporter's `datasets` dependency, and the existing
GR00T model environment lacks `h5py`. Both attempts failed before conversion.
A separate CPU-only exporter profile is prepared in `requirements.conversion.txt`
and `setup_conversion.sh`, pending the user's environment approval. It honors
the pinned LeRobot commit's Torch <2.7 / Gymnasium 0.29.1 requirements, using
Torch 2.6 CPU without modifying existing environments. The original LeRobot
metadata is at https://github.com/huggingface/lerobot/blob/a445d9c9da6bea99a8972daa4fe1fdd053d711d2/pyproject.toml.


The actual reference-PD playback and three overlay paths exposed a silent video
failure at `b918c78`: OpenCV reported 32 writes without creating an MP4. The
entrypoint now shares the FFmpeg RGB writer with GR00T playback and rejects empty
output. At `8d06529`, the same reference-PD gate passes with a real 32-frame,
50-fps, 2560x1920 video and a nonblack extracted frame. Both playback classes
retain their original public constructor/write/close methods; the static union
audit at `1431995` again has zero findings against both recorded parents.
GR00T video regression at `1431995` also passed: actual 32-step evaluation
and playback, with 32-frame third-person and ego videos verified at 50 fps. All
reservations were released and no account GPU contexts remained. Evidence is
local in `isaac62_reference_pd_8d06529` and `isaac62_vla_loop_1431995`. These
technical checks do not replace user review or the remaining matched-baseline
and full-reference replay checks.

The recorded-action replay gate at `4b57c55` now passes for an actual 64-frame
expert-written HDF5. Every requested 65-dimensional command matches the recording
exactly and all saved trace values are finite. This deliberately does not claim
physical replay equality: the accepted previous VLA decoder and the expert's
residual token path have different FSQ transforms. No decoder change was made.

At `c037633`, both full 252-frame stitch outputs were replayed kinematically in
Isaac 6.2 through the production reference playback entrypoint. Each clip contains
640 frames at 50 fps and 2560x1920, covering the accepted post-warmup start through
the reference endpoint (clock 0.54 to 13.32 seconds; duration 12.6). Applied poses
match their corresponding reference within 1e-6. Between the paired generated
references, root pose and clocks are exact; joint differences remain at most
7.54e-5 radians. This does not overturn the failed strict cross-runtime IK gate.
Both original clips and traces are local in
`out/isaac62_migration/isaac62_stitch_replay_{legacy,target}_c037633/` and indexed
in `out/reports/dashboard.html`, without recompression. User appearance review
remains required; these are reference clips, not learned-policy rollouts.

A matched static af60v7 evaluation harness at `759101c` ran the same public
evaluator, checkpoint hash, reference, seed and identical effective environment
configuration in Isaac 5.1 and 6.2. Each completed two episodes and released its
reservation. World/root-relative MPJPE was 4.3820/2.7107 cm in 5.1 (350 frames)
and 4.1842/2.7344 cm in 6.2 (428 frames). These bounded cases verify real evaluator
execution and configuration consistency, not scientific success-rate equivalence.
Legacy task/wrapper sources match the recorded g1 parent for this static case;
the older g1 environment does not provide a matched newer walking-task baseline.
Receipts, effective configurations, episode arrays and logs are local under
`isaac62_static_baseline_{legacy,target}_759101c`.

The production two-pass failure-conditioned rescue path passes its bounded
mechanics/label gate at `efd9222`: the initial student episode fails the numerical
four-rule criterion, then an independent sibling hands over at step 191 sampled
uniformly from [0,258). The retained recording has 605 frames, 414 expert-driven
frames and 374 closed-hand expert labels. The saved driver mask matches the
takeover exactly; open labels have zero finger targets, and closed labels use the
fixed demonstration pose consistently. The rescue episode itself also fails the
four-rule task criterion; retaining corrective failures is the specified DAgger
behavior, so this is not a policy success or performance claim. Its reservation
was released and all account GPU contexts were confirmed gone. Local receipts,
logs and the independent saved-label audit are in
`out/isaac62_migration/isaac62_dagger_rescue_broad_efd9222/`.

The first rescue probe correctly failed before collection because no calibrated
box was supplied. The next 200-step probe correctly failed coverage because the
fixture's 20-Hz reference grasp occurs after 258 steps at the 50-Hz simulation
rate. The corrected probe uses the existing expert calibration and a 640-step
cap, preserving production collection semantics rather than fabricating a grasp
or changing its timing. The separate one-second late-window case at `1f1a49c`
also passes: takeover is 245 within [208,258), exactly the final 50 simulation
steps before the same reference grasp. It retains 605 frames, 360 expert-driven
frames and 451 closed-hand labels. Independent reads of both HDF5s verify the
complete driver masks and open/closed finger targets; 95 late-case close labels
occur on student-driven states, confirming that expert targets are recorded
before takeover as well. No claim of improved task performance follows from
these checks. Both probes are terminal and their GPU contexts are released.
Local late-case evidence is in `isaac62_dagger_rescue_late_1f1a49c/`.

The proposed separate CPU exporter profile now resolves successfully without
creating an environment or installing packages. A global PyTorch extra index
initially shadowed an unrelated cmake dependency under uv's safe index policy.
Explicit official CPython3.12/Linux CPU-wheel URLs for Torch/Torchvision avoid
that problem while keeping normal dependency checks. The candidate lock and
resolver receipt are local in `out/isaac62_migration/conversion_profile_candidate/`.
Environment approval, installation/import checks and actual recording conversion
remain pending. Dependency resolution alone is not a working-converter claim.


### Volume-deformable port and actual runtime checks (2026-11-12)

The earlier unported-deformable statements above describe the historical state.
Commits `d5ade0f` through `5ddbe29` now implement the current volume-body schema,
cooked simulation/collision hierarchy, material binding, mass/density ownership,
nodal state writes and the removed diagnostic interface. Existing rigid-body
and Isaac 5.1 paths remain selected by the installed schema API. Removed
non-neutral material `damping_scale` is rejected explicitly rather than ignored.

The actual procedural asset check at `5ddbe29` passes 20 physics steps under
Isaac 6.2: a 0.8-kg body with material density 1300 falls 0.04061651 m, accepts
state restoration and nodal kinematic targets, and reports six finite diagnostic
arrays. Both simulation and collision quaternion norms pass (maximum error
4.18e-7). It uses 125/243 simulation/collision nodes and 384/480 tetrahedra.
The wrapper exits zero, simulation callbacks are cleared before SDK shutdown,
and process and GPU-context release were independently checked. An earlier
`91d24a9` attempt finished its numerical checks but hung during shutdown; that
failure is preserved and was cleaned up only after verifying process ownership.

The same procedural asset under the existing Isaac 5.1 environment also exits
zero and falls 0.04061649 m. Its cooked topology is different (64/146 nodes,
135/270 tetrahedra), so these are not elementwise-comparable meshes. It reports
zero collision quaternions, with norm error 1, despite no PhysX cooking error.
This confirms that the invalid legacy collision rotations seen in the earlier
single-tetrahedron fixture are not solely a fixture-cooking artifact. The new
adapter returns proper polar rotations; it does not fabricate zero rotations to
match invalid legacy diagnostics. This is an explicit diagnostic difference,
not a full parity claim.

For the explicit single-tetrahedron comparison at `254bf72`, simulation and
collision gradients pass the preselected 1e-5 tolerance, simulation quaternion
error is 2.69e-7, and simulation stress differs by at most 0.360 Pa (1-Pa gate).
Collision quaternion and stress comparisons remain failed; no tolerance was
relaxed. The legacy fixture also emits a cooking warning, unlike the normally
cooked asset above. The diagnostic matrix layout preserves the original
column-major buffers and the SDK's corotated-stress convention.

Local evidence is in `out/isaac62_migration/deformable_tensor_comparison.json`,
`isaac62_deformable_asset_5ddbe29/`, and
`isaac62_deformable_asset_legacy_retry_5ddbe29/`. Generic coverage remains
incomplete: implicit-material stress, heterogeneous padded topology, migration
of preexisting legacy USD assets, and non-neutral damping require further work.
These deformable diagnostics are not used by the tested rigid robot tasks.


A targeted rigid-robot regression at `5ddbe29` verifies the common spawner changes
did not break the pick-camera environment: initial reset, 32 finite steps,
third-person and camera PNGs, an independently counted 32-frame 1280x720/50fps
video, and a final reset pass. The process and GPU contexts were released.
This probe applies zero actions and explicitly does not validate a learned policy
or assess task success. Local results:
`out/isaac62_migration/isaac62_pick_task_5ddbe29_deformable_regression/`.


### Matched reset/snapshot investigation (2026-11-12)

At `0c3726a`, the same current collector, af60v7 model, static reference, seed17
and HS settings ran the strict reset and restore tests under both SDKs. Both
fail: the maximum repeated-reset joint deviation is 0.02175 rad in Isaac5.1 and
0.02171 rad in Isaac6.2. Snapshot replay joint-velocity maxima are 0.1461 and
0.09221 rad/s respectively, starting at replay frame0. These short cases do not
establish statistical equivalence, but contradict an Isaac6.2-only explanation.
No tolerance was relaxed; exact stitching remains unvalidated. The production
failure-conditioned rescue path uses independent rerolls, not these snapshots.
Logs and comparison: `out/isaac62_migration/matched_reset_snapshot_comparison_0c3726a.json`.

The investigation found a separate real restoration bug: CircularBuffer.buffer
returns a chronological copy, so writing into that property did not restore
observation history. `319c59a` saves/restores backing storage, cursor and
per-environment fill counts. Tests use the real circular buffer and verify
continued appends after an uninitialized, partial and wrapped history, including
one reset batch member. They pass. The collector setup contract was updated to
its cleanup-owned entrypoint at `8cc8354`; all68 current CPU contracts pass.
The ring fix is not included in the matched `0c3726a` simulation results and is
not claimed to resolve the remaining physical replay divergence. Collector SDK
shutdown now feature-detects exit_code, which Isaac5.1 does not accept.


### Differently cooked bodies in one view (2026-11-12)

The first actual mixed-mesh probe at `70f1023` exposed an adapter bug: it required
every body's rest-node array to fill the batch's maximum node count. The failure
was preserved and its process/GPU resources released. `d3796af` now retains each
cooked TetMesh's element count, pads rest-node storage, masks only nonexistent
elements, and still rejects degenerate real tetrahedra. Simulation diagnostic
padding is masked too. The per-body cooked counts are available on the adapter.
Zero padding is not reported as a physical rotation or material response.

The repaired two-body probe passes 20 real physics steps, state restoration,
kinematic writes, six finite diagnostic arrays and unit rotations for valid
elements. The different bodies have [384,216] simulation and [480,284] collision
elements. A stronger probe at `e43c2ff` applies a known10% x-stretch to the second
body, with a distinct75000-Pa/0.27 material, while the first remains unstretched.
Collision gradient error against the analytical affine response is9.64e-6
(preset1e-4 gate); stress error is0.562Pa (preset10Pa gate). Those checks distinguish
actual per-body values from finite padding or copied batch-member data.
The earlier single-tetrahedron legacy collision-parity failures remain failed;
this separate analytical gate does not change their tolerances or conclusion.
Local arrays/scenes/receipts: `out/isaac62_migration/isaac62_deformable_batch_e43c2ff/`.
Generic deformable coverage still does not establish implicit-material stress,
preexisting legacy USD migration or non-neutral damping support.


### Ordinary expert write and raw walking conversion (2026-11-12)

The ordinary collector gate at `8119201` uses the unchanged T20_max_v3 model4500,
accepted HS recipe, skip-start0 and1200-step horizon. Three preset references
(pick0,11,11000) were selected before outcomes. It stops after the first accepted
episode: preset motion0 passes the existing physical-hold filter and produces
one validated577-frame HDF5 with65-D actions and recorded observation/teleop
columns. It tries1/3 references, with zero rejected/error counts. This verifies
normal passing-filter/write mechanics, not a success-rate estimate. No filter
was weakened. The process, reservation and GPU contexts are released. Local
receipt/log: `out/isaac62_migration/isaac62_expert_write_8119201/`; its~630MB HDF5
remains on Biped at the receipt's relative path. A follow-up harness-only change
`9a99376` scales timeout by the preset attempt bound; it did not alter that run.

Raw walking conversion now has actual coverage at `45ebd17`, rather than reusing
an already converted walking prefix. The original pick0 qpos NPZ (195frames,
43columns, SHA2566b7d4be1a488c68ae5ea2dec2a59454d9acccebc25b7eaa62cb57190128c63b5)
was copied from Bluesclues with its original source manifest and verified
against that manifest. AdapterB runs the accepted reconstruction environment in
both existing5.1 and target6.2 Python profiles, CPU-only on one core. Both output
195frames,29joints and grasp frame52. Joints, root positions, object poses and
grasp position are exactly equal; global pose matrices differ by at most5.97e-8
under the preset1e-6 gate. The old JAX CUDA plugin logs CUDA_NO_DEVICE with GPUs
hidden, but CPU conversion completes; that diagnostic is preserved, not hidden.
No packages were changed. Local outputs/recipes/source provenance/comparison:
`out/isaac62_migration/isaac62_raw_walking_45ebd17/` and `raw_walking_fixture/`.
This covers raw-to-reference walking conversion, not upstream holosoma motion
inference or arm refinement (the accepted walking-prefix recipe disables it).


### Saved legacy deformable USD conversion investigation (2026-11-12)

The SDK ships `omni.physxassetvalidator.scripts.deformableSchemaChecker`, including
a volume conversion path. Probes `060189b` and `3e26b4c` attempt to use it on an
anonymous flattened copy of the actual5.1 cooked cube fixture. Neither converts
a body: the saved `PhysxDeformableBodyAPI` and `PhysxDeformableBodyMaterialAPI`
tokens remain in authored metadata, but6.2's GetAppliedSchemas omits them, both
before and after flattening. The supplied SDK checker uses GetAppliedSchemas
for discovery too. Thus the public conversion path is not currently validated;
no converted asset or successful6.2 file-load claim follows from these runs.
The source file was not edited; no SDK/environment source was patched. Both
failed processes terminated and released their GPU resources.

The probe also rejects two known silent-loss cases before conversion:
non-neutral legacy dampingScale, and body-only density without explicit mass
(the SDK converter deliberately drops that density). Existing procedural6.2
assets remain independently tested, including heterogeneous meshes and analytical
material response. A separate metadata-driven application conversion path would
need implementation and a real file-load/physics check to close this gap.
Local evidence: `out/isaac62_migration/isaac62_legacy_deformable_usd_3e26b4c/`,
including `schema_discovery.json`, the original failure receipt and sibling log.


### Metadata-driven saved-volume conversion and read-only diagnostics

`9315177` adds an application-level cooked-volume converter that discovers old
APIs in authored metadata instead of the SDK's filtered registered-schema list.
It operates only on an anonymous flattened copy, uses installed SDK geometry
helpers, maps explicit physical properties and legacy defaults, and leaves the
SDK/environment unchanged. The original SDK-discovery failures remain recorded.

The first converted asset exposed another adapter bug: constructing the SDK's
experimental DeformablePrim over a non-auto cooked Xform recreates its mesh
hierarchy, deleting the already registered simMesh and invalidating the tensor
view. `ebf49a5` avoids that constructor for these assets. Its diagnostics read the
existing solver meshes, reconstruct gradients/polar rotations and preserve the
same corotated-stress convention. Existing auto-generated assets keep their
previously tested diagnostic route. The saved cube then loads and simulates
successfully in6.2 without changing the source file.

At `33ad1ea`, the converted saved cube retains64simulation nodes/135elements and
138collision nodes/270elements from the file. It completes16freefall steps plus
four kinematic steps with a prescribed10% x-stretch. Both diagnostic paths pass
the preset1e-4 gradient and10Pa stress gates: simulation errors3.70e-6/0.239Pa,
collision errors2.80e-5/0.658Pa. All six diagnostic arrays are finite. The wrapper
exits zero and process/GPU release is independently confirmed. Source SHA256
7517235aacd9827389d58a3bb50bcc05f82b47fb12162523f8ffdc35d5d7ef69 is unchanged.
Local converted USD, arrays and receipts:
`out/isaac62_migration/isaac62_legacy_deformable_usd_33ad1ea/`.

This is bounded saved-volume coverage, not arbitrary USD equivalence. The
converter rejects non-neutral dampingScale, density-only mass configuration,
legacy surface/cloth/attachment types, and animated legacy solver arrays rather
than dropping them. Generic command-line conversion packaging and rejection-case
tests remain separate work; ordinary asset loading does not silently rewrite
legacy files. Implicit-material stress support also remains unverified.


### Reusable saved-volume converter and rejection checks

Commit `b7f5d4c` exposes the same tested conversion implementation through
`cluster/isaac62/convert_legacy_volume_usd.py SOURCE DESTINATION`. Run it in the
approved cluster Isaac 6.2 environment, with the usual account resource review
and GPU reservation. The destination must be a new file. Conversion is explicit;
ordinary asset loading never rewrites a legacy source automatically.

The integration probe at this commit exercises six loss-prevention cases on
anonymous stages: non-neutral dampingScale, density without explicit mass,
animated solver arrays, surface deformables, particle cloth and attachments.
All six reject without modifying the anonymous stage. The same shared converter
then passes the saved-cube 20-step physics and affine-diagnostic checks, with the
source SHA unchanged and clean exit. The standalone CLI at `c911bbc` also completes against the same saved source,
with a new converted file, unchanged source hash and exit zero. Its GPU reservation
is released, and an independent XML inspection finds no account GPU contexts.
Local converted USD, diagnostic arrays, receipt and terminal record:
`out/isaac62_migration/isaac62_legacy_deformable_usd_b7f5d4c/`.


The post-history-fix snapshot/reset retest at `b7f5d4c` retains both strict
failures: reset joint-position divergence is0.02171rad and snapshot replay
joint-velocity divergence is0.09221rad/s. Measured snapshot state restores
exactly before stepping. These match the earlier target results; correcting
the observation ring did not establish exact physical replay. Production DAgger
rescue retains its tested independent-reroll path. Failed receipts and logs are
preserved in `out/isaac62_migration/isaac62_static_reset_target_b7f5d4c/`.


### Direct expert recording and terminal-frame fix (2026-11-12)

The existing approved evaluation ablation (af60v7 actor output, usual FSQ snap,
no base-token addition) now uses the same DirectExpertEvalWrapper in the video
entry point. Actor observations, decoder and finger scalar remain unchanged.
The runtime probe accepts this path without starting a GR00T model service.

Actual run `0aeff68` exposed a video/trajectory mismatch: termination at step5
left four nonterminal trajectory frames but five video frames. The recorder
wrote after the environment had automatically reset, before checking dones.
`a5aa9cc` places video writes under the existing nonterminal guard. Its rerun
passes actual evaluator and recorder gates: four finite trace frames, four
third-person frames (2560x1920), four ego frames (1280x960), all at50fps with
nonblack frame samples. The actual scalar finger commands match saved actions.
The prior pred_meanq field is a placeholder in this mode and is explicitly
labelled as such, rather than counted as a predicted finger-angle check.

The ordinary saved GR00T92967 service/evaluator/recorder path is also rerun at
`a5aa9cc` and passes32frame traces and both camera videos. Both runs exit zero,
release reservations and leave no account-owned GPU contexts. These bounded
runs validate wiring, termination accounting and media, not task performance.
The capability-union audit at this commit has no findings against334g1 and
315locomotion project files; this remains a surface audit, not behavioral proof.

Local clips, traces, receipts and logs: `out/isaac62_migration/isaac62_direct_expert_a5aa9cc/`
and `out/isaac62_migration/isaac62_vla_loop_a5aa9cc/`. The failed first run is
preserved in `isaac62_direct_expert_0aeff68/`. The shared dashboard links original
videos without recompression. User appearance review remains pending.


### Runtime application label and preserved asset URLs (2026-11-12)

The shared legacy Kit experiences label the application5.1.0 even when the
installed runtime is6.2. `17fa94e` supplies the installed pip runtime's three-part
application version for default experiences on6.x. Legacy runtimes, custom
experiences and explicit Kit overrides retain their prior behavior. This fixes
the misleading settings/log namespace without changing the authored asset URLs.
Four focused launcher-argument contracts pass.

The actual pick task at this commit reports application_version6.2.0 and the
unchanged Isaac5.1 asset root. It completes32zero-action steps, final reset,
material-binding checks and three-camera image checks. Its video has32frames at
1280x720,50fps. Physicsdt0.005 anddecimation4 are unchanged. The wrapper exits
zero and releases its reservation; independent XML checks find no account GPU
contexts. Local evidence: `out/isaac62_migration/isaac62_pick_task_17fa94e_runtime_label/`
and `runtime_label_cpu_contracts.json`. This is a technical task/rendering gate,
not policy or task-success validation.

The latest Bluesclues capacity check still has all ten GPUs occupied by other
accounts, with at most20.5GiB free on any device. No optimizer test is launched
into that insufficient headroom. The Adroit access hold remains unchanged.


### Actual human-motion input export parity (2026-11-12)

`f5affef` exercises the unmodified Adapter A (`export_to_holosoma.py`) against
original OmniControl `Pick_sim/results.npy`, SHA256
696637ac3f7fa6f649479feb75a2b6b86af7fe28cc1c64de576e7ca84b5e8aaf.
The existing5.1 and target6.2 profiles each export motion0 into separate new
validation directories, CPU-only on one core. Both produce195frames,22joints,
finite joint/object/height arrays and identity object quaternions. All three
output arrays match exactly, under the preset1e-6 absolute gate. The original
results file hash remains unchanged. The fixture finishes with exit zero;
account CPU remains below0.01% at the surrounding samples and no GPU is used.

The first launcher attempt constructed the baseline interpreter under~/kevin
instead of~/miniconda3; it failed before either exporter ran. That log is kept.
The corrected launcher uses the existing approved profile; no packages or
source data are changed. Local receipts, actual commands/logs and both NPZs:
`out/isaac62_migration/isaac62_human_motion_export_f5affef_retry/`.
This adds actual input-export coverage before the already tested raw-walking
Adapter B. It does not establish full OmniControl diffusion sampling or holosoma
optimization. A later path audit finds the executable in holosoma's nested package;
its separate existing environment is required (see the review note below).

Implicit-material deformable stress remains unsupported: the installed SDK's
own diagnostic implementation substitutes zero modulus/Poisson values when no
material is bound, which does not establish the solver's actual default material.
The bridge therefore keeps its explicit error rather than inventing a constitutive
response. Explicit-material diagnostic paths remain independently validated.


### Remaining upstream profile boundary (2026-11-12)

A corrected path audit finds Biped's retargeter at
`~/kevin/holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py`.
The earlier top-level examples-path check was insufficient; the executable is
present. The installed6.2 profile lacks cvxpy, and the retargeting package pins
NumPy2.3.5 versus the simulator profile's2.3.1. Do not install conflicting pins
into the simulator. Both original hsretargeting interpreters exist; the generation
script prefers the one under~/kevin. Using it needs environment review under the
workspace agreement. No alternative environment was executed during this audit.

The shared holosoma checkout is atde6ea0135fed1c10d55c406688fb7a3ac2b5111e,
with a modifiedscripts/source_common.sh and an untracked retargeter backup.
Those are untouched. The prepared one-clip review plan uses a new Git export,
oneCPU,noGPU/no viewer and the unchanged default retarget recipe; it leaves
shared source/data and all existing packages intact. Plan:
`out/isaac62_migration/holosoma_existing_profile_review.json`.


The next real-recording conversion gate is prepared in
`cluster/isaac62/probe_recording_conversion.py`. It requires two distinct complete
HDF5s and the committed converter export in the approved CPU profile. It writes
fresh serial and two-shard conversions, invokes the production merger and exact
comparison, checks recorded token equality and episode/frame totals, and verifies
input hashes and absence of cached global normalization. It retains every phase
log/output on failure. Only syntax checks have run; this gate is not passed.
Official GR00T loader/video decoding and regeneration/comparison of whole-dataset
and relative-action normalization remain required after successful conversion.

### Approved conversion, remaining task families and original retarget (2026-11-12)

The user's subsequent approval resolved the conversion-profile, reward-mask and
existing hsretargeting-profile reviews above. The earlier pending statements are
historical; these three items have now been executed.

The approved27-to29 mapping retains each original named joint weight and assigns
zero to added waist roll/pitch joints. Five actual-function CPU tests cover this
mapping, unchanged legacy/current layouts, invalid masks, optional success counters
and optional button contact history. Runtime testing exposed missing bookkeeping
attributes; guards preserve existing counter/buffer updates when present and allow
ordinary task resets when absent. Drawer at3a188dd and button atc350d1c each pass
reset,32zero-action steps, rendering, and final reset under6.2. Each original
1280x720 video contains32frames at50fps. These are mechanical gates, not learned
policy success measurements. Both workers release their reservations and contexts.

The isolated CPU conversion environment is installed; simulator/model environments
are unchanged. Actual complete expert and DAgger recordings contain577 and605frames.
Serial conversion equals two separately converted shards after the production
merger. Source hashes and recorded64-dimensional action tokens are unchanged.
GR00T regenerates whole-dataset stats.json and relative_stats.json independently
for both datasets; every value, including percentile statistics, matches exactly.
The official saved92967 modality configuration and TorchCodec loader decode both
episodes in both datasets. All state/action columns match parquet exactly; sampled
images have valid dimensions and nonconstant content. No statistics are pooled
from shard percentiles. Evidence is local under
`out/isaac62_migration/isaac62_recording_conversion_3a188dd/` and
`isaac62_recording_roundtrip_42009ca/`.

The unmodified original holosoma retargeter fromde6ea01 runs CPU-only in the approved
existing hsretargeting environment, from an isolated Git export. Its default
one-clip SQP/foot-sticking recipe produces195frames and43qpos columns, with finite
values, unit root quaternions and unchanged input hash. The newly generated raw
clip then passes AdapterB in both legacy/target profiles: joints, root positions,
object and grasp arrays are identical; rotation-matrix maximum error is5.96e-8.
Shared holosoma source and installed packages remain untouched. Evidence:
`isaac62_holosoma_original_3a188dd/` and
`isaac62_raw_walking_new_retarget_3a188dd/` under the same local output root.

All these probes are terminal. Final full-account audit finds no account-owned
GPU contexts and negligible CPU use. Artifacts are pulled locally and the dashboard
includes the new drawer/button clips without recompression. Remaining gaps include
full GR00T Adam/save/resume on a sufficiently large free GPU, original diffusion
sampling and user visual review. The previously documented strict stitch-IK,
snapshot/reset, implicit-material stress and damping limitations remain unresolved;
these new passes do not waive those gates. The original walking research chain
remains stopped and was not resumed by this migration validation.
