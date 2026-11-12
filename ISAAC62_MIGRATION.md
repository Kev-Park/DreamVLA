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
