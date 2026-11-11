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
`out/isaac62_migration/`. The 6.2 runtime is not yet validated. Do not resume the
existing full DAgger study or mark this migration complete from static checks.

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
