# g1 / locomotion consolidation and Isaac Sim 6.2

The target is one maintained implementation containing both branches' capabilities,
running on Isaac Sim 6.2 with preserved policy inputs, outputs, physics recipes,
data formats, collection semantics, metrics, and rendering. A successful import
or a file-surface audit is not completion.

## Fresh rest-pose repair live validation (2026-11-16)

With the user's narrow GPU1 graphics exception, fresh conversion at `134c4f0`
passes the 50-inertial-body/43-joint physical-property audit. All 54 authored
rest-world matrices match the original within 5.551115123125783e-17; fixed-joint
graph, zero-force drives and joint velocity limits agree. The paired real PhysX
initial and canonical nonzero poses both pass the existing 1e-6 gate. Effective
masses and COM positions are exact, maximum inertia error is 7.45e-9, and named
43-joint/PPO/hand interfaces agree. The aggregate `preservation_pass` is still
false: raw importer joint order differs, although named mappings pass. This
receipt must not be represented as an unconditional preservation pass.

Standing drawer, squatting drawer and standing button each complete the fresh
asset's two-environment, 32-step reset/camera/contact/render integration fixture.
All three exit zero. The 39 output files are SHA-256 verified locally under
`out/isaac62_migration/isaac62_generated_reference_runtime_134c4f0/`. These are
zero-action technical tests, not learned-policy success tests; user appearance
review remains pending. Account CPU observed during rendering was about 2%;
GPU0 held substantive work and GPU1 only the approved 6-MiB graphics context.
Final audit confirms no owned GPU contexts or reservations remain.

No episode horizon was changed. The original contributed protocol explicitly
specifies 500 controls over ten seconds and 499 saved observations, while its
shared environment extends the horizon to reference length. Read-only timing
checks on the three newly generated references find standing drawer 10.75 s
with switch at 3.5 s, squatting drawer 11.70 s with switch at 6.0 s, and button
15.375 s with runtime FK-derived switch at 6.9 s. These sampled clips support
10 seconds for the declared fixed-length protocol, but do not prove that every
reference fits or that it yields better success. Full-reference behavior and
fixed-ten-second behavior are different experimental choices awaiting review.
Our existing static-pick GR00T chain caps rollouts at 500 controls (10 s), while
the walking chain caps them at 1200 controls (24 s); reference exhaustion and
other termination conditions can end either earlier. Timing evidence is in
`out/isaac62_migration/check_contributed_horizon_timing.json`.

## Matched visual/collision surface and joint-anchor audit (2026-11-16)

The fresh repaired actual G1 asset now also passes a separate matched-import
geometry audit against the original Isaac 5.1 USD: 103 visual/collision surfaces
and 53 joints, maximum surface error 2.682209010451686e-9 m and maximum joint-anchor
matrix error 3.372107105619276e-7, within the existing 1e-6 numeric gate. Collision
flags/approximations and primitive dimensions also agree within that gate.
Twelve primitive rows are not decimal-exact: maximum dimension difference is
7.450580569168253e-10 m, from original float32 values versus new double values.
The receipt records both exact values and errors; no original result is hidden.
This checks authored geometry inputs, not SDK-specific cooked hull equality,
render material equality or learned-policy success.

The initial probe incorrectly classified original collision meshes as visuals:
5.1 authors CollisionAPI on their parent Xforms, whereas 6.2 authors it on the
Mesh. Ancestor-aware traversal fixes the audit without changing either asset.
A subsequent strict string comparison of numeric dimensions also failed; it is
replaced by numeric comparison at the same 1e-6 gate, retaining exact-value flags.
These failed diagnostic receipts remain available. The passing receipt is local
at `out/isaac62_migration/isaac62_matched_surfaces_b2c98ee/receipt.json`; executable
probe is `cluster/isaac62/probe_matched_urdf_surfaces.py`. This is CPU-only and
leaves simulator assets and packages untouched.

## Unbound-material native crash and bounded repair (2026-11-16)

A new native three-strain cube probe exposes a real 6.2 compatibility bug that
explicit-material tests missed. The original 5.1 asset has no bound material and
returns finite, nonzero simulation/collision stresses. Its installed legacy schema
specifies Young's modulus 50,000,000 Pa and Poisson ratio 0.45; the actual stretch
case reports roughly 18.97 MPa axial stress. This rules out substituting zeros for
missing material diagnostics. These results do not establish the 6.2 solver's
implicit defaults or authorize changing a scientific material recipe.

The initial 6.2 attempt exits139 in the native physics tensor plugin. Its log
shows the simulation mesh deleted after a tensor view was created. Installed SDK
source confirms that DeformablePrim explicitly recooks existing auto bodies in
its constructor. The adapter previously instantiated that diagnostics wrapper
after creating its native view. The repaired adapter (`d2515ce`) skips that wrapper
for unbound bodies and uses its existing read-only cooked-topology diagnostics.
No material is created or assigned, and no packages or SDK source are changed.

The repaired unbound case completes all12steps and all three controlled strains;
its simulation/collision deformation-gradient queries are finite. Stress queries
retain their explicit unsupported error. Separately, the explicit two-body,
heterogeneous-topology regression completes20steps, six atomic unsupported-material
rejections and the affine check: maximum gradient error9.64e-6 (gate1e-4), stress
error0.5614Pa (gate10Pa). No task success is assessed. Final checks show no owned
GPU contexts or reservations. All11result files are pulled and SHA-256 verified,
including the failed native attempt, under the three local output directories
`isaac62_implicit_stress_d063608`, `isaac62_implicit_stress_d2515ce` and
`isaac62_explicit_stress_regression_d2515ce` in `out/isaac62_migration/`.

Unbound stress/default-material equivalence remains incomplete. The installed new
USD schema's zero-modulus fallback is not proof of its unbound solver material;
NVIDIA's current documentation also distinguishes schema fallback from scene
consumer defaults. Do not infer the installed solver defaults from either alone.
A next bounded check can compare strain relaxation for unbound and explicit legacy
material values within each SDK, without changing production spawning semantics.

## Recorded parents

- g1: `0505979bf6058b772edb1e7436bb757461aba88f`
- locomotion: `bb9d874aa8e528bc3f5d53c0421fca4a2d4ff9b1`
- Current integration branch: `g1`. No new branch was created.
- Additional preservation parent: Izat's `izat-drawer-button-code` commit
  `14c979c` (2026-11-12), merged without source changes at `5a0e406`.
  Its separate task/controller/data interfaces are part of the migration scope.

## Fresh generated-reference runtime and remaining pose gap (2026-11-16)

Fresh actual G1 conversion at `c45d5b0` passes the source-URDF physical-property
audit for 50 authored inertial bodies and 43 joint limits. The converted asset
has 54 rigid bodies, the original fixed-joint graph, identical zero-force drives
and joint velocity limits. Strict authored rest-world-transform parity remains
failed at maximum matrix-component error 0.000191593459; no tolerance was waived.
Receipts are in `out/isaac62_migration/isaac62_izat_robot_c45d5b0/`.

All three newly generated target references (standing drawer, squatting drawer,
standing button) then pass two-environment, 32-step reset/render tests against
that fresh asset. Each passes strict sensor/renderer camera-mount and contact
ordering checks and produces seven images and a 32-frame, 1280x720, 50-fps video.
All 45 result files were pulled and SHA-256 verified under
`out/isaac62_migration/isaac62_generated_reference_runtime_c45d5b0/`.
The existing `out/reports/dashboard.html` includes all three clips. These are
zero-action integration fixtures, not learned-policy or task-success results;
visual behavior remains for user review. All phases exited zero, their worker
and coordinator processes are absent, and the reservation and GPU contexts
are released.

The live resource audit caught a 6-MiB graphics-only context on Biped GPU1 in
addition to the substantive GPU0 context. This exceeded the current one-GPU
rule; the old late-window exception does not cover migration tests. The suite
was already terminal when root inspected it, so no process was terminated.
Further launches that might reproduce this context are held pending routing
validation or explicit approval. The logs mark only GPU0 active for rendering,
which does not exempt the graphics-only GPU1 allocation from the account rule.

Inspection of the installed Isaac 6.2 `SimulationApp` confirms that `active_gpu`
uses physical graphics indices, while `active_cuda_gpus` honors CUDA ordering.
The launcher now uses the latter and `max_gpu_count=1` when an explicitly
single-GPU application requests a runtime supporting that public option.
Older SDK and intentionally multi-GPU selection behavior is preserved.
`cluster/test_app_gpu_routing.py` passes four CPU checks, including a masked
CUDA-device case and distributed rank selection. This fixes a demonstrated
index-space mismatch; it does not yet prove that the inactive 6-MiB context is
gone. SDK source excerpts are saved in
`out/isaac62_migration/installed_simulation_app_gpu_routing.json`.
Public contract: <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_faq.html>.

The rest-pose investigation separately reproduces every original authored body
position, quaternion and evaluated world matrix exactly using the original
float32 transform accumulation. At the shoulders, the accumulated original
quaternion rounds to a scalar component of exactly one; the original flattened
USD transform evaluates differently from the new nested representation. See
`out/isaac62_migration/legacy_world_pose_prediction_c45d5b0.json`.
An anonymous, CPU-only namespace prototype flattens the 54 body placements,
authors the original poses, preserves all joint/rigid-body properties and all
89 instanceable prims, and leaves no invalid joint relationship targets. Both
rest-matrix error and error after a parent translation are zero. Source USDs
were not modified. This is evidence for a possible repair, not an implemented
or physics-validated converter change. Prototype and receipt:
`out/isaac62_migration/prototype_flat_rest_pose_c45d5b0.{py,json}`.

A second anonymous prototype preserves the existing hierarchy instead of
flattening body namespaces. It authors double local matrices compensating each
body against its parent's final transform. Maximum rest-matrix error is
5.551115123125783e-17; parent translation, joint/rigid properties, relationship
targets and all 89 instanceable prims remain correct. See
`out/isaac62_migration/prototype_matrix_rest_pose_c45d5b0.{py,json}`.
The converter repair candidate uses this approach, retains the SDK's generated
layers and namespaces, and advances the modern-only cache key to revision 14.
Three CPU tests cover the original float32 pose accumulation and malformed
graphs. The committed helper at `1d3acb9` now passes against a separately owned
wrapper of the real asset: all 54 rest matrices agree within
5.551115123125783e-17, all non-transform properties/paths remain exact, repeated
application changes nothing, and a parent rotation plus translation agrees
within 4.440892098500626e-16. SHA-256 verifies that all nine source asset layers
remain unchanged. Receipt:
`out/isaac62_migration/validate_rest_pose_helper_1d3acb9.json`.
Fresh conversion/live physics/rendering is still required. Do not treat the
CPU helper validation as a completed live gate.

The GPU routing CPU tests also pass in both approved Biped environments; receipt:
`out/isaac62_migration/gpu_routing_cpu_a14e178.json`.

## Broader CPU preservation checks (2026-11-16)

The startup-path audit also found that contributed `asset_robot.py` and
`train_ppo.py` left `SimulationApp`'s multi-GPU default in effect. Both now pass
`multi_gpu=False` explicitly to AppLauncher, matching the existing rollout
setting and activating the modern CUDA-index renderer selector. Policy,
optimizer, environment count and task configuration are unchanged. Both actual
stage help entrypoints pass after the edit; live startup remains pending the
graphics-context decision. The original-generation and contributed production
entrypoints do not bypass AppLauncher with another direct SimulationApp call.

For the separately open deformable damping issue, NVIDIA's
[PhysX 5.4-to-5.5 migration guide](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/MigrationTo55.html)
confirms removal of DampingScale and suggests adjusting elasticity damping for
a similar effect. The [Omni Physics migration table](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/deformable_migration.html)
lists no replacement attribute. These sources do not establish an exact
numerical mapping or prove that the installed old solver ignored the attribute.
Keep the non-neutral rejection explicit until paired old/new behavior is
established; do not silently multiply another damping coefficient.

The committed `probe_urdf_rest_poses.py` at `43dc543` passes 28 separately owned
wrapper cases: nine rotated/non-diagonal/merged-body inertia fixtures, four
prismatic/chained/non-cardinal mimic fixtures, eight natural-drive cases, six
fixed/floating/root-selection cases, and the actual G1. Every case preserves
all non-transform prim properties, paths, relationships and instanceability;
input-layer SHA-256 hashes are unchanged. Maximum rest-matrix errors are at
most 1.1102230246251565e-16 for the small fixtures and
5.551115123125783e-17 for G1. Reapplication is exact and parent rotation plus
translation errors are at most 4.440892098500626e-16. These are CPU wrapper
checks, not fresh importer or live PhysX tests.

The initial harness stopped 14 cases because older receipts lacked the newer
`urdf_source` field. Original committed probe sources identify their existing
shared `option_fixture.urdf`; only those 14 cases were rerun, and they pass.
The initial failed harness report is preserved separately from the successful
retry. Local outputs are `isaac62_rest_pose_matrix_43dc543/` and
`isaac62_rest_pose_matrix_43dc543_metadata_retry/` under the migration directory.

The original SMPL and full HumanML loader paths have now also been exercised by
`probe_human_smpl_loader.py` at `1f711f4`, in the previously approved OmniControl
environment on one CPU. The initial test exposed user-site contamination:
`~/.local` NumPy 1.26.4 and Torch 2.7 override the environment's own packages.
SMPL's chumpy import and motion-loader masks fail on removed NumPy aliases;
text-only loading passes. No package or source workaround was applied.

With `PYTHONNOUSERSITE=1`, the same environment's NumPy 1.23.5 and Torch
2.14.0+cu130 complete the bounded CPU gate. The real SMPL model handles
axis-angle, matrices, quaternions and 6D rotations; cross-representation error
is at most 1.4901161193847656e-8. Masked frames and translation are checked
(translation error 5.960464477539063e-8). The a2m, a2mpl, vibe and 6,890-vertex
outputs are finite with their expected dimensions. Actual train, eval,
ground-truth and text-only loader modes each read one batch using their
original eight-worker DataLoader. The available test split contains four
motion examples and five text-only examples; this is not a full-corpus test.
Five recorded model/config/split input checksums remain unchanged. The original
`TrajGen/model`, `TrajGen/data_loaders` and rotation-conversion sources are
unchanged from the original g1 parent. No GPU, diffusion sampling, training,
package installation or input-data edit was involved in this gate. Do not
infer GPU compatibility of this isolated stack from the CPU result; the prior
diffusion fixture used the user-site Torch stack.

Both failure and passing outputs are local in
`isaac62_human_smpl_loader_1f711f4/` and
`isaac62_human_smpl_loader_1f711f4_isolated/`. All 62 pulled files across these
four new result directories match remote SHA-256 checksums. A final live
resource check finds no loader workers, GPU contexts or reservations under
our account, with aggregate account CPU approximately 0.0078% of the host's
64 logical CPUs. Receipts: `cpu_suite_local_file_verification.json` and
`cpu_suite_resources_and_files_1f711f4.json` in local migration outputs.

## Stitch gradient compatibility follow-up (2026-11-16)

The strict full-stitch failure at `a4f8523` is reproducible: joint angles differ
by up to 7.53998756e-5 radians, with 35 of 252 frames exceeding the unchanged
1e-5-radian gate. Root, object, legs and left-arm outputs are identical.
Observation-only tracing at `708f504` is bit-exact with the plain run in each
environment. All seven IK inputs, first FK transforms and first loss are also
identical across environments. The first backward gradient differs by
8.94069672e-8; the Adam updates subsequently diverge. Full trace evidence is in
`out/isaac62_migration/isaac62_stitch_trace_708f504/trace_comparison.json`.

Inspection of the installed library sources identifies a changed default:
pytorch-kinematics 0.9.1 uses ordinary autograd, while 0.10.0 defaults to an
analytical FK backward. The repair at `4dc37e8` requests ordinary autograd via
the public `forward_kinematics_tensor(..., analytical_grad=False)` API when
available. Older libraries retain the original high-level call. Losses,
targets, float32 arithmetic, 1,200 iterations, Adam, learning-rate schedule and
joint limits are unchanged; no packages are installed or patched. All four
paired plain/traced runs completed the full 1,200 iterations and 252-frame
export/reload. Every exported array and all 112 recorded input/FK/loss/gradient/
Adam trace arrays now match bit-for-bit across the environments. The legacy
output also remains bit-exact with the original `a4f8523` legacy fixture.
This closes the existing strict full-stitch numerical gate without relaxing it.
Evidence is in `out/isaac62_migration/isaac62_stitch_trace_4dc37e8/comparison.json`;
all 16 retrieved artifacts have matching remote/local SHA-256 hashes. The CPU-only
test session and workers are terminal, and no account-owned GPU contexts remain.
This is numerical reference-generation validation, not a new visual approval.
Other gradient-based reference refiners also call this library and require a
follow-up audit; a stitch-only repair does not prove their preservation.

The follow-up at `ddb5101` factors that behavior into
`TrajGen/holosoma_adapters/kinematics_autograd.py` and uses it in the stitch,
simple table-clearance refiner and 29-DOF augmented-Lagrangian refiner. The
factory retains the original named-transform dictionary and its iteration
order, including dictionary joint inputs and optional frame selection. It
does not mutate the chain or the installed package. Its paired fixture invokes
the actual Adapter-B script, exports/reloads references, and observes costs,
parameters and gradients. Simple refinement keeps its 1,500-step budget; the
AL fixture uses two outer rounds of 25 Adam updates and a separate rebased
source-path/orientation augmentation pass. These are bounded validation
settings, not new production recipes. All eleven phases at `ddb5101` completed
with exit zero, including the paired original-source/legacy, candidate/legacy
and candidate/target runs and two full-stitch regressions. All three refiner
modes preserve the original legacy export and trace arrays bit-for-bit. Across
environments, joints, root positions, objects, grasp positions and all recorded
costs/parameters/gradients are identical. The augmentation palm path and hand
orientation anchors are also identical. Only the serialized JAXLie global-pose
matrix has the already-observed 5.96046448e-8 rounding difference. Both full
stitch exports still match the accepted `4dc37e8` output bit-for-bit.

The full simple run records 15 trace arrays, bounded AL 18 and AL plus
augmentation 36. Each refiner exports 195 frames. Original receipts and
`comparison.json` are local under
`out/isaac62_migration/isaac62_refiners_ddb5101/`; all 66 retrieved files match
their remote SHA-256 hashes. The session and workers are terminal, with no
account-owned GPU contexts. This validates bounded optimization/export parity,
not AL convergence, motion quality or task success.

The AST consumer inventory in
`out/isaac62_migration/reference_fk_consumer_inventory.json` also identifies
older task-specific refiners, the diagnostic script and Izat's three reference
generators as follow-up candidates. A gradient-bearing file alone is not proof
that every FK call contributes to its loss. Those entrypoints have not been
silently declared covered by the walking fixture.

### Historical optimizer bindings and derivative contract

At `ee6207f`, thirteen historical optimizer/diagnostic scripts explicitly bind
the same ordinary-autograd helper. Their existing direct script invocation is
retained; the import setup locates the shared source relative to each script.
No loss, optimizer, target, iteration, visualization branch or file-writing
behavior is changed. A whole-program AST comparison against `19939bb`, after
normalizing only this import setup and FK binding, passes for all thirteen
files (`out/isaac62_migration/legacy_refiner_ast_ee6207f.json`).

The real shared FK derivative fixture passes sixteen cases in both installed
environments: 27/29 DOF, batches one/seven, tensor/dictionary joint inputs,
and all 39 links or three explicitly selected links. All 64 saved input,
transform, weighted-loss and gradient arrays match bit-for-bit, including link
dictionary order. Evidence is local under
`out/isaac62_migration/isaac62_fk_autograd_ee6207f/`. This is source-preservation
and derivative-component evidence; it does not claim execution of the older
interactive generator/viewer entrypoints.

### Contributed generator headless follow-up

The `0cc3ce9` repair uses that same FK binding in Izat's standing/squatting
drawer and standing-button generators. It also defers the existing Viser,
yourdfpy and PyRoKi imports and viewer-only URDF/heightmap construction until
their original visualization branches. These dependencies are present in the
original reference environment but absent from the target simulator profile;
no installation, fake module or dependency replacement is used. The complete
cost and target-trajectory function ASTs are unchanged except for the FK binding
(`out/isaac62_migration/contributed_ik_objective_ast_0cc3ce9.json`).

The paired fixture invokes the real `prepare-references` and generator CLIs,
with trusted original retargeted inputs and one seeded target per kind. It
retains each generator's full per-point optimization budget and original early
stop, plus reference export/reload. Original-source/legacy, candidate/legacy
and candidate/target runs are compared separately. All nine phases completed
with exit zero. The original and refactored legacy outputs and recorded cost,
parameter and gradient traces are bit-exact. Every exported array and trace
also matches the target environment bit-for-bit, including root quaternions,
global pose matrices and task target metadata. Standing/squatting drawer each
use 2,000 backward steps and export 215/234 frames; button uses 10,000 steps and
exports 205 frames. All target runs load none of Viser, yourdfpy or PyRoKi.
Evidence is local in
`out/isaac62_migration/isaac62_contributed_ik_0cc3ce9/comparison.json`.
The session and workers are terminal; no account-owned GPU contexts remain.
These are complete per-point generation checks, not an entire grid, a physical
task-success evaluation, an interactive-viewer test or user appearance approval.
The next integration gate will load these new references in the simulator and
render them with a freshly converted robot asset using the current importer.

## Drawer/button contribution integration (2026-11-15)

Izat's 69-file contribution under `contrib/drawer_button` is now included in
`g1`. The initial merge matched the contribution parent exactly; subsequent
compatibility fixes are recorded below. It adds standing and
squatting drawer references, standing-button references, isolated SONIC PPO
tasks/controllers, explicit action/history contracts, successful-episode
recording, conversion, native GR00T fine-tuning/serving, evaluation banks and
experimental DAgger bookkeeping. Preserve its task IDs and scientific contracts;
do not replace its 78-dimensional actions with the walking collector's interface.

The unchanged contributed CPU suite passes 77 tests, with two explicit skips
(real exporter and native GR00T configuration integration), in the existing
Bluesclues `dreamcontrol_51` profile. All 14 stage help commands pass. No packages
were installed. Receipts are local in
`out/isaac62_migration/check_izat_cpu_sim.json`. A preceding attempt in the GR00T
profile could not run because that environment does not contain pytest; it is
not a passing test result.

The contribution's own dependency inventory records Isaac 4.5, and its README
explicitly disclaims completed GPU integration. Its isolated task/controller
snapshots require their own tests; the earlier native drawer/button gates do not
cover them. The later live-gate records below establish configuration, runtime,
rendering, PPO restore, conversion, original motion generation and native GR00T
training checks. Production collection and both evaluation tracks remain open.
Historical success rates are not evidence for these newly integrated entrypoints.

### Current contributed integration follow-up (2026-11-16)

The real `train_ppo` CLI at `798ee2b` completed one drawer iteration with two
environments and its original rollout/optimizer settings. It wrote `model_0.pt`,
the training contract, configuration and completion receipt. The saved model's
68 tensors and 51 optimizer state tensors are finite. This complements the
separate two-update/save/exact-restore/resumed-update gates for both tasks; it
is not a task-performance result.

The original button panel's build initially failed because its referenced
`models/ElevatorButtonUpDown.usd` was not copied. The dependency fix `0d5c050`
includes that source model and its texture, with original Git LFS hashes checked.
Actual rendering then exposed inherited USD clones sharing authored transforms.
The fixes through `45503e9` preserve the original asset, allow only verified
inherited visual clones, and author independent initial translations and cap
positions. The actual USD inheritance test checks independent placement,
animation, reset and atomic rejection of a duplicate local panel. The real
two-environment button render at `45503e9` passes 32 steps, resets, strict camera
and renderer poses, contact ordering and the production visual helper. Its
original frames and MP4 are local under
`out/isaac62_migration/isaac62_izat_button_runtime_45503e9/`. The panel's cosmetic
cap is not a simulated physical press or a task-success assertion.

The contributed native GR00T training path at `5b8c101` completed one actual
Adam update and exported three safetensors files totaling 12,576,200,896 bytes.
Loss is 0.6314754486, optimizer step is one, and sampled trainable weights change
by up to 9.99663e-5 with finite values. It exercises the actual native dataset,
processor, model, trainer and final save through the contributed launcher.
The test uses the existing walking checkpoint and a 50-frame converted synthetic
fixture, with explicitly recorded batch-one/one-update/no-periodic-checkpoint
overrides. It neither establishes task success nor validates optimizer resume;
the full Adam fresh-process resume gate is separately recorded at `107ade9`.
Peak allocated CUDA memory is 38,516,082,688 bytes. Training and its reservation
are terminal and released. Receipt and log are local in
`out/isaac62_migration/isaac62_izat_native_train_5b8c101/`; the exported model
remains on Bluesclues. Fresh-process contributed serving at `7b4a713` also
passes: the actual `serve.py` reloads that export, accepts protocol ping/reset,
and produces finite `(40,64)`, `(40,7)` and `(40,7)` arrays for two requests.
The requests use seeds 17 and 19, not a claim of bitwise repeatability. Inference
times are 0.5964 and 0.1499 seconds. Protocol shutdown returns successfully and
the fresh server exits zero. Raw model predictions are continuous; this test
does not move the collector's later FSQ operation into the server. Original
receipts, predictions and logs are local in
`out/isaac62_migration/isaac62_izat_native_serve_7b4a713/`.

An original episode-timing inconsistency requires user review before production
collection/evaluation: `rollout.py` asserts 500 steps at 0.02 s and expects 499
recorded frames, but the shared environment extends the selected reference to
approximately 15.38 seconds. Both behaviors also exist in the contribution's
original parent. No timing behavior has been changed. The concrete options are
in `out/isaac62_migration/contributed_horizon_review_plan.json`. This hold does
not prevent independent model/protocol and native-task regression checks.

The read-only physical-property audit at `7537c9c` compares the converted robot
against the original URDF, covering all 50 authored inertial bodies and all 43
movable joint limits. Maximum absolute errors are 2.75e-7 kg for mass, 7.25e-9 m
for center of mass, 1.51e-9 kg m² for the reconstructed inertia tensor, and
9.46e-8 rad for joint limits. All fixed gates pass. This does not establish
collision geometry or dynamic trajectory equivalence. Local receipt:
`out/isaac62_migration/isaac62_urdf_physics_7537c9c/receipt.json`.

A native pick regression initially failed twice before stepping because its
working directory selected unmaterialized Git LFS pointers in the validation
checkout. File existence alone missed this provisioning error. The real robot,
bottle and kitchen assets in the original checkout match the current pointer
SHA256 OIDs. The corrected fixture uses that original asset working directory,
while imports still use the updated editable validation checkout. An initial
speculative source change to absolute-path resolution was ineffective and was
removed at `bbce322`; shared spawning behavior is unchanged. Both failed receipts
are preserved, and neither is counted as a passing regression.
The corrected native pick fixture at `bbce322` then completed 32 steps, initial
and final resets, ten nonconstant RGB images and a 1280x720, 50 fps, 32-frame
video. It uses the original Isaac 5.1 scene assets in Isaac 6.2 and zero policy
actions. Its task success is not assessed, and it does not contain the stricter
contributed camera-mount/contact-position diagnostics. Original artifacts are
local in `out/isaac62_migration/isaac62_native_pick_regression_bbce322/` and the
generic review dashboard references the original video without recompression.
The stitched-reference sibling `isaac62_native_walk_regression_bbce322` passes
the same 32-step/reset/render gate using `StitchMAX200v3/pick_14030.pkl`, with
zero terminations or truncations. This is a technical scene regression, not a
learned walking-policy performance evaluation. Both processes, all account-owned
GPU contexts and their reservations are confirmed released. The final native
walking-reference account CPU sample is 0.00017% of host capacity.

### Robot collision and rest-frame audit (2026-11-16)

The actual USD audit at `3a2e8dc` traverses instance proxies and compares every
collision against the original URDF: 39 convex-hull mesh inputs, eight spheres,
four cylinders and two boxes. All 53 pass a fixed 1e-6 geometric tolerance;
maximum observed error is 3.43e-8. Mesh source SHA256 values are recorded and
their transformed point sets match the STL inputs. Primitive centers, sizes and
axes match, collisions remain enabled, and there are no missing or extra shapes.
This checks the inputs to convex cooking, not SDK-specific cooked hull equality.
All 43 movable joint axes and anchors also pass, with maximum error 2.25e-7.

The strict ideal-URDF rest-transform gate remains failed: 53 of 54 link
frames pass, while `mid360_link` rotation differs by 1.73377246e-6 from the
ideal matrix (fixed tolerance 1e-6). The matched original converter produces
that same rotation matrix. The original report is retained at
`out/isaac62_migration/isaac62_urdf_geometry_3a2e8dc/receipt.json`.
Its assumption that four empty sensor links should remain non-rigid was
subsequently disproved by the matched SDK5.1 conversion described below.

### Matched robot physics and named policy interfaces (2026-11-16)

Running the original g1 converter on the identical URDF in the existing Isaac
5.1 environment exposed a real migration regression: the legacy import has 54
rigid bodies, while the initial modern import had only 50. PhysX assigns each of
the four empty fixed sensor links a default effective mass of 1 kg. Omitting
those links reduced the robot from 40.1682 kg to 36.1682 kg. This is not an
approved physics change. The bridge now preserves those four fixed bodies with
explicit measured legacy mass, center of mass and inertia. It does not modify
the source URDF or the SDK-generated source layers.

The repaired asset `isaac62_izat_robot_b859963` has 54 rigid bodies and 53 joints
(43 movable, ten fixed). Paired live PhysX checks in the same SDK6.2 process
show exact per-body masses and centers of mass, no missing or extra bodies,
and maximum inertia-element difference 7.45e-9. The small total-mass difference
is floating-point summation order. The matched original conversion initially
tripped the fixture's incorrect 50-body expectation after successfully writing
the asset; that failed receipt is retained, with independent USD/live checks
providing the comparison evidence.

Raw joint ordering differs between the legacy and modern imports, including
hand index/thumb indices, so `preservation_pass` remains false for that strict
raw-order gate. The actual named application interfaces were tested separately
at `1bf401d`: all 43 canonical dataset/GR00T joint positions, the actual 29-joint
PPO `joint_pos_rel` observation and both seven-joint binary finger outputs
match exactly. This uses original exported dataset metadata and the real scene
entity resolver and action terms, not a surrogate ordering implementation.
`named_io_pass` is true; raw ordering is explicitly recorded as unequal. Receipt:
`out/isaac62_migration/isaac62_urdf_live_mass_1bf401d/receipt.json`.

The repaired 54-body button task also passes 32 steps, reset, strict sensor and
renderer poses, contact-row ordering and file rendering at `8d1fa08`. Original
frames/video are local in `isaac62_izat_button_runtime_8d1fa08`. The repaired drawer task at `1bf401d` also passes the same 32-step/reset,
strict camera/renderer and 54-body contact gates. Its original video and frames
are local in `isaac62_izat_drawer_runtime_1bf401d`. Drawer PPO at `8342540`
passes two bounded updates, exact model/Adam restoration into a new runner and
one resumed update with changed parameters. This is the same simulator process,
not fresh-process or RNG parity. Button PPO at `51cef96` also passes those update/save/exact-restore/resumed-update
gates. Both processes, GPU contexts and reservations are confirmed released.
These are technical integration checks, not task-performance results.

The matched SDK5.1 conversion used complete original source/application files
and supported process-local Kit settings to avoid shared user preferences, as
used in NVIDIA's [startup tests](https://github.com/isaac-sim/IsaacSim/blob/develop/repo.toml).
No packages, shared user configuration or extension trust settings were changed.

### Matched rest-transform follow-up (2026-11-16)

The repaired geometry audit at `51cef96` checks all 54 rigid link transforms,
43 movable joint axes/anchors and all 53 collision shapes. Collision and axis
checks pass; the ideal-URDF `mid360_link` error remains 1.73377246e-6 at the
unchanged 1e-6 tolerance. A separate matched-original transform comparison also
fails, with maximum matrix-component difference 0.000191593459 concentrated on
shoulder descendants. Do not infer that matched preservation passed from the
single ideal-URDF failure list: `matched_baseline.preservation_pass` is false.

A read-only diagnostic records the original shoulder's authored quaternion
(w=1 with small nonzero xyz components), its evaluated identity rotation and
the modern hierarchical transform retaining a small rotation. The same-named
shoulder pitch frames agree to numerical precision; the discrepancy begins at
shoulder roll. No transform replacement or tolerance waiver has been applied.
Evidence: `isaac62_urdf_geometry_51cef96/receipt.json` and
`diagnose_matched_shoulders.json` in local migration outputs.

### Live PhysX pose parity (2026-11-16)

The paired `cef7837` test reads actual PhysX link transforms for all 54 bodies,
separately from USD/Gf matrices. After aligning at the pelvis, the initial poses
pass the fixed 1e-6 position/rotation-matrix gate (maxima 2.98e-7 m / 7.11e-7).
An identical nonzero named joint state with identical root poses, evaluated by
forward kinematics without a physics advance, also passes (4.77e-7 m / 8.64e-7).
Named model/finger interfaces remain exact. Thus the measured USD shoulder
matrix discrepancy does not appear in these actual PhysX pose comparisons.
This does not waive the USD gate, prove arbitrary-trajectory dynamics parity or
change the separately failed raw-joint-order flag. Both paired articulations
and their reservation are terminal/released. Evidence is local under
`isaac62_urdf_live_mass_cef7837/`.

### Native camera verification follow-up (2026-11-16)

The new native fixture validates sensor and renderer poses against the physical
robot ancestor and unchanged mount/configuration, with a fixed 1e-5 gate.
The initial `35ecfad` fixture failed on treating a regex configuration path as
a concrete prim; `462dc42` resolves the actual sensor prim. That run captures
all 32 frames and passes three 44-body contact-order checks, but its camera
rotation oracle fails at up to 1.47e-4. The accepted rounded camera quaternion
has squared norm 0.99973. A paired CPU diagnostic confirms SDK5.1 and SDK6.2
have identical Gf quaternion semantics; normalizing the input predicts a
different orientation from the original USD authoring path. The `8c42258`
fixture uses that original Gf.Rotation interpretation and records the normalized
alternative separately. Its native pick run passes all three camera comparisons:
maximum physical position/rotation errors 1.57e-7 m / 9.76e-8 and maximum
renderer position/rotation errors 1.64e-7 m / 1.57e-7. All three 44-body contact
checks are exact. The normalized alternative remains different by 1.28648e-4.
Original 32-frame video and images are local in
`isaac62_native_pick_regression_8c42258`; the process and reservation are released.
No production camera value or tolerance was changed. The stitched-reference
counterpart at the same commit also passes 32 steps and all three contact checks
with 44 bodies. Maximum physical position/rotation errors are 8.69e-8 m /
1.001e-7; renderer errors are 9.06e-8 m / 1.366e-7. Its original images and
32-frame MP4 are local in `isaac62_native_walk_regression_8c42258`. Both native
workers, reservations and all account-owned GPU contexts are confirmed released.
These are zero-action bounded technical checks, not policy-success estimates.

The native kitchen logs reference three missing EXR files not present among
tracked textures: `beige_wall_001_nor_gl_1k.exr`,
`laminate_floor_03_rough_1k.exr`, and `laminate_floor_03_nor_gl_1k.exr`.
Thirty collected JPG/PNG/MDL dependencies were checked remotely and are real
files, not LFS pointers. MDL texture_return-to-color conversion errors also
remain. These asset/material issues are not hidden by the RGB non-black check;
no replacement textures or unreviewed appearance changes have been applied.

### Matched kitchen renderer diagnostics (2026-11-16)

The same static kitchen fixture now completes under both existing SDK5.1 and
SDK6.2 profiles at `c1f871c`. It uses the identical source USD checksum,
backdrop transform, camera and fixture lighting. Both capture three non-black
640x480 RGBA frames and both log the same three missing EXRs and the same
`door_paint_001/MDL_TypeCast1/{x,y,z}` texture_return-to-color errors. These
specific warnings therefore predate the migration; they do not establish a
new 6.2 regression. No production texture, material or camera was modified.
The first 5.1 fixture captured images but failed annotator teardown because it
passed a HydraTexture instead of a path; that failed receipt is retained and
the corrected fixture completes in both SDKs. All workers/reservations and
account-owned GPU contexts are released.

Original images are linked side by side in
`out/isaac62_migration/paired_kitchen_render_review.html`. Numeric/provenance
comparison: `paired_kitchen_render_comparison.json`. User appearance review is
pending; no visual equivalence or robot-success claim is made.

### Remaining URDF options and velocity-limit repair (2026-11-16)

The expanded original-g1 probe at `0328707` completes eight independent
primitive fixtures: default, collision capsule replacement, the mimic flag,
explicit base/interior root names, instanceable capsule replacement, and
force/acceleration natural-frequency drives. Unlike the earlier probe, one
failure no longer hides the remaining option results. Source visual cylinders
are retained while collision cylinders become capsules, with source float32
height/radius and capsule extents. Both root-name requests keep all three bodies
and the original fixed-root articulation API location; this small fixture alone
does not establish general rerooting semantics.

The original `convert_mimic_joints_to_normal_joints` flag is inverted relative
to its name: true preserves a PhysX mimic constraint and suppresses that
joint's drive, whereas false produces independent driven joints. The actual
original outputs and source both establish this. Preserve that observed
contract; do not silently correct the flag's meaning during the refactor.
Original receipt: `isaac51_urdf_options_0328707/receipt.json` in local migration
outputs. The corresponding pre-repair 6.2 receipt records seven independently
failed cases across four option families, rather than stopping at the first.

The bridge now implements collision-only capsule replacement in its owned USD
wrapper, retaining visual geometry, transforms, APIs and bindings. Float32 source
dimensions and analytic capsule extents match the original importer. Unexpected
instanced collision-cylinder proxies still reject explicitly instead of changing
instanceability. The first real retry at `24f2ef3` exposed that USD Python does
not bind the C++ static capsule extent overload; its failed evidence is retained.
The repair uses explicit local half-extents. Modern cache revision 4 invalidates
assets from before this feature. The full independent matrix at `3e3ab34` completes all measurements: default
and both capsule cases pass; mimic, two root-name requests and two natural-drive
cases remain explicit failures. Collision capsule type, dimensions, axis, extent
and standard hinge gains compare exactly against the actual 5.1 outputs. Visual
cylinders are unchanged from the modern default, and their type/axis/extent match
5.1. Their dimensions retain a small existing double/float32 difference (.4
versus .4000000059604645 for height); no new visual change was introduced by the
capsule option. The final process returns exit 1 as required by five failed cases.
All worker, tmux, reservation and account GPU contexts were released; external
jobs were untouched. Receipts: `isaac62_urdf_options_3e3ab34/receipt.json` and
`capsule_compatibility_comparison.json`. This proves conversion output, not new
physics or visual-behavior equivalence.

The probe also exposed Kit shutdown swallowing pending Python failures and
returning exit zero. Its shutdown now passes the receipt's success/failure code
through the supported SDK close arguments. A zero wrapper status from the older
probe must never override its failed receipt.

The mimic implementation's first paired metadata test at `3e47ad4` matches
original axes, explicit position bounds, velocity metadata, standard gains and
all five PhysX mimic parameters exactly. This required preserving original
float32 position-limit conversion before the mimic's 20% source-limit expansion.
Do not treat matching authored USD alone as behavioral proof: the subsequent
four-articulation live comparison at `b976fdf` failed both flag states over 240
steps (maximum named-joint errors .4795514 and .1329596 rad, against a fixed
.0002-rad gate). The modern default joint angles remained coupled with a .1-rad
offset despite the legacy flag being false. The suspected competing Newton mimic
metadata is now explicitly disabled in the PhysX wrapper; the original PhysX
constraint remains conditional on the observed legacy flag. The repeated live
comparison at `d241860` passes both flag states without changing the .0002-rad
threshold: maximum errors are .0000366122 and .0001472086 rad. The .5401598-rad
enabled/disabled finger response gap proves the fixture actually distinguishes
both settings. Effective velocity limits match exactly, including the unbounded
mimic limit (5.9390470094e36 in both assets). This isolates the competing Newton
constraint as the cause of the large earlier failure. Both original-generated
and new assets run in the same 6.2 engine; this is importer parity, not a claim
of cross-engine determinism. Worker, tmux, reservation and account GPU contexts
were released; other users' jobs were untouched.

The test also exposed why the original unbounded mimic velocity must be
preserved at the engine level rather than authoring the USD schema's nominal
1e6 fallback as an actual limit. The wrapper removes its ordinary-joint limit
and blocks competing Newton velocity metadata for mimic joints. USD schema
fallbacks differ between the installed SDKs (old 1e6, new infinity), but actual
PhysX limits match. Metadata-only equality is insufficient. Evidence:
`isaac62_urdf_mimic_runtime_b976fdf/receipt.json` (failed),
`isaac62_urdf_mimic_runtime_d241860/receipt.json` (passed), and
`isaac62_urdf_options_d241860/receipt.json`.

Remaining mimic scope includes non-cardinal axes (explicitly rejected), chained
references, prismatic cases and their live validation. The initial supported
cardinal rotational fixture is not evidence for every URDF mimic arrangement.
An independent authored-default audit also found original articulation solver
settings 32 position/1 velocity iterations, whereas the initial new asset left
those fields unauthored (`mimic_solver_metadata.json`). The next section records
the subsequent repair and validation of those defaults.

An independent read of original root overrides establishes a salient behavior
not covered by the earlier root-API-location metric: `root_link_name='arm'`
rewires the world fixed joint's body1 from `base` to `arm` with identity local
anchors, while leaving all bodies and the articulation-root API path present.
This matches the pinned original C++ implementation; it is not a no-op or a
request to discard other bodies. Evidence: `original_root_override_metadata.json`.
At that point the root-override port was still unsupported. The following
section records its subsequent implementation and fixed/floating-base tests.

### Root selection, solver defaults and diagonal mass frames (2026-11-16)

The new converter now preserves the original fixed/floating root-link contract:
all links remain; fixed bases attach the world joint to the selected link with
identity anchors and put ArticulationRootAPI on that joint; floating bases put
the API on the selected body. Generated alternative articulation-root schemas
are removed to avoid competing root choices. Original solver defaults are now
explicit: 32 position iterations, 1 velocity iteration and the configured
self-collision setting. The five actual original/new conversion cases at
`6d31a77` match root roles, attachment targets/anchors and solver fields exactly.
Evidence: `root_metadata_comparison.json` and the two SDK receipts.

Live validation exposed a small but real additional difference. The first
240-step root comparison at `ee385ee` passes four cases, but a floating interior
root misses the fixed quaternion-component gate (1.015251e-4 versus 1e-4).
An identical-original-asset control passes, and a same-origin, collision-filtered
comparison at `1609106` reproduces the miss. Thus placement alone does not explain
it; neither failed result is waived. Effective masses, body inertia tensors and
COM positions match exactly, but the modern converter permutes eigenvectors of
already-diagonal tensors, giving different principal mass frames. The original
`ImportHelpers.cpp::Diagonalize` exits at identity for diagonal tensors.

The wrapper now preserves that identity frame and original diagonal ordering
for positive diagonal inertias with unmerged links and zero inertial rotation.
It does not assume this shortcut applies to merged, rotated or non-diagonal
inertias; those broader mass-frame cases remain to be audited. After this repair
at `0bfa259`, the same-origin test passes all five root cases with unchanged
thresholds. Floating-root maxima are 3.20375e-7 rad in named joint positions,
1.18511e-6 m in body positions and 2.02656e-6 in quaternion components. Measured
mass frames now agree as well. Replaying the original *separated* test with the
same pre-repair probe source (`ee385ee`) and repaired assets also passes all
five, including floating-root quaternion error 2.00048e-6. This establishes a
repair of the original failed gate, not just a passing alternative fixture.

Receipts: `isaac62_urdf_mimic_runtime_ee385ee/` (failed),
`isaac62_urdf_root_placement_control_ee385ee/` (identical-original control),
`isaac62_urdf_mimic_runtime_1609106/` (failed same-origin),
`isaac62_urdf_mimic_runtime_0bfa259/` (repaired same-origin), and
`isaac62_urdf_root_separated_repaired_ee385ee/` (repaired original test).
All these jobs are terminal, their reservations released and account GPU
contexts absent. The follow-up `isaac62_urdf_mimic_regression_0bfa259` also passes
both mimic flag states after the root/solver/mass-frame changes, with maximum
joint errors 1.19209e-7 and 2.08616e-7 rad, body-position errors below 3.6e-7 m,
and quaternion errors below 3.9e-7. Effective velocity limits still match exactly.
This proves paired importer behavior in the 6.2 engine for this primitive
fixture, not complete cross-engine numerical determinism.

The actual contributed G1 was then freshly converted at `0bfa259` and tested
against the original asset, rather than relying on the three-link fixture.
The live test retains all 54 bodies and 43 joints, with no missing/extra bodies,
exact masses and COM positions, inertia differences at most 7.45e-9 and all 43
effective velocity limits exact with actuator overrides disabled. Initial and
canonical nonzero-state physical poses pass the existing 1e-6 gates (maximum
position 4.77e-7 m, rotation-matrix component 8.64e-7). Actual canonical GR00T,
29-joint PPO and both binary hand-command interfaces pass. Raw hand-joint order
still differs, so its separate strict preservation flag remains false; named
mapping is required and tested. Receipt: `isaac62_urdf_live_mass_0bfa259/receipt.json`.

The fresh G1 also passes the real contributed button task regression at
`0bfa259`: two environments, 32 zero-action steps, reset and final reset, all
three 54-body contact-order checks exact, and all three camera mount/renderer
checks passed. Maximum camera position/quaternion-component errors are
2.3842e-7 m / 1.7881e-7; renderer position/rotation-matrix errors are at most
1.9916e-7 m / 2.8678e-7. Seven sampled PNGs and the original third-person video
pass basic nonblack/media checks. The video has 32 frames at 1280x720, 50 fps,
0.64 s. No learned-policy success or subjective appearance is assessed.
Original artifacts are local at `isaac62_izat_button_runtime_0bfa259/`; the
shared dashboard now indexes 75 runs without recompressing media. User visual
review remains pending. The renderer, reservation and all account GPU contexts
were released; a local SSH/SCP client review found no remaining clients.

A further audit found modern imported joints lacked authored PhysX velocity
limits, retaining only Newton metadata. The bridge now authors the original
PhysX field from URDF limits in its owned wrapper. Angular conversion matches
the original C++ float32 parsing and `180.0f / float(M_PI)` rounding exactly.
The first repair's live values differed by up to 1.90735e-6 and that strict
failure is retained. The fresh `5844642` G1 conversion matches every original
USD velocity-limit float bit-for-bit across all 43 joints and retains all 54
bodies. The live old/pre-fix/repaired comparison at `f695448`, with actuator
overrides disabled, also matches all 43 repaired effective limits exactly.
The preserved pre-fix asset still has finite, nearly identical effective limits:
for example, a hip limit is 32.0 versus the old/repaired 31.99999809265 rad/s.
Its maximum discrepancy is 1.90735e-6. Thus absent authored PhysX attributes did
not mean disabled limits; the measured runtime difference was rounding.
Both live pose gates and named model/finger interfaces still pass. Raw joint
order remains different and that separate strict flag stays false. Receipt:
`out/isaac62_migration/urdf_live_mass_f695448/receipt.json`. Reference implementation:
[NVIDIA's original AddSingleJoint](https://github.com/isaac-sim/urdf-importer-extension/blob/9908f0341ee4c375a94d594c4eeb318dea3f4cf2/source/extensions/isaacsim.asset.importer.urdf/plugins/import/UrdfImporter.cpp).

Modern URDF cache keys are versioned so cached pre-repair assets are regenerated.
A failed modern conversion also removes its early-written cache marker, so a
partially generated USD cannot be mistaken for a complete conversion on retry.
Legacy SDK cache keys remain unchanged. The actual fault-injection test at
`f695448` passes: a converter that writes a partial USD then raises leaves no
cache marker; the next construction reconverts; a later construction reuses
the valid cache; modern old keys differ while legacy keys match. The test uses
the actual converter/base control flow with a process-local injected conversion
function, not an installed dependency patch. Receipt:
`out/isaac62_migration/urdf_cache_f695448/receipt.json`. All test processes,
reservations and account-owned GPU contexts are confirmed released.

### Contributed live gates (2026-11-15)

The actual contributed `sample-human` entrypoint completed one 196-frame
button-press XYZ sample in the approved existing OmniControl environment.
It loaded original checkpoint SHA256
`886b722e95496f4ceafd704c4e1d0578888733d12ce67429677b80a91a9c36d0`,
the original four normalization arrays, CFG 2.5 and all 1,000 diffusion steps.
The generated motion `(1,22,3,196)` and hints `(1,196,22,3)` are finite.
The contributed process-local XYZ adapter does not exercise SMPL or the dataset
loader. Runtime was 176.09 seconds. Initial tmux environment forwarding failed
before package import; a verified terminal failure and released reservation
preceded the corrected launch. No dependencies or model implementation were
patched. Original output and provenance are local in
`isaac62_izat_human_46f367a/`. Both owned processes and GPU allocation were
confirmed released before the next fixture.

The real Isaac 6.2 gate at `69f4e96` registers and constructs both contributed
task configurations and their actual PPO runner configurations. Both pass with
two environments, physics dt 0.005, decimation 4 and three original action terms
(`joint_pos`, `left_hand_action`, `right_hand_action`). This does not instantiate
the environments or prove stepping/rendering. Full configurations, log and
terminal zero are local in `isaac62_izat_configs_69f4e96/`; no owned GPU contexts
or reservations remain.

The actual contributed fine-tuning launcher also assembles a native configuration
against the installed GR00T implementation at `3281663`: 35,554 steps for the
99,800-frame fixture at 5.7 passes, effective batch 16, constant 1e-4 learning
rate, frozen VLM, horizon 40 and absolute actions. Only the final training
execution boundary is intercepted, using stdlib mocking; no model is loaded.
This closes the configuration-assembly gate without installing pytest into the
production GR00T profile. It does not establish contributed training or serving.
Local receipt: `out/isaac62_migration/check_izat_native_config.json`.

### Real exporter and URDF migration fixes (2026-11-16)

The contributed converter failed against the installed SONIC exporter because
that schema additionally requires robot/object world-pose telemetry absent from
the contributed raw recordings. The converter now retains its original recorded
schema, omitting only those three unrecorded telemetry fields and rejecting that
omission if they ever become configured GR00T modalities. It does not invent
world poses or alter policy observations/actions. The actual 50-frame integration
at `acef058` passes exact comparisons of every recorded policy array after the
declared storage cast, all 50 decoded RGB frames at 50 fps, unchanged source hashes
and byte-unchanged completed-dataset resume. Scalar width-one Parquet columns
are checked explicitly. Non-FSQ synthetic labels were correctly rejected during
fixture development; the final fixture retains that validation. Receipts:
`out/isaac62_migration/check_izat_export_fsq.json` and the earlier failed
`check_izat_export.json`.

Actual robot preparation exposed a shared-framework migration gap:
`isaacsim.asset.importer.urdf._urdf` no longer exists in SDK 6.2. The contributed
asset command also allowed fast application shutdown to hide an exception behind
exit code zero. It now prints errors before shutdown, uses the supported SDK exit
code when available, and rejects a missing USD. The shared URDF converter retains
its legacy path and bridges to the public `URDFImporter`/`URDFImporterConfig`
when available. This follows NVIDIA's
[URDF migration contract](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/migration_guides/isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline.html).
The bridge retains the requested asset filename using relative layered references
and explicitly selects the generated PhysX variant; without that selection the
first generated asset composed zero physics joints.

Robot preparation at `bb570ce` exits zero and produces a composable asset. A
separate read-only USD audit finds exactly the original 43 movable joint names,
all requested force drives with zero stiffness/damping, a valid default prim and
instanceable geometry. This is not yet physics/rendering proof, nor legacy/target
robot-geometry parity. The unused generated MuJoCo layer warns about absent gains;
the selected PhysX layer's requested zero gains are verified. Audit receipt:
`out/isaac62_migration/audit_izat_robot_physx.json`. A later API-detection-only
change at `fc061e6` avoids assuming wheel metadata for standalone SDK installs;
its next runtime regression remains to be executed.

Natural-frequency drives were initially rejected explicitly rather than silently
dropped. The preservation implementation and paired evidence are recorded in the
natural-drive follow-up below; broader inertia and mimic coverage remains open.
The SDK also emits a different robot hierarchy; camera attachment paths and
full collision/inertia/limit/physics/rendering parity still need validation before
accepting the contributed simulation pipeline.

### Natural-frequency drive preservation (2026-11-16)

The original importer?s equivalent-inertia calculation is ported with its float32
rounding and original traversal order. On the actual original G1 asset, all 43
joint inertias match bit for bit (`g1_legacy_natural_inertia_audit.json`). The
implementation preserves the original, nonstandard use of inertia magnitudes
rather than substituting a different physical model.

Source `b607d99` introduced this port and exposed two differences in the expanded
original fixture: an unmatched frequency regex retains the parser?s unscaled
strength of 625, and the URDF spring field is `spring_stiffness`. Fix `1cd0602`
preserves both, including raw URDF damping for unmatched joints. Converter cache
version 11 invalidates earlier generated compatibility assets.

Fresh original 5.1 and target 6.2 fixtures each complete all eight cases: default,
force/acceleration natural drives, velocity/none targets, mixed regexes, partial
frequency coverage and the original mimic flag. Every natural-drive stiffness,
damping and drive type matches exactly. The `none` fixture exercises nonzero
spring stiffness 0.31 and damping 0.27. Receipts are in
`isaac{51,62}_urdf_options_1cd0602/` under local migration output.

The stricter metadata comparison still reports two differences: the default PD
path lacks the original equivalent-inertia annotation, and the constrained mimic
joint uses the target schema?s infinite velocity default instead of the legacy
1,000,000 default. These are recorded, not treated as an all-properties pass;
see `natural_drive_metadata_comparison_1cd0602.json`. The subsequent natural-mimic live test below establishes effective velocity and
response preservation for this fixture despite the schema-default difference.

The paired 240-step force/acceleration response test uses the actual imported
gains and limits, without the earlier common-gain overrides. Both pass the existing
joint/body/orientation gates: maximum joint error is 2.98023e-8 rad for force
and 7.45058e-8 rad for acceleration; maximum body-position error is 3.27826e-7 m.
Effective stiffness/damping match exactly. Hinges traverse more than 0.55 rad,
so the comparison exercises motion. Evidence:
`isaac62_urdf_natural_runtime_1cd0602/receipt.json`. The fixture exits zero and
releases its GPU process/reservation. A separate 240-step natural-mimic fixture
also passes: maximum joint error 8.94070e-8 rad and body-position error
2.94298e-7 m. Effective gains and velocity limits match exactly, including the
unactuated mimic finger. Evidence:
`isaac62_urdf_natural_mimic_runtime_1cd0602/receipt.json`. Its process, reservation
and GPU contexts are also released. Prismatic and chained/non-cardinal
mimic fixtures and merged/rotated/non-diagonal inertia coverage remain open.

### Prismatic and chained mimic follow-up (2026-11-16)

The expanded source probe `d17c38b` adds prismatic PD, prismatic natural-frequency,
a three-joint mimic chain and a non-cardinal-axis case. It preserves each tiny
source URDF alongside its conversion receipts. The original SDK completes all
four conversions. The target completes the first three with identical joint
types, axes, limits, drive types and gains. Its non-cardinal case is explicitly
rejected by the existing preservation guard; the overall fixture correctly exits
one after still recording all independent case results. Evidence:
`extended_mimic_metadata_d17c38b.json`, `isaac62_urdf_options_d17c38b/receipt.json`
and the original receipt under
`isaac51_urdf_options_d17c38b/eval_videos/isaac51_urdf_options_d17c38b/`.

The original non-cardinal import authors empty axis tokens, a rotX mimic API,
and natural-frequency/damping attributes under a separate `physxMimicJoint:rot:`
prefix. Its C++ `getJointAxis`/`configureMimicAPI` code confirms this behavior.
A successful original conversion alone does not establish how its physics
fallback behaves. Do not replace it with guessed modern semantics or count this
case as preserved. Further original-response and frame/constraint inspection is
required before implementing this path.

Live comparison `d69246e` covers all three successful cases for 240 steps at
1/120 s, preserving imported gains and limits. The probe compares every joint
and body, including the extra tip joint/body, with explicit meters for prismatic
coordinates. All pass the existing body/orientation gates and joint tolerances:

| Fixture | Maximum joint-position difference | Maximum body-position difference |
| --- | --- | --- |
| Prismatic PD | 3.18323e-11 m | 1.49012e-7 m |
| Prismatic natural frequency | 4.65661e-10 m | 1.45286e-7 m |
| Three-joint mimic chain | 8.94070e-8 rad | 3.12924e-7 m |

Every compared joint moves: prismatic excursions exceed 0.23 m, and the chain
moves about 0.737, 0.736 and 0.374 rad across its three joints. Effective gains
and velocity limits match exactly. Both generated imports are simulated in the
target SDK; this tests importer preservation within 6.2, not cross-SDK solver
identity or task performance. The expected missing-explicit-actuator warnings
remain because only the first joint has an actuator configuration. Receipt/log:
`isaac62_urdf_extended_runtime_d69246e/`. All fixture workers, reservations and
account-owned GPU contexts are verified released. Non-cardinal support was still
incomplete at this stage; its repair is recorded below. Broader inertia-frame
coverage remains open.

### Non-cardinal mimic preservation (2026-11-16)

The original SDK 5.1 control at `27b4dc3` establishes usable physics for the
original oblique-axis fixture: 240 imported-gain steps, about 0.78538 rad hinge
excursion, finite joint/body state and exit zero. Both identical original copies
match exactly. Its empty axis tokens therefore do not justify dropping this
path during migration. Original receipt:
`isaac51_urdf_noncardinal_original_27b4dc3/receipt.json`.

Fix `ee1396a` follows the pinned C++ `getJointAxis`/`configureMimicAPI` behavior:
ordered float32 epsilon checks identify nearly cardinal axes; genuinely oblique
axes use rotX in the reoriented joint frame. The modern asset retains its valid
X token rather than copying the original empty token. Original gearing, offset,
reference-axis fallback and expanded limits are retained. The nonstandard
`physxMimicJoint:rot:` frequency/damping prefix is preserved; those values are not
moved into `rotX`, which would activate different constraint settings. The
live receipt confirms rot-prefix values are authored while rotX frequency and
damping remain un-authored schema defaults of zero. Converter cache version 12
invalidates earlier compatibility assets.

The fresh target fixture completes all four extended conversions. Its paired
240-step response test passes oblique mimic plus the prismatic PD, prismatic
natural-frequency and three-joint mimic regressions. Every joint and body is
compared, and all effective gains and velocity limits match exactly. For the
oblique case, maximum joint error is 5.96046e-8 rad, body-position error
2.98023e-7 m and quaternion-component error 3.46452e-7. Evidence:
`isaac62_urdf_options_ee1396a/receipt.json` and
`isaac62_urdf_extended_runtime_ee1396a/receipt.json`.

Comparing the original SDK 5.1 trace directly to the repaired SDK 6.2 trace also
passes the same gates for this one fixture; report:
`noncardinal_cross_sdk_response_ee1396a.json`. This is bounded numerical evidence,
not a claim of general solver identity or task success. Existing unresolved
visual-reference and intentionally incomplete actuator-configuration warnings
remain in the original/paired logs; no mimic/constraint parser error is logged.
All owned workers, reservations and GPU contexts are verified released. Merged,
rotated and non-diagonal mass-frame coverage and other documented migration
requirements remain open.

### Rotated inertia and conditional fixed-link merge preservation (2026-11-16)

The new eight-case fixture measures rotated inertial origins, non-diagonal
tensors, fixed/floating roots, massive fixed payloads and natural-frequency
drives. The first probe (`b357d88`) incorrectly assumed a merge request would
remove a massive fixed child; its failed receipts remain preserved. Corrected
probe `d35ad86` records observed topology and independently checks mass totals.
The original SDK retains the massive payload and fixed joint even when merging
is requested; an additional original empty-link control at `3ddd5e8` establishes
that a truly empty fixed child is removed. Original receipts:
`isaac51_urdf_options_d35ad86/` and `isaac51_urdf_options_3ddd5e8/`.

The unmodified modern merge prepass combined even the massive payload into its
parent, changing body count and fixed attachments. Conservation of aggregate
mass does not preserve those body/contact interfaces. Fix `3ddd5e8` disables
that full SDK prepass and retains the SDK's ordinary empty-link removal. The
existing helper restores empty physical links when the user requests no merge.
Unrotated diagonal mass-frame restoration now also applies to the retained
massive bodies under a merge request. Converter cache version 13 invalidates
previous compatibility assets. No model inputs/outputs or training recipe was
changed, and no SDK package was patched.

Nine fresh target conversions match the original body names, fixed attachment
graph, joint gains, axes and limits. Body masses and centers of mass match
exactly. Rotated/non-diagonal eigenvalue ordering and principal-axis quaternions
can differ while representing the same physical tensor: explicitly reconstructing
R diag(I) R^T gives maximum entry error below 6.2e-9. This was checked rather
than replacing the target diagonalizer unnecessarily. Evidence:
`mass_property_comparison_3ddd5e8.json` and
`isaac62_urdf_options_3ddd5e8/receipt.json`.

All eight paired 240-step imported-gain response tests pass with every joint
and body compared, including each retained payload. Maximum joint-position
error is 1.49012e-7 rad; maximum body-position error is 4.47035e-7 m and
quaternion-component error is below 6.8e-7. Effective gains and velocity limits
match exactly. Both original and target imports run in SDK 6.2; this establishes
bounded importer preservation, not general cross-SDK solver identity or task
success. Receipt/log: `isaac62_urdf_mass_runtime_3ddd5e8/`. The earlier four-case
unmerged response also passes independently under `d35ad86`.

Every worker, reservation and account-owned GPU context is verified released.
A local SSH-client audit found no stale clients. These gates cover the planned
representative merged/rotated/non-diagonal cases; the other migration
requirements and pending scientific/appearance reviews remain open.

### Nested contacts and actual SONIC batching (2026-11-16)

The contributed drawer's real two-environment startup found two shared contact
assumptions that the new URDF hierarchy invalidates: sensor discovery only
looked at direct children, and contact API activation stopped at the first rigid
body. Both now traverse nested links. PhysX list-pattern views use pattern-major
rows, so sensor buffers explicitly map actual prim paths into environment/body
order while retaining the original flat-asset path. The activation routine also
now applies the sleep setting to each current body, rather than the root prim.
At `058f474`, both environments reset and all 50 contact-body positions exactly
match independently indexed articulation positions. Stepping then fails before
physics on the released encoder's fixed ONNX batch dimension; no rendering pass
is claimed for that probe. Evidence: `isaac62_izat_drawer_runtime_058f474/`.

The released encoder and decoder advertise batch one. The encoder additionally
bakes that size into reshapes and its mode-selection mask. A reshape-only fix
failed the independent numerical comparison: different environments overwrote
the same mask row. At `87b8096`, the in-memory ONNX translation retains all
original weights/constants and supplies a separate mask row/index per environment,
alongside inferred reshape batch sizes. Original model files are never changed.
Actual original ORT single-row calls versus translated batches 1, 2 and 7 pass,
including mixed G1/teleop/SMPL modes and reversed row order. Encoder outputs are
exact; decoder maximum absolute error is 2.98023224e-7 against fixed tolerances.
The CPU receipt is local in `check_izat_onnx_parity_modes.json`. This establishes
per-row numerical equivalence for the checked inputs, not a simulation or PPO
pass. The next real runtime at this source reaches first-step reward computation
and exposes the same optional `n_successes` logging-counter assumption already
fixed in the native drawer task. The contributed copy now uses that existing
guard too, with unchanged reward values. The real `79efc07` regression passes
reset, 32 action steps and final reset across two environments. All 43 joint
positions/velocities are finite; three independent comparisons of all 50 contact
body poses have zero maximum error. There are no terminations or truncations in
this bounded fixture, and task success is not assessed. Four RGB frames and the
32-frame, 1280x720, 50-fps MP4 are local in
`out/isaac62_migration/isaac62_izat_drawer_runtime_79efc07/` and indexed in the
existing dashboard without recompression. Owned processes, GPU contexts and the
reservation are released.

The new importer nests the physical `d435_link` below the torso instead of
directly below `Robot`. Contributed recording/rollout camera construction now
resolves that unique mount from the actual robot USD, keeping the original
intrinsics, resolution and local pose. It does not create a disconnected camera
at the old flat path. The subsequent real fixture checked camera world
position/orientation against independently read URDF torso-to-mount transforms
and the original camera offset. Its initial failed evidence is recorded below. An initial standing-button reset at `a26cfa0` first exposed an optional
`last_contacts` diagnostics-buffer assumption. The contributed reset now has the
same presence guard as the already migrated native button task. The replacement
two-environment standing-button reset then reaches a camera-pose failure. The
camera position differs from the independent articulation/URDF expectation by
0.0379229 m. The torso's PhysX and Fabric poses agree, while its authored USD
pose remains at startup. Reading the camera through Fabric, explicitly updating
it at capture, and attaching directly to the physical torso did not resolve
the discrepancy (`af91ee1`). The bounded `128fa64` probe retained reset and
stepped diagnostics and video before applying the same strict final acceptance
check, establishing a persistent mismatch rather than reset-only staleness. Contributed PPO, actual recording, native GR00T
training/serving and both evaluation tracks still require their own runtime gates.

The first mismatch was stale USD pose telemetry. The attempted migration fix
using the deprecated core Fabric view introduced a rendered offset too: its
`_create_fabric_view_indices` calls `SetWorldXformFromUsd` on cameras without
world transforms, writing the stale authored world pose over their live
inherited mount. The rendered offset then remains fixed relative to the torso.
The shared camera reader now queries IFabricHierarchy without authoring a world
transform. The unhelpful rigid-parent attachment workaround was removed; the
original d435 mount and camera-local pose are preserved. The strict real button
regression at `6d2e633` passes reset, 32 steps and final reset across two
environments. All three sensor-pose comparisons pass; independently inverted
renderer matrices agree with the URDF/PhysX mount within 5.99e-8 m position and
2.07e-7 rotation-matrix components. Contact-body comparisons remain exact.
Original videos, frames, telemetry and terminal/resource receipts are local in
`out/isaac62_migration/isaac62_izat_button_runtime_6d2e633/`. This is technical
rendering validation, not task-success or visual-behavior approval. The drawer
regression with the same strict camera check also passes at `aa33234`: three
renderer comparisons are within 8.62e-8 m and 2.02e-7 rotation-matrix components;
all 50 contact-body poses match exactly throughout the fixture. Both tests
release their owned processes, GPU contexts and reservations. A separate
contributed PPO regression at `40932cc` also passes for the drawer: two updates,
checkpoint save, a new runner restoring every model and Adam state exactly, then
a third update advancing optimizer state and changing finite model parameters.
The action width remains 66. This is a same-process/new-runner test, not a fresh
simulator or RNG-continuity test. The earlier `aa33234` fixture failed when its
inspection save accessed a logger not yet initialized by learn(); that test
error is retained and corrected by comparing live restored state directly.
The standing-button PPO regression separately passes the same gates at
`40932cc`. Both workers exit zero, release their reservations and leave no
account-owned GPU contexts. Small receipts/logs are local under
`isaac62_izat_{drawer,button}_ppo_40932cc/`; fixture checkpoints remain remote.

A fresh remote fetch on 2026-11-16 confirms Izat's contribution tip remains
`14c979c`, and ancestry verification confirms it is included in the current
integration branch. No later contribution commit was found.

## Full GR00T save/resume gate (2026-11-15)

The actual full-model validation passed on Bluesclues GPU 4 using committed probe
`107ade9`, the existing GR00T environment, checkpoint 92967 and its matching
walking dataset. All 1,620,515,968 trainable parameters remained enabled with
FP32 weights and AdamW state, BF16 computation and the bounded batch-one fixture.
Update one saved full model, optimizer, scheduler and Trainer state. A fresh
process verified the saved trainable-weight samples and optimizer/global step
one, then reached step two with finite loss and changed weights. Peak allocated
CUDA memory was 38,521,350,144 bytes. Both processes exited zero and the owned
GPU reservation was released. Reports are under local
`out/isaac62_migration/isaac62_gr00t_training_blue_107ade9/`.
This proves bounded optimization/save/resume, not full-study convergence or
bitwise-continuous dataset RNG: the native resume loader explicitly reseeds.
Adroit job 3463916's current state is unknown because its shared connection is
on authentication hold; no further job was submitted there.

## Current validation state (2026-11-15)

The migration is **not complete**. The table below summarizes current evidence;
subsequent dated entries retain the development history, including failed attempts.
Local receipts and original videos are under EMBER's `out/isaac62_migration/`;
`out/reports/dashboard.html` indexes the rendered artifacts.

| Area | Current evidence and remaining work |
| --- | --- |
| Merged project surface | Latest audit has no missing files, declared symbols or CLI options against either recorded parent. This is not behavioral proof. |
| Task configurations | All 56 constructors were checked in both original 5.1 parents and current 6.2: 44 pass and the same 12 fail. No new failures or recorded configuration-metadata differences. The 12 original broken registrations remain explicitly failed. |
| Task runtime and rendering | Representative tracking, static pick, walking, motion-only, drawer and button reset/step/render gates pass. Policy, camera and reference-overlay paths have local artifacts. User appearance review remains pending. |
| Models and optimization | Saved SONIC/residual CUDA parity, GR00T service integration and residual optimizer/save/resume checks pass. Full GR00T optimizer/save/fresh-process resume now passes on Bluesclues. Izat's native training/export and fresh serving/reload gates also pass bounded fixtures; repaired drawer/button PPO restores model and Adam state exactly before a resumed update. |
| Collection and reusable data | Passing expert write, failed student rescue, broad/late takeover labels and finger commands are checked. Actual 577+605-frame HDF5 conversions match full versus sharded/merged data, whole-dataset normalization and official GR00T loading. |
| Reference generation | Actual input export, original holosoma optimization, old/new Adapter-B arrays and full-reference playback pass their stated gates. Original OmniControl sampling completed one 196-frame sample with all 1,000 diffusion steps. Bounded original SMPL conversion and all four HumanML loader modes now pass with user-site imports disabled in the approved environment. Full-stitch exported arrays and optimization traces, other refiners and the three contributed reference generators pass paired numerical gates. User visual review remains open. |
| Known limits | Exact snapshot/replay fails in both original and target SDKs; production rescue uses independent rerolls. Implicit-material stress, removed damping scaling and legacy collision-diagnostic parity remain explicitly unresolved/unsupported rather than silently approximated. |

### Material rejection leaves the USD stage unchanged

A live 6.2 regression test at `877a257` proved that rejecting a non-neutral
`damping_scale` left an empty newly created material in the stage. `efdefca`
validates the removed property before any material is authored. The same live
probe now passes six rejection cases: scales 0, 0.5 and 2, each with a new path
and an existing material. Every rejection leaves the entire stage layer unchanged.
The two differently cooked bodies also pass 20 physics steps, state/kinematic
writes, finite diagnostics, valid quaternion checks and independent affine
stress/gradient checks. Supported material parameters and the legacy API path
are unchanged. Both the original failing and corrected passing receipts are
preserved in `isaac62_deformable_batch_877a257/` and
`isaac62_deformable_batch_efdefca/`. Resources were released after both jobs.

NVIDIA's [migration guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/deformable_migration.html)
lists no replacement for damping scaling. Its
[material schema description](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/omniphysics_deformable_schema.html)
also explains that Young's modulus has no physical default. These sources do not
justify guessing an unbound solver material or replacing removed damping behavior.

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

### Full constructor audit, new reference render and Adroit training gate (2026-11-12)

The registry-wide audit discovers56motion-task configurations. Actual isolated
exports of both recorded parents in Isaac5.1 and the current merged source in6.2
produce the same44passing constructors and12failing constructors. No constructor
failure was added, and the recorded class, physicsdt, decimation, default environment
count and agent-config entry metadata have no differences between the three runs.
This is constructor coverage, not runtime/success coverage for all56tasks.

The12failures are preexisting in both parents: missing modules for bimanual-pick-drop,
hit-target, kick-real, dishwasher and stand-still; missing PlayCfg classes for pick,
pick-real, pick-place, drawer and punch; and both sit variants reference the absent
both_feet_in_air_penalty helper. They remain explicitly failed, with registrations
retained. No task or reward replacement was invented. Source provenance and actual
receipts/comparison are local in `task_config_failure_provenance.json` and
`isaac62_task_config_baselines_5930fe3/` under the migration output root.

The baseline harness first omitted the original application kit files; its missing
experience failure is preserved. Subsequent construction completed, but SDK5.1
rejects the6.2 close(exit_code=...) API and its slow shutdown also segfaulted.
The final harness uses each SDK's supported fast-shutdown contract and persists
failure receipts first. SDK5.1's zero fast-shutdown exit is explicitly not treated
as a passing result. Final receipts retain all12failures in each source; the
comparison establishes no new constructor failures, not that every constructor passes.

The newly optimized original holosoma output is now also replayed in6.2 through the
production reference-playback entrypoint. It renders512frames at50fps,2560x1920,
covering0.54..10.76seconds of the9.75-second reference. Finite trace and exact
kinematic-reference checks pass; video and frame are pulled locally to
`isaac62_new_retarget_replay_2d02dfa/` and indexed in the dashboard. This is reference
playback with overlays, not a learned-policy rollout or an appearance judgment.

Adroit access was restored by the user. The standard login-node HTTP proxy was
absent and has been restored in the own-account conn tmux session. Full GR00T
optimizer/save/fresh-process resume validation is queued as Slurm3463916, from
exported commitb6b38aa, on one A10080GB with4CPUs/96GBRAM and25minutes wall time.
It uses the existing saved92967 checkpoint, matching existing walking LeRobot data
and unchanged GR00T environment. No package installations or full research runs are
involved. The job owns and cleans its compute-to-login proxy tunnel. Authoritative
state is PENDING(Priority); the earlier test-only start forecast was November18,
not a guaranteed ETA. No GR00T optimizer/resume pass is claimed until receipts prove it.

The upstream OmniControl checkpoint was found under Biped's existing locomotion
worktree. A bounded generation check is awaiting approval to use the existing
omnicontrol environment under the workspace environment-switch rule. No model was
replaced and that environment has not been executed during this follow-up.
