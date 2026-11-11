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

`cluster/audit_capability_union.py` checks the project file, top-level symbol and
CLI-option union against both recorded parents. It deliberately does not claim
behavioral equivalence. Runtime, visual and optimization evidence is still required.

Initial merge audit artifacts and test receipts are under the EMBER workspace's
`out/isaac62_migration/`. The 6.2 runtime is not yet validated. Do not resume the
existing full DAgger study or mark this migration complete from static checks.
