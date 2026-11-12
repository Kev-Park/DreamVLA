# Data and evaluation protocol

## Timing and tensor interface

| Interface | Shape / rate | Meaning |
| --- | --- | --- |
| Physics | 200 Hz | Four simulation steps per control step |
| SONIC controller | 50 Hz | Frozen encoder/decoder, ten-step proprioceptive history |
| PPO expert action | 66 | 64 residual channels and two hand primitives |
| Full body token | 64 | FSQ lattice token, **not** the PPO residual |
| VLA action | 78 | Full token + seven left-hand + seven right-hand commanded joint targets |
| VLA state | 46 | 43 measured G1 joint positions + three projected-gravity components |
| VLA camera | 480 × 640 × 3 RGB | Single robot-mounted camera, no overlay or reference markers |
| GR00T prediction | 40 × 78 | Forty 50 Hz action steps (0.8 seconds) |
| Execution/replanning | 20 steps / 2.5 Hz | Execute the first 0.4 seconds, then replan |

State joint order is left leg [0:6], right leg [6:12], waist [12:15], left arm
[15:22], left hand [22:29], right arm [29:36], right hand [36:43]. This is a
feature-model order, not necessarily Isaac's internal joint order. The recorder
looks up names and rejects mismatches. The SONIC decoder has its own explicit
29-body-joint order in `isaac/sonic_action.py`.

The expert uses privileged reference/object observations. The VLA receives only
RGB, measured joints, gravity and language. EEF state, root orientations, debug
arrays and object positions may be retained for audit, but are not in the VLA
modality configuration. Hand labels are commands, not measured joint positions.

The default expert composition is

```text
token = clip(round(16 * reference * (1 + 0.5 * residual)) / 16, -1, 15/16)
```

This is unbounded multiplicative modulation before token clipping, without tanh.
GR00T predicts the resulting **full** token; evaluation must not multiply it by
the reference a second time. Continuous predictions are snapped once before
decoder execution. Hand outputs are projected to the nearest task open/closed
primitive. A zero residual preserves the reference token, not a zero motion.

## Raw recordings and LeRobot

Each accepted demonstration contains `data.npz`, `metadata.json` and
`ego_view.mp4` (plus a first-frame PNG for inspection). Arrays are numerical,
loaded with `allow_pickle=False`. Metadata identifies the teacher, reference,
scene, reset seed, timing and task outcome. No generated recording is committed.

Successful native episodes execute 500 controls over ten seconds. There are
499 saved pre-action observations: the final action triggers an automatic reset,
so that terminal boundary is intentionally not mislabeled as a teacher action
for the next episode. Video is 50 Hz, captured independently of the 2.5 Hz VLA
inference schedule. Repeating sparse inference frames is not smooth video.

Conversion uses SONIC's official feature/modality exporter and creates LeRobot
v2.1 metadata, Parquet tables, per-episode MP4 video and statistics. It checks
scene equality, accepted flags, label dimensions, finite values, FSQ membership,
contiguous control steps, frame counts and input content hashes. An explicit
synchronous PyAV writer is installed only in the converter process to ensure
encoding is flushed before an episode is finalized; dependency source files are
not patched. Optional teleoperation fields are schema fillers, not supervision.

Masked DAgger suffix data is rejected by the generic converter/trainer. Silently
training on student-prefix actions as if they were expert corrections is invalid.

## Frame epochs versus action-window passes

For frame lengths `L_i`, horizon `H=40`, and effective optimizer batch `B`:

```text
F = sum(L_i)                         # recorded observations
W = sum(L_i - H + 1)                 # complete action-chunk starts, no end padding
S = ceil(epochs * F)                 # requested sample exposure
U = ceil(S / B)                      # optimizer updates
window passes = S / W
```

Thus 200 complete demonstrations at 499 frames and batch 16 give 35,554 updates
for 5.7 frame-based epochs. This is not 5.7 passes over a cache containing only
23 sampled starts per episode. `exposure.py` provides an exact partial-batch
sampler for custom cached-head trainers; the native online launcher uses a
nominal update budget and NVIDIA's sampler, including its padding/sharding
behavior. Its last full optimizer batch can overshoot the target by <B samples.
Do not claim bitwise-equal sample visitation across those two training paths.

## Metrics and evaluation tracks

Evaluation uses an immutable bank of reference names and reset seeds. Training
and held-out reference contents must not overlap. Both the expert and student
can be evaluated against the same bank; DAgger development cases must be kept
separate from an untouched final reporting bank. A rollout is an attempt, not an
accepted training demonstration. Failed eval rollouts are never replaced.

Drawer `accepted` is the legacy criterion: no early termination, upright, and
bottom-drawer displacement >0.05 m for the last 50 controls. `paper_success`
requires >0.15 m over that same final second. Record both and explicitly name
which numerator is reported. These are thresholds on simulated joint position,
not visual human judgments. Do not call the 5 cm acceptance rate the 15 cm score.

Standing-button success retains DreamControl's reference-time-gated fingertip
touch objective, completes the episode, and limits each ankle's XY drift and
upward lift to 5 cm after the first 0.5 seconds of settling. It is not a force,
contact-sensor, or physical button-depression task. The panel animation is visual
only and does not change reward or success.

`repeatability` uses seeded resets; standing-button XY/yaw perturbations are
disabled. `fresh_start` uses independent seeds and the standing task's existing
±1 cm XY / ±2° yaw perturbations. **For drawer the extracted recorder currently
only changes reference/reset seeds, not an additional XY/yaw distribution.** It
must not be reported as a separate pose-randomization benchmark. Adding such a
benchmark requires a versioned reset contract and new validation, not relabeling
historical results. This limitation is intentionally visible in the release.

All evaluation observation corruption is disabled. This does not disable PPO
training corruption, physics/reset randomization, or GR00T sampling noise.
Use 100 attempts per track for final reporting, report raw successes/attempts,
and retain per-case outcomes. Use short banks for integration smoke tests only.

MPJPE is not inferred from action MSE or token cosine similarity. Any future
tracking-error report must specify joint set, coordinate frame/alignment,
reference sampling, episode weighting and meters-to-millimeters conversion.
This package does not carry historical MPJPE results or generate an unspecified
cross-task average.
