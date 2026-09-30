# DAgger rework (2026-09-30)

Replaces the stochastic-beta / disagreement-trigger arbitration with **failure-conditioned expert
rescue**. Both older modes stay in the code, flag-gated and off, so the arms remain comparable.

## Design (agreed)

1. **One rollout ↔ one reference motion.** The student is expected to match the residual's behaviour
   under that reference frame by frame.
2. **Student starts from the reference initialisation**, honouring `--skip-start`, so student and
   reference are frame-aligned.
3. **Labels come from the residual at every visited state, including the grasp.** The palm-box
   grasp gate is removed from the label path entirely — grasp supervision reaches the residual
   through the reference (`is_closed` in its observations) and leaves via its own finger head.
4. **The student drives by default.** Only if the student's rollout FAILS does the expert take over,
   from a frame `k ~ Uniform[episode_start, grab_idx)`, as a stand-in for "where the student began
   drifting off-distribution".

### Mechanics: two independent passes (MEASURED 2026-09-30)

The environment does **not** reproduce. Two identical rollouts -- same motion, same seed, full reset
each time, expert-driven with no VLA involved -- diverge immediately:

```
joint_pos   max 2.940e-02   first frame over 1e-6: 0
root_pos    max 1.910e-03   first frame over 1e-6: 0
object_pos  max 1.405e-03   first frame over 1e-6: 104
growth: f0=5.11e-04  f4=7.61e-03  f19=2.78e-02  f119=4.87e-03
```

Frame-0 divergence of 5.11e-04 rad is too large for one step of float noise, so something differs at
reset beyond what `_set_all_seeds` covers. It is bounded rather than chaotic -- peaking near 1.7 deg
and pulled back by the tracking controller -- so rollouts are *similar*, just not identical.

Therefore **exact reproduction of a failed student rollout is impossible**, and both earlier plans
are dead: the snapshot/stitch (which also failed its own gate at 0.48 rad) and the deterministic
re-roll (seeding the VLA cannot help when the env alone will not reproduce).

**What is implemented instead:** on pass-A failure, run an INDEPENDENT pass B where the student
drives to `k` and the expert takes over. Pass B's student segment is a sibling sample, not a
reproduction. This is sound for DAgger, which requires states drawn from the learner's distribution
rather than one specific trajectory -- and at ~1.7 deg peak deviation the sibling is a close one.

Cost: two full rollouts per failed episode. No snapshot machinery, no determinism requirement.

`--dagger-selftest-determinism` and `--dagger-selftest-restore` are kept as the record of why.

### Superseded: snapshot, branch, stitch

`k` is sampled *before* the rollout and the sim state snapshotted there, so failure detection costs
one full student rollout and the rescue only a partial one (`k -> end`) rather than two full passes.
It also avoids replaying the student segment, which would be irreproducible: GR00T's flow-matching
head samples from `torch.randn`, so a replayed student rollout would diverge from the one that
actually failed.

The stitched record is frames `[0,k)` from the student pass + `[k,end)` from the rescue pass.

Snapshot must carry: sim state (robot root pose/vel, joint pos/vel, object root pose/vel),
`episode_length_buf`, and a deep copy of the decoder `HistoryBuffer`. `_base_token` is recomputable
from motion time. **The restore must be proven bit-exact before the stitch is trusted** — otherwise
it silently fabricates a discontinuity at `k`.

### What each half of a stitched episode contributes

| segment | states from | labels from | role |
|---|---|---|---|
| `[0, k)` | student | residual at those states | **corrective** — on-policy DAgger |
| `[k, end)` | expert | residual | **demonstration** — plain BC |

Corrective content therefore scales with `k`: a rescue at k≈0 is nearly pure demonstration, one near
`grab_idx` is mostly corrective.

### Settled

- On failure, keep **only** the stitched (rescued) episode. Keeping the failed student pass would
  feed the filter something it rejects anyway.
- `k` upper bound is `grab_idx`.
- **Reference origin: sample from the pre-existing references used in the fine-tune.** Decided by the
  user 2026-09-30. `--dagger-demo-root <fine-tune demo set>` restricts the sweep to those motion ids
  and pairs every student rollout (and its rescue) with that clip, which is what the residual then
  supervises against. No fresh references, no residual-training-set references.
- **Expert labels come from the residual acting on the paired reference** (`expert_base=reference`),
  not on the student's base token. Decided by the user 2026-09-30. The residual is pi*(s, t) for
  the reference clip the episode is paired with; the student's token never enters the label path.
- **Scope of validation: plumbing, expert-agnostic.** The rework is validated when the collect ->
  4-rule score -> rescue -> write -> filter/convert path behaves correctly, not by the rescue success
  rate of any particular expert (af60v2 is only usable under its own physics, see below).
- Success/failure uses the **4-rule criterion with the hand-calibrated palm box**, with the
  calibration chosen *explicitly* per config (the pre-audit box rejects 100% of post-audit
  episodes, which would make every rollout read as a failure and trigger a rescue every time).

## TO REVIEW

1. ~~Reference origin~~ -- **settled 2026-09-30, see above** (sample from the references already used
   in the fine-tune; this is what `--dagger-demo-root` does).
2. **Cross-iteration aggregation.** Classic DAgger grows D across iterations, and this pipeline has
   done that implicitly (base + dagger1 + dagger2 -> 250 episodes). Confirm the next fine-tune
   trains on af60v2 demos **+** the rescued set rather than the rescued set alone.
3. ~~`expert_base` for the rescue segment~~ -- **settled 2026-09-30, see above** (the residual acts
   on the paired reference; `--dagger-expert-base reference` is the default and the only supported
   configuration for the rework; the `student` value stays only as an A/B knob).
4. **`k` upper bound.** Fixed at `grab_idx` for now. Revisit once the failure distribution is known:
   the observed modes are dominated by slipping and toppling, which occur at or after the grasp, so
   a pre-grab-only takeover never demonstrates rescuing the lift.
5. **Uniform `k` in frames vs normalised phase.** Uniform over frames biases toward clips with long
   approaches, since `grab_idx` varies per motion.
6. **The palm box is the last hand-calibrated component.** It is now per-config
   (`BOX_*` vs `BOX_*_AF60`) and unvalidated against hand labels in its new form. Candidate
   replacements: auto-fit from each task's demo hold phase, or derive the closing volume from the
   gripper's kinematics.
7. **`HS_HAND_DAMPING`.** Left at 1.25 while stiffness went 5 -> 15, which drops the damping ratio
   by sqrt(3). Harmless at MID (still ~5x more damped than the `unitree.py` asset default) but
   should scale if stiffness goes higher.
8. **`mid_expert` retrain** (deferred). Retrain the residual at deployment physics
   (`HS_OBJ_MASS=0.603 HS_HAND_STIFFNESS=15 HS_HAND_EFFORT=10 HS_HAND_VELOCITY=3`). The current
   expert trained at 0.1 kg / 5-3-1 and is run at neither.

## BUILT AND VALIDATED (2026-09-30)

`--dagger-rescue` is implemented in `collect_sonic_adapter.py` and exercised on real rollouts.
Mechanism validated:

| behaviour | evidence |
|---|---|
| preconditions guarded | `forcing --dagger-beta 0`; `--dagger-rescue-box` required; grasp gate forbidden |
| `grab_idx` from pass A itself | 178 and 266 on two motions; no extra reset needed |
| `k ~ U[0, grab_idx)` | k=151/178 and 227/266; sampling verified to vary (85/47/85/89/40/24/77/13% across motions) |
| 4-rule verdict drives the branch | pass A FAIL -> pass B every time |
| one-way takeover at k | expert drove 348/499 and 272/499, consistent with the sampled k |
| pass A discarded, pass B kept | only the final filenames remain on disk |

### Efficacy: every rescue failed -- because the env changed under the expert (RESOLVED 2026-09-30)

All rescue arms on 09-30 (smoke, blend 15, early-k, k=0, k=0 without the VLA query) produced 1/25
successes. Two earlier drafts of this section blamed phase drift, then a handover lurch. Both were
wrong: the discriminating run was **expert from frame 0 through the DAgger path = 1/8**, against a
paired **15/16 for the same expert, same motions, same flags, in pure-demo mode** -- and the failing
episodes track the reference identically (root deviation 2.8-4.3 cm at f100 in both) right up to the
grasp, where the bottle is knocked off instead of lifted.

The measured right-hand joints differ: demos close to ~0.4 rad at grasp+20 and ~1.0 rad at +80; every
rescue run hits 1.2 rad by +20. Same command sign, different hand dynamics. Cause:

| commit (09-29 evening) | change | af60v2 trained/collected with |
|---|---|---|
| `b78cacb` | `OBJ_MASS` 0.1 -> **0.603 kg**; hand effort/velocity/stiffness 3/1/5 -> **10/3/15** | 0.1 kg, 3/1/5 |
| `c27441e` | object `max_angular_velocity` USD clamp (100 deg/s) -> **1000** | 100 |
| `0b232d5` | startup mass draw `add(0.0,0.4)` -> `add(-0.2,0.2)` about `OBJ_MASS` | U[0.1,0.5] kg |

`isaaclab_tasks` loads from the main checkout `~/kevin/DreamVLA`; the fast-forward at 02:35 UTC 09-30
(pulling the collector work) brought these defaults in. The af60v2 demos (09-29 13:42-18:49 UTC) and
the af60v2 GR00T eval (22:06-22:34 UTC) both predate it and are valid; every rescue run postdates it and
drove an expert trained at 0.1 kg / 3-1-5 against a 6x heavier bottle with a 3x faster hand. Pass A
(student 0%), `expert_base` (0/N both), the "lurch" and the blend result were all measured under this
mismatch and say nothing about the design.

Not a code defect in the rescue path: the per-step expert path (`dagger.step` -> `env.step(latent)`)
is the pure-demo path; `expert_token` is pure; the decoder history is re-seeded from motion_lib on
every reset; `st["action"][64]` edits a CPU copy. The blend is also not degenerate: it removed the
handover jerk (speed spike 3.5-4.6x -> 1.0-1.3x) with 4-7 cm tracking error throughout.

**Re-run under the expert's physics** (`rescue_op_*`, 09-30 18:52 UTC): the A/B env vars restore it
exactly -- `HS_HAND_STIFFNESS=5 HS_HAND_EFFORT=3 HS_HAND_VELOCITY=1 HS_OBJ_MAX_ANGVEL=100
HS_OBJ_MASS=0.3` (add(-0.2,0.2) about 0.3 = the old U[0.1,0.5] draw). Arms: k=0 machinery check;
k~U[0,grab_idx) hard switch on reference base; same on student base; same + 15-frame blend. Results in
`results/vla_eval/README.md`: k=0 3/5, hard switch 5/8, blend 6/8; 11 rescued episodes converted to
LeRobot cleanly. **Plumbing validated end to end, expert-agnostic.**

Follow-ups this adds to the list:

9. **Bias `k` earlier.** Uniform sampling over `[0, grab_idx)` lands late often (two of two draws at
   85%), and late handover is where recovery is least possible. An earlier-weighted distribution
   would hand over while the student is still near the reference.
10. **Gate the collection on the student.** Refuse to collect if the student's success rate on the
    reference set is ~0: the run can only produce failures, which the filter then discards.
11. **Pin the physics to the expert.** A residual checkpoint is only valid under the env defaults it
    was trained with. Until `mid_expert` (0.603 kg / 10-3-15 / clamp 1000) exists, every af60v2 run
    -- demo collection, rescue, GR00T eval -- must set the five env vars above, and the collector
    should record the effective hand gains / mass / clamp in `metadata_json` so a mismatch is visible
    in the data rather than found by bisection.
