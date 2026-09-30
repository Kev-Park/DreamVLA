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
- Success/failure uses the **4-rule criterion with the hand-calibrated palm box**, with the
  calibration chosen *explicitly* per config (the pre-audit box rejects 100% of post-audit
  episodes, which would make every rollout read as a failure and trigger a rescue every time).

## TO REVIEW

1. **Reference origin.** Defaulted to the **VLA fine-tuning reference set**. Alternatives: the
   residual's own training set, or freshly sampled references. Note the expert degrades off its
   training conditions (measured: 50.0% vs the student's 51.9% when run at physics it never trained
   on), so novel references would give degraded labels.
2. **Cross-iteration aggregation.** Classic DAgger grows D across iterations, and this pipeline has
   done that implicitly (base + dagger1 + dagger2 -> 250 episodes). Confirm the next fine-tune
   trains on af60v2 demos **+** the rescued set rather than the rescued set alone.
3. **`expert_base` for the rescue segment.** Defaulted to **student** (residual composed onto the
   VLA's token) per instruction. Measured caution: in the three-way test that arm scored **5/54 =
   9.3%** against the student's own 51.9%, while native mode (residual on the reference base token)
   scored 50.0%. If rescue success comes back near 9%, switch to `reference`.
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

### Efficacy: the rescue produced NO successes -- and the cause is upstream

| arm | pass A (student) | rescue |
|---|---|---|
| `expert_base=student` | 0/2 pass | **0/2 pass** |
| `expert_base=reference` | 0/2 pass | **0/2 pass** |

So `expert_base` is NOT the explanation (to-review item 3 is answered negatively for both settings at
this n). The diagnosis is the one recorded above under label validity: the residual is **pi*(s, t)**,
a reference tracker whose base token is indexed by motion time. Handing over at k=151 of grab_idx=178
puts the reference clock 85% of the way to the grasp while the student -- which scores **0/60** on
eval -- has already drifted far off it. The expert cannot recover a phase it is no longer aligned
with, and its own competence gate (`ee_body_pos`, 0.25 m) marks exactly that boundary.

**The rescue design presupposes a student that is sometimes right.** With pass A failing every time,
there is no successful student behaviour to preserve and the expert must recover from an arbitrarily
drifted state. The blocker is therefore NOT the DAgger design but the student: the af60v2 GR00T
fine-tune scored 0/60 (see `results/vla_eval/README.md`). Fixing that comes first; the rescue
machinery is ready and waiting behind it.

Two follow-ups this suggests, beyond the existing list:

9. **Bias `k` earlier.** Uniform sampling over `[0, grab_idx)` lands late often (two of two draws at
   85%), and late handover is where recovery is least possible. An earlier-weighted distribution
   would hand over while the student is still near the reference.
10. **Gate the collection on the student.** Refuse to collect if the student's success rate on the
    reference set is ~0: the run can only produce failures, which the filter then discards.
