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

### Mechanics: snapshot, branch, stitch

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
