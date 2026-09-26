# VLA eval results store

Persistent, version-controlled home for pick-task eval numbers.

## Vocabulary (2026-09-21)

| name | what it is |
|---|---|
| `base` | 111 episodes: the original demos + the old dagger1/dagger2 rollouts, filtered by the 4-rule criterion (270 -> 111) |
| `dagger1` | the FIRST true-DAgger dataset: 108 rollouts collected off `base_run01`, of which 64 pass the 4 rules |
| `base_dagger1` | 175 episodes = `base` + the 64 kept `dagger1` rollouts |
| `base_run01`, `base_dagger1_run01` | the GR00T checkpoints fine-tuned on those two datasets |

The pre-rename names (`clean_v1`, `clean_dagger3`, `clean_v2`) are gone from disk; the old
`dagger1`/`dagger2` collections referred to here are the pre-criterion ones, kept only inside `base`.

## Success criterion

`vla_sonic/grasp_success.py` -- four rules, validated against 36 human-labelled episodes
(FP=0, FN=0) scored on trajectory dumps matched to the rendered rollout:

1. `fallen_height`      object drops below half the table height at any point
2. `in_box_at_end`      object inside the palm-frame box at the final frame
3. `fallen_horizontal`  object within 10 deg of horizontal at any point
4. `slipping`           palm-frame vertical excursion > 2 mm over the last second

**Do not compare these numbers to anything measured before 2026-09-20.** The old `[PHYS] held`
test counted lift-then-drop-on-the-floor as success and is wrong in both directions.

## Runs (4-rule criterion, 20 episodes, same 4-way motion split, HS_EVAL_NO_EE_TERM=1)

| run | training data | train_loss | success |
|---|---|---|---|
| `base_run01` | 111 eps / 55,389 frames | 0.1269 | **5/20 = 25%** |
| `base_dagger1_run01` | 175 eps / 87,325 frames | 0.1292 | **9/20 = 45%** |

### Failure taxonomy -- why another DAgger round is not worth running

| failure mode | base | base_dagger1 |
|---|---|---|
| KNOCKED (+slip+notbox) | 6 | 6 |
| FELL+KNOCKED (+slip+notbox) | 4 | 4 |
| slip only | 4 | 1 |
| slip+notbox | 1 | 0 |

DAgger removed 4 of the 5 non-knockover failures and left the 10 knockovers EXACTLY unchanged.
One episode of headroom remains, so the ceiling without fixing knockovers is ~50%. Knockovers are
an approach problem (the hand disturbs the bottle before closing) and the expert's corrective
labels for them are already in the data -- more aggregation cannot fix it.

### DAgger yield per round (fraction of collected rollouts passing the 4 rules)

| round | yield |
|---|---|
| old dagger1 | 8/108 (7%) |
| old dagger2 | 55/108 (51%) |
| `dagger1` (true DAgger) | 64/108 (59%) |

## Training-data composition of `base` (filtered from 270)

| source | kept | fell | horizontal | slipping | not in box |
|---|---|---|---|---|---|
| demos | 48/54 (89%) | 3 | 3 | 6 | 4 |
| old dagger1 | 8/108 (7%) | 54 | 82 | 99 | 94 |
| old dagger2 | 55/108 (51%) | 33 | 43 | 50 | 49 |

## Why this store exists

An earlier "~45% held" figure could not be reconciled with a re-run of the same checkpoint,
because the log that produced it had been deleted from the cluster's `/tmp`. Record runs here so
provenance survives.

## Generalisation eval (object spawn jitter)

The bottle spawns at exactly (2.100, 0.000) in all 54 reference motions (sd 0.000), so all
existing variation is robot placement. `HS_OBJ_JITTER_X/_Y` jitters it. Scored with the 4-rule
criterion, MID hand config + 0.603 kg:

| arm | success | 95% CI | vs reference |
|---|---|---|---|
| no jitter | 52/60 = 86.7% | [75.8, 93.1] | — |
| ±3 cm x / ±5 cm y | 46/60 = 76.7% | [64.6, 85.6] | p=0.157 ns |
| ±6 cm x / ±10 cm y | 33/60 = 55.0% | [42.5, 66.9] | p=0.0001 |

Pooled over 180 episodes the logistic slope on displacement is −25.8/m (p<0.0001): odds of
success ×0.77 per additional cm.

**Gotcha that cost four relaunches:** the editable install resolves `isaaclab_tasks` to
`~/kevin/DreamVLA`, NOT the worktree you `cd` into, and `PYTHONPATH` does not survive Isaac's
`SimulationApp`. Env-package changes must be pulled into the main checkout.

## DAgger intervention signals (beta=0 rollouts, 81 episodes)

Testing whether anything cheaper and more general than a hand-calibrated gate can decide when
the expert takes over. All four signals are recorded by `--dagger-diagnostics`.

| signal | jitter vs nominal | predicts failure | verdict |
|---|---|---|---|
| FSQ lattice residual | AUC 0.49 | 0.67 mean / 0.57 p90 | **dead** — aliased noise |
| feature density (kNN cosine) | AUC 0.46 | 0.43 | **dead** |
| feature density (Mahalanobis, LOO) | AUC 0.50 | 0.57 | **dead** |
| root tracking deviation | AUC 0.50 | 0.49 (inverted at p90) | **dead** |
| expert/VLA token disagreement | AUC 0.59 | **0.74 p90** | survivor |

- FSQ residual fails because the grid step is 1/16 but per-channel prediction error is ~0.099,
  so the residual aliases: 14.6% of frames sit at the 1/32 ceiling, dynamic range 1.6x.
- Feature density scored AUC 0.935 until the covariance was fitted leave-one-EPISODE-out, after
  which it is 0.50. The pooled VLM embedding cannot see a 6-10 cm object shift.
- Root deviation stays bounded (median 7.8 cm, p99 18 cm) because SONIC holds the reference —
  failures are in the hand/object interaction, not whole-body tracking.
- Disagreement fires 3.4 s before termination in 22/25 early-terminated episodes.

Two design facts found by measurement, both in `collect_sonic_adapter.py`:
1. A fixed threshold cannot be calibrated offline — thresholding at the beta=0 median spent only
   9.4% of frames on the expert once it drove, because intervening restores agreement. Hence
   `--dagger-trigger-budget`, which targets the budget and self-calibrates the threshold.
2. Committing the decision every frame gave 79 control switches/episode vs ~31 for stochastic
   beta; decisions now commit at re-plan boundaries so both arms switch at the same rate.
