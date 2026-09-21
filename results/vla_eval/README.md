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
