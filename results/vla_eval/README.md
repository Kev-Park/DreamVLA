# VLA eval results store

Persistent, version-controlled home for `eval_vla_sonic.py` numbers. Populate it with:

```bash
python Training/scripts/reinforcement_learning/rsl_rl/summarize_vla_eval.py \
    --tag <run-name> --logs out/<run-name>_eval.log --notes "<what makes it comparable>"
```

* `summary.csv` — one row per run (rewritten in place when a tag is re-summarised)
* `<tag>_episodes.csv` — one row per episode: motion, held flags, max lift, topple, grasp timing

Videos stay out of git: they live in `~/kevin/eval_videos` on the cluster and `out/` locally.

## Read `held5_upright`, not `held`

`eval_vla_sonic.py`'s headline `[PHYS] held` counts an episode as a success when the object is
lifted and stays near the palm **even if it toppled over**. `held5_upright` additionally requires
the object to have stayed upright, and is the number to quote.

## Why this store exists

An earlier "~45% held" figure for dagger1 could not be reconciled with a re-run of the same
checkpoint (25% / 15%), because the log that produced it had been deleted from the cluster's
`/tmp`. The surviving evidence pointed at it having come from a **DAgger collection** log rather
than an autonomous eval — in collection the residual expert drives roughly half the timesteps
(`expert 232 / vla 267`) and the grasp gate fires the close, so those held rates (42–81%) are not
comparable to a pure-VLA eval. Record runs here so provenance survives.

## Runs

| tag | held2 | held5 | **held5_upright** | toppled | notes |
|---|---|---|---|---|---|
| `dagger1_base` | 5/20 | 3/20 | **3/20** | 12/20 | pure VLA, 20 eps, `HS_EVAL_NO_EE_TERM=1`, 4-way split |
| `dagger2_adroit` | 11/20 | 9/20 | **6/20** | 12/20 | same settings, A100 fine-tune |

Both used the same 54 filtered demo motions and the same 4-way interleaved split, so they are
directly comparable.

## Training-data topple rates (2026-09-20)

Measured over the 270-episode aggregate that produced `latband60_dagger2_adroit`:

| source | toppled | total | |
|---|---|---|---|
| demos | 10 | 54 | 19% |
| dagger1 | 90 | 108 | 83% |
| dagger2 | 52 | 108 | 48% |
| **all** | **152** | **270** | **56%** |

The eval topple rate (12/20 = 60%) tracks the training-data rate (56%), which is expected:
`collect_sonic_adapter.py --reject-topple` defaults to **False** ("keeps toppled-but-held grasps"),
and at `collect_sonic_adapter.py:1063` the topple check is gated on `dagger is None`, so it is
skipped for DAgger rollouts even when the flag is passed.
