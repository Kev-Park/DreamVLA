# Faster HDF5 -> LeRobot conversion (sharded)

`convert_isaac_hdf5_to_lerobot.py` is strictly single-process: one `exporter.add_frame()` /
`save_episode()` loop, no `--num-workers`, no `Pool`, no image-writer threads. Measured cost:

| episodes | wall clock |
|---|---|
| 111 | 59 min |
| 175 | 78 min |

~0.5 min/episode, on a box with **104 CPUs** that sit idle throughout. LeRobot 0.1.0 also defaults
to **`libsvtav1` at CRF 30**; AV1 is far slower than H.264 and is the likely inner bottleneck.

## The approach: shard the conversion, DON'T merge the datasets

GR00T already accepts multiple dataset roots. In `gr00t/experiment/launch_finetune.py`:

```python
"datasets": [{"dataset_paths": [ft_config.dataset_path], "mix_ratio": 1.0, ...}]
```

`dataset_paths` is a **list** -- the CLI just wraps a single `--dataset-path` into it. So the plan is:

1. split the filtered HDF5 set into N disjoint directories (symlinks are fine),
2. run N converters in parallel, producing N independent LeRobot datasets,
3. train with all N paths in `dataset_paths`.

Expected ~N x speedup (8 shards: 78 min -> ~10 min).

### Why NOT merge the shard datasets into one

Merging looks easy because `meta/episodes_stats.jsonl` holds per-episode stats, so episodes and
their stats can be concatenated and renumbered. It is a trap:

* `meta/stats.json` carries **`q01` and `q99`**, but per-episode stats only carry
  `count/max/mean/min/std`. **Percentiles cannot be aggregated from per-episode summaries.**
* GR00T normalises with **q01/q99 percentile clipping**, so a naively merged dataset gets silently
  wrong normalisation statistics -- worse than a slow conversion, because nothing errors.

Recomputing q01/q99 from the merged parquet is possible (numeric columns only, no video, so it is
cheap), but the multi-path route avoids the problem entirely and touches no dataset internals.

## BEFORE USING THIS: verify how GR00T combines stats across datasets

**Unverified as of 2026-09-21.** If GR00T computes normalisation **per dataset** rather than over
the union, then sharding changes the input scaling and any run using it is NOT comparable to a run
trained on a single dataset. That would confound exactly the kind of A/B these rounds exist to make.

To check: find where the data loader builds its statistics (the module path differs from upstream in
the `Kev-Park/Isaac-GR00T` fork -- `grep -rn "q01\|statistics" gr00t/` and follow `dataset_paths`),
and confirm the stats are computed over the concatenation. Until that is confirmed, use sharding
only for runs where a fresh baseline is also being measured.

## Remaining piece

The CLI exposes only `--dataset-path` (singular). Either add a `--dataset-paths` option to
`launch_finetune.py` in the fork (local edit -> push -> pull, as for every repo), or drive the
config directly from a wrapper that sets `data.datasets[0].dataset_paths` to the shard list.

## Not recommended: changing the codec

Switching `libsvtav1` -> `h264` would give ~3-5x on its own, but it changes the pixel data the VLA
trains on. `base` and `base_dagger1` were encoded with AV1, so an h264 successor would not be
comparable. Leave the codec alone unless re-encoding everything.


# Parallelising collection and eval

Both `collect_sonic_adapter.py` and `eval_vla_sonic.py` are already sharded by `--motion-range` /
`--motions-from` across 4 GPUs. Measured on bluesclues (2026-09-21, 4 collectors + 4 eval workers
sharing the box):

| per Isaac process | value |
|---|---|
| CPU | ~390% (about 4 cores) |
| RSS | 13.5-15.2 GB |
| GPU | ~14.8 GB of 49 GB |
| throughput | ~8.8 min per 500-step rollout |

Machine: 104 cores, 1007 GB RAM, load average 61 with 8 Isaac processes.

**Cheap win -- more shards.** CPU is the binding constraint, not GPU or RAM: ~4 cores each caps the
box at roughly 13 concurrent Isaac processes. One shard per GPU (8-10 shards) is the practical
setting and takes a 120-rollout round from ~3.5 h to ~1.5 h. Note a round measured while evals run
concurrently is slower than collection alone.

**Big win -- vectorised envs.** Everything runs `--num_envs 1`: one Isaac instance, one robot, one
rollout, ~8.8 min for 10 s of simulated time. Isaac is designed to step hundreds of envs per GPU,
amortising physics, rendering and startup, so 16 envs in ONE process beats 16 processes. This is a
refactor, not a flag: `DaggerDriver`, `GraspGate` and the recorder all index env 0 and assume a
single trajectory. Build it with per-env verification -- a silent cross-env mix-up would poison the
dataset exactly the way the broken-symlink and replay-divergence bugs did.

The same applies to eval: 60 episodes currently costs ~80 min.
