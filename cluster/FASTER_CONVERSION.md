# Faster HDF5 -> LeRobot conversion (sharded)

`convert_isaac_hdf5_to_lerobot.py` is strictly single-process: one `exporter.add_frame()` /
`save_episode()` loop, no `--num-workers`, no `Pool`, no image-writer threads. Measured cost:

| episodes | wall clock |
|---|---|
| 111 | 59 min |
| 175 | 78 min |

~0.5 min/episode, on a box with **104 CPUs** that sit idle throughout. LeRobot 0.1.0 also defaults
to **`libsvtav1` at CRF 30**; AV1 is far slower than H.264 and is the likely inner bottleneck.

## The approach (VALIDATED 2026-10-05): shard the conversion, then merge exactly

    cluster/convert_sharded.sh <the converter's own arguments> [--shards 8]

`convert_sharded.sh` lists the inputs exactly as the converter does (`sorted(root.glob('**/*.hdf5'))`),
splits them into N **contiguous** slices, runs N converters in parallel and merges the shards in
order with `merge_lerobot.py`. Because the converter sorts globally, the shards concatenated in
order reproduce the single-process episode order; the merge only renumbers `episode_index` / `index`
(parquet rewritten with LeRobot's own `datasets.to_parquet`, the two columns' episode stats recomputed
with LeRobot's `compute_episode_stats`) and moves video files unchanged. It prints the same
`[done] wrote N frames across M episode(s)` line, so it is a drop-in for chain scripts.

**Validation** (24 rollouts of the af60v7 round-5 aggregate, 4 shards vs one process, `verify_merge.py`):
every parquet, video and meta file byte-identical, and GR00T's generated `stats.json` /
`relative_stats.json` byte-identical. Wall clock 809 s vs 1491 s on a box at load ~110 (other users).

### Why NOT train on the shards directly via `dataset_paths`

Checked in `gr00t/data/dataset/sharded_mixture_dataset.py::merge_statistics`: across datasets GR00T
pools mean/std correctly (length-weighted) but sets **q01 = min of the per-dataset q01s and q99 = max
of the q99s** -- a conservative envelope, not the union's percentiles. GR00T normalises on q01/q99,
so N shard datasets would be scaled differently from one dataset. Merging avoids this.

### stats.json / relative_stats.json

The converter never writes them; GR00T's `generate_stats()` / `generate_rel_stats()` compute them from
the parquet at fine-tune time over the whole dataset. `merge_lerobot.py` therefore writes neither and
**never copies a shard's** -- the previous version copied shard 0's `relative_stats.json`, which GR00T
would have accepted as valid and used to normalise relative actions with one shard's statistics.

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
