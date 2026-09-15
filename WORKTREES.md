# Worktree-based development

Several branches can be developed and run **concurrently on one cluster** from independent
checkouts, with no per-checkout configuration and nothing for a plain `git clone` user to do
differently. This page is the whole contract.

## Layout

```
~/kevin/
  DreamVLA/                  main checkout (unchanged; keeps working exactly as before)
  holosoma/                  shared holosoma fork
  GR00T-WholeBodyControl/    shared
  sonic/                     shared SONIC .pt weights
  hs_input/  hs_pick_out/    pooled retargeting stages (Adapter A out, holosoma npz)
  ref_motions/               pooled generated reference datasets (<name>/pick_<id>.pkl)
  datasets/                  GR00T/LeRobot collected episodes (unrelated; left alone)
  wt/<branch>/
    DreamVLA/                git worktree of <branch>
    .venv/                   ~30 MB overlay on dreamcontrol_51 (see below)
    holosoma/                OPTIONAL paired holosoma worktree
```

## Daily use

```bash
cd ~/kevin/DreamVLA
./worktree.sh new my-branch                  # DreamVLA worktree + overlay venv
./worktree.sh new my-branch kevin-cbf-v2     # ...plus a paired holosoma worktree on that branch
./worktree.sh ls
./worktree.sh rm  my-branch

source ~/kevin/wt/my-branch/.venv/bin/activate
cd ~/kevin/wt/my-branch/DreamVLA/Training
python scripts/reinforcement_learning/rsl_rl/train_sonic_adapter.py ...    # exactly as in main
```

Training logs are CWD-relative (`Training/logs/...`), so each worktree keeps its own. Everything
else is found automatically by the rules below.

## How paths resolve (no configuration)

Implemented once in `Training/source/isaaclab_tasks/isaaclab_tasks/utils/repo_paths.py` and
mirrored as shell functions in `TrajGen/holosoma_adapters/*.sh` and `worktree.sh`.

| kind | function | search order | examples |
|---|---|---|---|
| sibling **code** | `sibling(name)` | beside *this* checkout → `~/kevin` → `$HOME` | `holosoma`, `GR00T-WholeBodyControl` |
| pooled **data** | `pooled(name)` | `~/kevin` → beside this checkout → `$HOME` | `hs_input`, `hs_pick_out`, `ref_motions` |
| reference dataset | `dataset(p)` | `p` if it exists, else `pooled("ref_motions", basename(p))` | `--ref-motions-path Holosoma_Pick_29_full` |

Code resolves worktree-local first so a paired `holosoma/` worktree is picked up automatically and
two branches can run different retargeters at once. Data resolves shared-first so every checkout
sees the same datasets. Any of them is overridden by an env var named after the directory
(`HOLOSOMA_DIR`, `REF_MOTIONS_DIR`, `HS_INPUT_DIR`, ...).

### Datasets

Generated `.pkl` datasets live in **`~/kevin/ref_motions/<name>/`**, not in the repo.
(`~/kevin/datasets/` is the GR00T/LeRobot collected-episode root and is unrelated.)

- produce: `gen_dataset.sh <gpu> <name> <ids...>` — a bare name writes to the pool; a path is used as-is
- consume: `--ref-motions-path <name>` from any checkout. Old `../TrajGen/sample/<name>` paths still
  resolve (basename lookup), and tracked source datasets like `../TrajGen/sample/Pick_sim2` are untouched.
- the main checkout keeps `TrajGen/sample/Holosoma_* -> ~/kevin/ref_motions/*` symlinks so historical
  commands keep working. New datasets do not need one.

`hs_pick_out` is pooled too, so **a run with a non-default retargeter config must set `HS_NPZ_DIR`**
or it overwrites the pooled `pick_<id>_original.npz` other branches read.

## The overlay venv

`python -m venv --system-site-packages` on top of `dreamcontrol_51`, with this checkout's six
repo-local packages (`Training/source/isaaclab{,_assets,_mimic,_rl,_tasks}`, `Training/isaac_utils`)
installed editable. The overlay's finders are consulted before the base env's, so the worktree's
copies win; torch / isaacsim / gear_sonic / gr00t / everything else is inherited untouched.
`dreamcontrol_51` itself is never modified.

`holosoma_retargeting` is an editable install in `hsretargeting` pointing at the main clone;
`gen_dataset.sh` / `grasp_batch.sh` prepend the resolved holosoma checkout to `PYTHONPATH`, which
outranks editable installs, so a paired worktree is really what runs.

## Rules of thumb

- A worktree cannot check out a branch that is already checked out elsewhere (git refuses). Use a
  different branch per worktree; `worktree.sh new` creates one from HEAD if it does not exist.
- Only pass a holosoma branch when the work touches the retargeter; otherwise the shared clone is fine.
- Editing a *sibling* repo that is not paired (e.g. GR00T-WholeBodyControl) still edits the shared
  clone. Pair it or accept that it is shared.
- Commits still happen locally on Windows and reach the cluster by `git pull` **in the worktree**
  (`git -C ~/kevin/wt/<branch>/DreamVLA pull`). Nothing changes about where code is edited.
- `worktree.sh rm` removes the checkout and venv; branch and commits are untouched.
