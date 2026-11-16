"""Create nested seeded target plans; planning itself does not run IK or use GPUs."""

import argparse
from pathlib import Path

import numpy as np

from .io import digest, write_json


def points(bounds, count, seed):
    axes = [np.linspace(lo, hi, n) for lo, hi, n in bounds]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if not 1 <= count <= len(grid):
        raise ValueError("Requested count exceeds this reference grid")
    order = np.random.default_rng(seed).permutation(len(grid))
    return grid[order[:count]].tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("drawer", "standing_button"), required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--base-stand", type=Path)
    p.add_argument("--base-squat", type=Path)
    p.add_argument("--base-button", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Generated-reference directory, not the plan file",
    )
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.plan.exists() or a.output.exists():
        raise FileExistsError(
            "Choose a new plan/output; never overwrite reference provenance"
        )
    here = Path(__file__).parent
    if a.task == "drawer":
        if a.count < 2 or not a.base_stand or not a.base_squat:
            p.error(
                "Drawer requires >=2 points and both trusted retargeted base motions"
            )
        bases = dict(
            stand=str(a.base_stand.resolve()), squat=str(a.base_squat.resolve())
        )
        plan = dict(
            version="drawer-phase1-dense-grid-v1",
            output=str(a.output.resolve()),
            base=bases,
            base_sha256={k: digest(v) for k, v in bases.items()},
            seed=a.seed,
            generator_sha256={
                k: digest(here / f"generate_drawer_{k}.py") for k in bases
            },
            points=dict(
                stand=points(
                    ((0.28, 0.30, 10), (-0.2, -0.1, 25), (0.7, 0.8, 25)),
                    (a.count + 1) // 2,
                    a.seed,
                ),
                squat=points(
                    ((0.42, 0.44, 10), (-0.2, -0.1, 25), (0.5, 0.7, 25)),
                    a.count // 2,
                    a.seed + 1,
                ),
            ),
        )
    else:
        if not a.base_button:
            p.error(
                "Standing button requires --base-button, a trusted retargeted motion"
            )
        plan = dict(
            version="standing-button-reference-plan-v1",
            output=str(a.output.resolve()),
            base=str(a.base_button.resolve()),
            base_sha256=digest(a.base_button),
            seed=a.seed,
            points=points(
                ((0, 0.1, 10), (-0.1, 0.1, 25), (-0.1, 0.05, 25)), a.count, a.seed
            ),
            generator_sha256=digest(here / "generate_button.py"),
        )
    write_json(a.plan, plan, durable=True)


if __name__ == "__main__":
    main()
