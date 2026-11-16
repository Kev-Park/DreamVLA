"""Bounded, resumable I/O for original drawer IK generators.

Reference pickles must be trusted, locally produced DreamControl inputs.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from .io import read, digest, write_json


class Generation:
    def __init__(self, kind, script):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--point-plan", type=Path, required=True)
        parser.add_argument("--max-points", type=int)
        args = parser.parse_args()
        self.enabled = True
        self.kind = kind
        self.plan = read(args.point_plan)
        self.hash = digest(args.point_plan)
        if self.plan["version"] != "drawer-phase1-dense-grid-v1":
            raise ValueError("Unsupported reference plan")
        if digest(script) != self.plan["generator_sha256"][kind]:
            raise ValueError("Generator changed since the point plan was created")
        if digest(self.plan["base"][kind]) != self.plan["base_sha256"][kind]:
            raise ValueError("Base retargeted motion changed since planning")
        self.points = np.asarray(self.plan["points"][kind], dtype=np.float64)
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 3
            or not np.isfinite(self.points).all()
        ):
            raise ValueError("Expected finite N x 3 target points")
        if args.max_points is not None:
            if args.max_points < 1:
                raise ValueError("--max-points must be positive")
            self.points = self.points[: args.max_points]
        self.total = len(self.points)
        self.output = Path(self.plan["output"]) / kind
        self.output.mkdir(parents=True, exist_ok=True)
        import torch

        torch.set_num_threads(1)

    def filename(self, index):
        return self.output / f"{self.kind}_{index:05d}.pkl"

    def resume(self, index):
        path = self.filename(index)
        if not path.exists():
            return None
        with path.open("rb") as stream:
            data = pickle.load(stream)
        provenance = data["_phase1_expansion"]
        if (
            provenance["plan_sha256"] != self.hash
            or provenance["point_index"] != index
            or provenance["target_xyz"] != self.points[index].tolist()
        ):
            raise ValueError("Reference belongs to a different generation plan")
        return np.asarray(data["_ik_optimized_joints"]).copy()

    def save(self, index, data, optimized, cost):
        arrays = [
            np.asarray(data["joints"]),
            np.asarray(data["global_pose"].translation()),
            np.asarray(data["global_pose"].rotation().wxyz),
        ]
        if (
            len(arrays[0]) < 200
            or not np.isfinite(cost)
            or any(not np.isfinite(x).all() for x in arrays)
        ):
            raise ValueError("Invalid reference or optimizer cost")
        data["_phase1_expansion"] = dict(
            plan_sha256=self.hash,
            point_index=index,
            kind=self.kind,
            target_xyz=self.points[index].tolist(),
            final_ik_cost=float(cost),
            generator_sha256=self.plan["generator_sha256"][self.kind],
        )
        data["_ik_optimized_joints"] = np.asarray(optimized).copy()
        path = self.filename(index)
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            pickle.dump(data, stream)
        temporary.replace(path)
        write_json(
            self.output / "progress.json",
            dict(
                kind=self.kind,
                completed=index + 1,
                target=self.total,
                plan_sha256=self.hash,
            ),
            durable=True,
        )
