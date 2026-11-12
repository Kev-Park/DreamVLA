"""Create immutable training-reference and held-out rollout manifests."""

import argparse
from pathlib import Path
import random

from .contracts import TASKS
from .io import read, write_json, digest


def references(folder):
    folder = Path(folder).resolve()
    paths = sorted(folder.glob("*.pkl"))
    if not paths:
        raise ValueError(f"No trusted reference pickles: {folder}")
    return folder, {path.name: digest(path) for path in paths}


def prepare(task, train, heldout, checkpoint, output, *, episodes=100, seed=0):
    if task not in TASKS or episodes < 1:
        raise ValueError("Unknown task or invalid rollout count")
    train, train_hashes = references(train)
    heldout, eval_hashes = references(heldout)
    if set(train_hashes.values()) & set(eval_hashes.values()):
        raise ValueError("Training and held-out reference contents overlap")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rng = random.Random(seed)
    names = sorted(eval_hashes)
    bank = dict(
        version="drawer-button-eval-bank-v1",
        reference_root=str(heldout),
        reference_sha256=eval_hashes,
        seed=seed,
    )
    for track in ("repeatability", "fresh_start"):
        bank[track] = [
            dict(reference_name=rng.choice(names), seed=rng.randrange(2**31))
            for _ in range(episodes)
        ]
    output.mkdir(parents=True)
    write_json(output / "eval_bank.json", bank, durable=True)
    config = dict(
        version="drawer-button-run-v1",
        task=task,
        n=len(train_hashes),
        reference_root=str(train),
        reference_names=sorted(train_hashes),
        reference_sha256=train_hashes,
        eval_bank=str(output / "eval_bank.json"),
        eval_bank_sha256=digest(output / "eval_bank.json"),
        teacher_checkpoint=str(Path(checkpoint).resolve()),
    )
    write_json(output / "arm.json", config, durable=True)
    return config


def load_run(folder):
    cfg = read(Path(folder) / "arm.json")
    if cfg["version"] != "drawer-button-run-v1" or cfg["task"] not in TASKS:
        raise ValueError("Invalid run configuration")
    if digest(cfg["eval_bank"]) != cfg["eval_bank_sha256"]:
        raise ValueError("Evaluation bank changed")
    bank = read(cfg["eval_bank"])
    for root, hashes in (
        (cfg["reference_root"], cfg["reference_sha256"]),
        (bank["reference_root"], bank["reference_sha256"]),
    ):
        if references(root)[1] != hashes:
            raise ValueError("Reference contents or membership changed")
    return cfg, bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--train-references", type=Path, required=True)
    p.add_argument("--eval-references", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prepare(
        a.task,
        a.train_references,
        a.eval_references,
        a.teacher_checkpoint,
        a.output,
        episodes=a.episodes,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
