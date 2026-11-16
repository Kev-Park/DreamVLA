"""Atomic, bounded checkpoints for a single uninterrupted head-training schedule."""

import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng(shuffle_rng):
    n = np.random.get_state()
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        python=random.getstate(),
        numpy=dict(
            name=n[0],
            keys=n[1].tolist(),
            pos=n[2],
            has_gauss=n[3],
            cached_gaussian=n[4],
        ),
        shuffle=shuffle_rng.bit_generator.state,
    )


def restore_rng(state, shuffle_rng):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count changed")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state(
        (
            n["name"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["pos"],
            n["has_gauss"],
            n["cached_gaussian"],
        )
    )
    shuffle_rng.bit_generator.state = state["shuffle"]


def atomic_save(payload, path, *, rotate=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        "." + path.name + ".pending-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    # Keep an interrupted temporary file for inspection; never damage the last
    # two complete checkpoints if serialization/fsync fails.
    with temp.open("xb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    if rotate and path.exists():
        os.replace(path, path.with_name("resume_previous.pt"))
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_resume(
    path,
    head,
    optimizer,
    step,
    contract,
    order,
    shuffle_rng,
    elapsed,
    adapter_init,
    *,
    min_free_bytes=0,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(Path(path).parent).free < min_free_bytes:
        raise RuntimeError("Below the configured checkpoint disk safety floor")
    payload = dict(
        version="groot-head-exact-resume-v1",
        completed_updates=step,
        contract=contract,
        config=head.config.to_dict(),
        state_dict=cpu_tree(head.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()),
        order=np.asarray(order).tolist(),
        rng=capture_rng(shuffle_rng),
        elapsed_seconds=elapsed,
        adapter_initialization=adapter_init,
    )
    atomic_save(payload, path, rotate=True)


def load_resume(path, contract):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("version") != "groot-head-exact-resume-v1"
        or payload.get("contract") != contract
        or not 0 < payload["completed_updates"] <= contract["options"]["steps"]
        or sorted(payload["order"]) != list(range(len(contract["window_names"])))
    ):
        raise ValueError(
            "Resume checkpoint differs from the original run, schedule, or data"
        )
    return payload


def restore_training(payload, head, optimizer, shuffle_rng):
    head.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"], shuffle_rng)
    return np.asarray(payload["order"], dtype=np.int64), payload["completed_updates"]


def reconcile_log(output, completed):
    """Preserve interrupted history; retain only updates committed by the checkpoint."""
    log = Path(output) / "training.jsonl"
    if not log.exists():
        return
    rows = []
    for line in log.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row["step"] <= completed:
            rows.append(row)
    archive = Path(output) / "resume_history"
    archive.mkdir(exist_ok=True)
    log.rename(archive / f"training_before_resume_{time.time_ns()}.jsonl")
    with log.open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
