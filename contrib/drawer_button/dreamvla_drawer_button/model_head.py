"""Restore a reduced diagnostic SONIC head onto an IN-MEMORY policy copy.

Never saves or modifies a source model. Evaluation-only learning experiment.
"""

import hashlib
from pathlib import Path
import torch

ADAPTERS = (
    "state_encoder.layer1",
    "state_encoder.layer2",
    "action_encoder.W1",
    "action_encoder.W2",
    "action_encoder.W3",
    "action_decoder.layer1",
    "action_decoder.layer2",
)
ROW_KEYS = {f"{name}.{param}" for name in ADAPTERS for param in ("W", "b")}


def validate_mapping(target, source, row=11):
    if target.keys() != source.keys():
        raise ValueError("Head keys differ; do not partially load a diagnostic head")
    rows = []
    full = []
    for key, dest in target.items():
        src = source[key]
        if key in ROW_KEYS:
            if (
                src.shape[0] != 1
                or dest.shape[0] <= row
                or src.shape[1:] != dest.shape[1:]
            ):
                raise ValueError(f"Invalid active-row restoration: {key}")
            rows.append(key)
        else:
            if src.shape != dest.shape:
                raise ValueError(f"Head shape mismatch: {key}")
            full.append(key)
        if not torch.isfinite(src).all():
            raise ValueError(f"Nonfinite learned parameter: {key}")
    if set(rows) != ROW_KEYS:
        raise ValueError("Expected all fourteen embodiment adapter tensors")
    return rows, full


def restore_state(target, source, row=11):
    rows, full = validate_mapping(target, source, row)
    with torch.no_grad():
        for key in rows:
            target[key][row : row + 1].copy_(source[key])
        for key in full:
            target[key].copy_(source[key])
    return rows, full


def apply_diagnostic_head(head, path):
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["config"]["action_horizon"] != 40
        or payload["config"]["max_num_embodiments"] != 1
    ):
        raise ValueError("Unexpected diagnostic model configuration")
    rows, full = restore_state(head.state_dict(), payload["state_dict"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return dict(
        kind="groot_diagnostic_head",
        head_path=str(path.resolve()),
        head_sha256=digest.hexdigest(),
        active_embodiment_row=11,
        restored_adapter_tensors=len(rows),
        restored_shared_tensors=len(full),
        source_checkpoint_unchanged=True,
        small_data_memorization_experiment=True,
        production_validated=False,
    )
