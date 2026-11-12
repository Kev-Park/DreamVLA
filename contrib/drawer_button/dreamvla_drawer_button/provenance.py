"""Content identity for local policy artifacts, without recording credentials."""

import hashlib
import json
from pathlib import Path

from .io import digest


def model_identity(folder, head=None, *, denoising_steps=4):
    folder = Path(folder)
    paths = sorted(
        {
            path
            for pattern in ("*.json", "*.safetensors", "*.bin", "processor/*.json")
            for path in folder.glob(pattern)
            if path.is_file()
        }
    )
    if not paths:
        raise ValueError("No local model artifacts found for policy provenance")
    manifest = {str(path.relative_to(folder)): digest(path) for path in paths}
    if head is not None:
        manifest["separate_action_head"] = digest(head)
    manifest["denoising_steps"] = denoising_steps
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
