"""CPU-testable failure-replay and cumulative-data bookkeeping.

This is not an autonomous collection launcher. Expert residual-history semantics
must be resolved before using these utilities for full DAgger experiments.
"""

import random
from pathlib import Path

from .contracts import validate_episode
from .io import digest


def replay_plan(outcomes, attempts, seed=0, max_takeover_seconds=10.0, control_hz=50):
    """Sample takeover times only from failed student rollouts; no quota retries.

    Replay must use exactly the original reference and reset seed. This is a
    recipe, not an assertion that replay without a simulator snapshot is exact.
    """
    if attempts < 1 or max_takeover_seconds <= 0 or control_hz < 1:
        raise ValueError("Invalid replay budget")
    failed = [row for row in outcomes if not row["accepted"]]
    if not failed:
        return []
    rng = random.Random(seed)
    result = []
    for attempt in range(attempts):
        row = failed[attempt % len(failed)]
        end = min(int(max_takeover_seconds * control_hz), int(row["steps"]))
        if end < 1:
            raise ValueError("Failed rollout must contain at least one control step")
        result.append(
            dict(
                attempt=attempt,
                reference_name=row["reference_name"],
                seed=row["seed"],
                takeover_step=rng.randrange(end),
                source_attempt=row["attempt"],
            )
        )
    return result


def aggregate(original, corrections, task):
    """Success-only full-state replay; never mislabel a student prefix as expert.

    Every included observation must have an expert label, including prefixes.
    Masked-prefix/suffix-only variants require a different, mask-aware trainer.
    """
    episodes = [Path(x).resolve() for x in [*original, *corrections]]
    if len(set(episodes)) != len(episodes):
        raise ValueError("Duplicate cumulative demonstration paths")
    fingerprints = set()
    result = []
    for path in episodes:
        meta = validate_episode(path, task)
        sha = digest(path / "data.npz")
        if sha in fingerprints:
            raise ValueError("Duplicate cumulative demonstration contents")
        fingerprints.add(sha)
        if path in [Path(x).resolve() for x in corrections] and not meta.get(
            "full_state_expert_labels"
        ):
            raise ValueError(
                "Correction episodes must carry expert labels at every recorded state"
            )
        result.append(dict(path=str(path), data_sha256=sha, frames=meta["frames"]))
    return result
