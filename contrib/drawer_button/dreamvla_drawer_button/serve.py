#!/usr/bin/env python3
"""Serve the fine-tuned Open Drawer GR00T policy to the Isaac evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
from .paths import add_external_paths

from .protocol import (  # noqa: E402
    message_text,
    receive_message,
    send_message,
)


STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
ACTION_DIMS = {
    "motion_token": 64,
    "left_hand_joints": 7,
    "right_hand_joints": 7,
}


def _build_observation(message: dict[str, np.ndarray]) -> dict:
    image = np.asarray(message["ego_view"])
    if image.dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError(
            f"ego_view must be uint8 HWC RGB, got {image.dtype} {image.shape}"
        )

    state = {}
    for name, width in STATE_DIMS.items():
        value = np.asarray(message[name], dtype=np.float32)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(
                f"State {name} must have shape {(width,)}, got {value.shape}"
            )
        state[name] = value[None, None]

    prompt = message_text(message, "prompt")
    return {
        "video": {"ego_view": image[None, None]},
        "state": state,
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def _normalize_action(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for name, width in ACTION_DIMS.items():
        value = action.get(name, action.get(f"action.{name}"))
        if value is None:
            raise KeyError(f"Policy omitted action {name!r}; keys={sorted(action)}")
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
            raise ValueError(f"Action {name} has invalid shape {value.shape}")
        value = value[0]
        if not np.isfinite(value).all():
            raise ValueError(f"Action {name} contains non-finite values")
        result[name] = value
    horizons = {value.shape[0] for value in result.values()}
    if horizons != {40}:
        raise ValueError(f"Inconsistent action horizons: {sorted(horizons)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        help="Optional separately trained action head; source checkpoint stays unchanged",
    )
    args = parser.parse_args()
    from .provenance import model_identity

    policy_sha256 = model_identity(
        args.model_path, args.head_checkpoint, denoising_steps=args.denoising_steps
    )

    add_external_paths()
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.denoising_steps <= 0:
        raise ValueError("--denoising-steps must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading GR00T checkpoint: {args.model_path.resolve()}", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.UNITREE_G1_SONIC,
        model_path=str(args.model_path.resolve()),
        device=args.device,
        strict=True,
    )
    diagnostic_head = None
    if args.head_checkpoint is not None:
        from .model_head import apply_diagnostic_head

        diagnostic_head = apply_diagnostic_head(
            policy.model.action_head, args.head_checkpoint
        )
        print(json.dumps({"diagnostic_head": diagnostic_head}), flush=True)
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    inference_count = 0
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # This protocol is deliberately local-only. It never accepts remote clients.
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": "127.0.0.1",
                "port": args.port,
                "device": args.device,
                "denoising_steps": args.denoising_steps,
                "diagnostic_head": diagnostic_head,
            }
        ),
        flush=True,
    )

    try:
        while True:
            connection, address = server.accept()
            print(f"Isaac evaluator connected from {address}", flush=True)
            with connection:
                connection.settimeout(300.0)
                while True:
                    try:
                        request = receive_message(connection)
                    except EOFError:
                        break
                    try:
                        kind = message_text(request, "kind")
                        if kind == "ping":
                            send_message(
                                connection,
                                {
                                    "kind": "ok",
                                    "message": "pong",
                                    "policy_sha256": policy_sha256,
                                },
                            )
                            continue
                        if kind == "reset":
                            if "seed" in request:
                                episode_seed = int(np.asarray(request["seed"]).item())
                                np.random.seed(episode_seed)
                                torch.manual_seed(episode_seed)
                                torch.cuda.manual_seed_all(episode_seed)
                            policy.reset()
                            send_message(connection, {"kind": "ok"})
                            continue
                        if kind == "shutdown":
                            send_message(connection, {"kind": "ok"})
                            return
                        if kind != "infer":
                            raise ValueError(f"Unknown request kind: {kind}")

                        observation = _build_observation(request)
                        started = time.monotonic()
                        with torch.inference_mode():
                            action, _ = policy.get_action(observation)
                        elapsed = time.monotonic() - started
                        result = _normalize_action(action)
                        inference_count += 1
                        send_message(
                            connection,
                            {"kind": "ok", "inference_seconds": elapsed, **result},
                        )
                        if inference_count == 1 or inference_count % 25 == 0:
                            print(
                                f"Inference {inference_count}: {elapsed:.3f}s, "
                                f"horizon={result['motion_token'].shape[0]}",
                                flush=True,
                            )
                    except Exception as exc:
                        send_message(
                            connection,
                            {
                                "kind": "error",
                                "message": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        raise
            print("Isaac evaluator disconnected", flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
