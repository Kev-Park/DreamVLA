"""Explicit, checkpoint-bound contracts for the PPO repair campaign."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import torch


@dataclass(frozen=True)
class RepairSpec:
    mode: str = "multiplicative_free"
    scale: float = 0.1
    snap: bool = True
    base_observation: bool = False
    action_rate_factor: float = 1.0

    def __post_init__(self):
        if self.mode not in ("multiplicative_free", "additive"):
            raise ValueError("Unsupported composition")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Scale must be finite and positive")
        if not math.isfinite(self.action_rate_factor) or self.action_rate_factor < 0:
            raise ValueError("Invalid action-rate weight multiplier")
        if type(self.snap) is not bool or type(self.base_observation) is not bool:
            raise ValueError("Boolean fields required")

    def to_dict(self):
        return asdict(self)

    @property
    def upper(self):
        return 15 / 16 if self.snap else 1.0


def compose(reference, residual, spec):
    token = (
        reference * (1.0 + spec.scale * residual)
        if spec.mode == "multiplicative_free"
        else reference + spec.scale * residual
    )
    if spec.snap:
        token = torch.round(token * 16.0) / 16.0
    return token.clamp(-1.0, spec.upper)


def validate_checkpoint(checkpoint, expected):
    import yaml

    params = Path(checkpoint).resolve().parent / "params"
    spec = RepairSpec(**json.loads((params / "repair_spec.json").read_text()))
    if spec != expected:
        raise ValueError(f"Checkpoint repair contract {spec} != requested {expected}")
    with (params / "env.yaml").open() as stream:
        env = yaml.load(stream, Loader=yaml.BaseLoader)
    with (params / "agent.yaml").open() as stream:
        agent = yaml.load(stream, Loader=yaml.BaseLoader)
    action = env["actions"]["joint_pos"]
    if (
        "TeacherRepairAction" not in action["class_type"]
        or action["residual_transform"] != spec.mode
        or float(action["residual_scale"]) != spec.scale
        or [float(x) for x in action["latent_clip"]] != [-1.0, spec.upper]
        or action["token_postprocess"]
        != ("fsq32_lattice_round" if spec.snap else "none")
        or agent["policy"]["class_name"] != "DreamVlaResidualActorCritic"
        or float(env["rewards"]["action_rate_l2"]["weight"])
        != -0.005 * spec.action_rate_factor
    ):
        raise ValueError("Saved environment/actor disagrees with repair contract")
    return spec
