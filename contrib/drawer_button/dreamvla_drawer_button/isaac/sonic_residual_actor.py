"""Opt-in zero-mean residual actor, leaving stock RSL-RL actors untouched."""

import torch.nn as nn
from rsl_rl.modules import ActorCritic


class DreamVlaResidualActorCritic(ActorCritic):
    """Same architecture/distribution as stock; zero only the final actor layer.

    As in DreamVLA AdapterActorCritic, the initial POLICY MEAN is zero. PPO
    exploration remains enabled. Checkpoint loading replaces these initial
    values normally; it must never zero an already trained policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        linears = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise RuntimeError("No actor output Linear found")
        nn.init.zeros_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)
        print(
            "DreamVlaResidualActorCritic: actor output weight/bias zero; exploration unchanged",
            flush=True,
        )


def register_residual_actor():
    """RSL-RL 2.3 resolves class names in its runner module's namespace."""
    import rsl_rl.runners.on_policy_runner as runner_module

    name = "DreamVlaResidualActorCritic"
    existing = getattr(runner_module, name, None)
    if existing is not None and existing is not DreamVlaResidualActorCritic:
        raise RuntimeError("Conflicting residual actor registration")
    setattr(runner_module, name, DreamVlaResidualActorCritic)
