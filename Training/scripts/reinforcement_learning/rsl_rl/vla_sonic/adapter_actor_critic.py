"""ActorCritic with a zero-initialized actor output layer, for residual-adapter training.

Used with ``TokenAdapterVecEnvWrapper``: the actor's output is a token RESIDUAL (+ finger
scalar), so zeroing the final Linear's weight and bias makes the policy mean exactly 0 at
init — i.e., step-0 behavior is pure zero-shot frozen-SONIC playback (residual = 0, finger
scalar = 0 → BinaryJointPositionAction "open", the correct pre-grasp default). Exploration
noise then perturbs around the competent base instead of around random-network output.

Everything else (critic, distribution, PPO machinery) is stock rsl_rl ActorCritic. Same
``*args, **kwargs`` passthrough + registration pattern as split_head_actor_critic.py.
"""

from __future__ import annotations

import torch.nn as nn

from rsl_rl.modules import ActorCritic


class AdapterActorCritic(ActorCritic):
    """Stock ActorCritic whose actor's last Linear layer is zero-initialized."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        linears = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise RuntimeError(
                "AdapterActorCritic could not find any nn.Linear in the actor — "
                "rsl_rl ActorCritic structure may have changed."
            )
        last = linears[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        n_actor = sum(p.numel() for p in self.actor.parameters())
        n_critic = sum(p.numel() for p in self.critic.parameters()) if hasattr(self, "critic") else 0
        hidden = [layer.out_features for layer in linears[:-1]]
        print(
            f"[AdapterActorCritic] actor hidden={hidden}, out={last.out_features} "
            f"(last layer ZERO-INIT → step-0 policy mean = 0 = pure frozen-SONIC behavior); "
            f"params: actor={n_actor:,} critic={n_critic:,}"
        )
