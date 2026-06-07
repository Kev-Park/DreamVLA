"""Split-head ActorCritic for the SONIC body-token + finger-scalar action layout.

The standard rsl_rl ``ActorCritic`` has a single MLP whose final linear layer projects a
shared embedding to all action dims via independent linear weights. For our 65-D action
space (64 body token + 1 finger scalar), this means the body output dims and the finger
output dim read from the same embedding via pure linear projections. The trunk learns
features that are useful for *every* output, including the strong ``is_closed`` feature
that the finger reward backpropagates into; the linear body projections can't suppress
this entanglement and end up reading the same signal, which is why the body tokens shift
into the "curl cluster" at the grasp transition even though no reward incentivizes it.

This class replaces the standard single-head actor with:
    trunk  : obs → shared embedding (same dims as before: [512, 256, 256])
    body   : embedding → MLP → 64 body tokens
    finger : embedding → MLP → 1 finger scalar
Each head has its own nonlinear MLP. The nonlinearity per head is the key part — it gives
each head capacity to learn *different combinations* of the embedding features, so the
body head can downweight is_closed in favor of target_ref signals while the finger head
amplifies is_closed. A pure linear readout couldn't do this conditional logic.

Critic, std parameter, distribution sampling, and PPO update machinery are all unchanged
from the base ActorCritic — the rest of rsl_rl sees the same interface (a 65-D action
mean comes out of self.actor) regardless of the internal architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic


# Right-hand finger scalar count. Total action = body_token_dim + finger_dim = 64 + 1 = 65.
_BODY_TOKEN_DIM = 64
_FINGER_DIM = 1


def _resolve_activation(name: str):
    return {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "crelu": nn.CELU,
    }[name]


def _build_head(in_dim: int, out_dim: int, hidden_dims: list, activation_class) -> nn.Sequential:
    """MLP with `hidden_dims` ELU-activated hidden layers and a linear output layer."""
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation_class())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SplitHeadActor(nn.Module):
    """Shared trunk + two parallel output heads (body tokens + finger scalar).

    The body and finger output streams each have their own nonlinear MLP that reads the
    shared trunk embedding. This is the architecture change that lets the body head
    learn different feature combinations than the finger head, breaking the linear-
    readout coupling that the standard single-head actor had.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        trunk_dims: list,
        body_head_dims: list,
        finger_head_dims: list,
        activation_class,
    ):
        super().__init__()
        assert num_actions == _BODY_TOKEN_DIM + _FINGER_DIM, (
            f"num_actions ({num_actions}) must equal body_token_dim ({_BODY_TOKEN_DIM}) "
            f"+ finger_dim ({_FINGER_DIM}). Did you forget to update the wrapper?"
        )

        # Shared trunk: obs → embedding (dim = trunk_dims[-1])
        trunk_layers = []
        prev = num_obs
        for h in trunk_dims:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(activation_class())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)
        embedding_dim = trunk_dims[-1]

        # Body head: embedding → body tokens (64-D)
        self.body_head = _build_head(embedding_dim, _BODY_TOKEN_DIM, body_head_dims, activation_class)

        # Finger head: embedding → finger scalar (1-D)
        self.finger_head = _build_head(embedding_dim, _FINGER_DIM, finger_head_dims, activation_class)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        body = self.body_head(h)       # (..., 64)
        finger = self.finger_head(h)   # (..., 1)
        return torch.cat([body, finger], dim=-1)  # (..., 65)


class SplitHeadActorCritic(ActorCritic):
    """ActorCritic with a SplitHeadActor; critic and PPO machinery untouched.

    Defaults: trunk = [512, 256, 256] (matches G1FlatPPORunnerCfg's actor_hidden_dims);
    body_head_dims = [128]; finger_head_dims = [32]. Adds ~49k params on top of the base
    ~340k actor (~15% increase, well within standard humanoid-PPO sizes).

    The ``head_dims`` are not currently exposed through ``RslRlPpoActorCriticCfg``, so we
    take them as constructor kwargs with sensible defaults. Override by passing
    ``body_head_dims=[...]`` / ``finger_head_dims=[...]`` in the policy cfg if needed.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation: str = "elu",
        body_head_dims=[128],
        finger_head_dims=[32],
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        # Build parent's critic, std, distribution machinery, etc. The parent also builds a
        # standard single-MLP actor at self.actor — we replace it below.
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        # Replace the standard actor with the split-head version, on the same device.
        device = next(self.actor.parameters()).device
        activation_class = _resolve_activation(activation)
        self.actor = SplitHeadActor(
            num_obs=num_actor_obs,
            num_actions=num_actions,
            trunk_dims=actor_hidden_dims,
            body_head_dims=body_head_dims,
            finger_head_dims=finger_head_dims,
            activation_class=activation_class,
        ).to(device)

        n_actor = sum(p.numel() for p in self.actor.parameters())
        n_critic = sum(p.numel() for p in self.critic.parameters())
        print(
            f"[SplitHeadActorCritic] actor=split-head "
            f"(trunk={actor_hidden_dims}, body_head={body_head_dims}, finger_head={finger_head_dims}); "
            f"params: actor={n_actor:,}  critic={n_critic:,}"
        )
