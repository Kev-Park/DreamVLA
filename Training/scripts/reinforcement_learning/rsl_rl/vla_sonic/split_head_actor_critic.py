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

    Defaults: body_head_dims = [128]; finger_head_dims = [32]. Trunk dims and activation
    are inherited from whatever rsl_rl uses to build the standard actor (via the policy
    cfg's ``actor_hidden_dims`` / ``activation``).

    Constructor uses ``*args, **kwargs`` passthrough because rsl_rl's ``ActorCritic``
    signature varies across versions (e.g. rsl_rl 3.0.1 takes ``obs, obs_groups,
    num_actions, ...`` as positional/required kwargs; older versions take
    ``num_actor_obs, num_critic_obs, num_actions, ...``). We let the parent build its
    standard actor under whatever its signature is, then introspect the result to extract
    input/output dims, hidden dims, and activation — and replace ``self.actor`` with a
    SplitHeadActor mirroring those choices.
    """

    def __init__(self, *args, body_head_dims=None, finger_head_dims=None, **kwargs):
        # Extract our custom kwargs (defaults inline since list defaults are mutable).
        body_head_dims = [128] if body_head_dims is None else list(body_head_dims)
        finger_head_dims = [32] if finger_head_dims is None else list(finger_head_dims)

        # Let parent handle all its required args under whatever signature it uses.
        super().__init__(*args, **kwargs)

        # Introspect parent's actor to extract dims for our split-head replacement.
        old_actor = self.actor
        linears = [m for m in old_actor.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise RuntimeError(
                "SplitHeadActorCritic could not find any nn.Linear in parent's actor — "
                "the rsl_rl ActorCritic structure may have changed. Inspect parent.actor."
            )
        num_obs = linears[0].in_features
        num_actions = linears[-1].out_features
        trunk_dims = [layer.out_features for layer in linears[:-1]]  # hidden layer sizes

        # Activation class — find first non-Linear module that looks like an activation.
        _activation_types = (nn.ELU, nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.SELU, nn.CELU)
        activation_class = nn.ELU  # safe default
        for module in old_actor.modules():
            if isinstance(module, _activation_types):
                activation_class = type(module)
                break

        # Build and replace, on the same device as parent's actor.
        device = next(old_actor.parameters()).device
        self.actor = SplitHeadActor(
            num_obs=num_obs,
            num_actions=num_actions,
            trunk_dims=trunk_dims,
            body_head_dims=body_head_dims,
            finger_head_dims=finger_head_dims,
            activation_class=activation_class,
        ).to(device)

        n_actor = sum(p.numel() for p in self.actor.parameters())
        n_critic = sum(p.numel() for p in self.critic.parameters()) if hasattr(self, "critic") else 0
        print(
            f"[SplitHeadActorCritic] actor=split-head "
            f"(trunk={trunk_dims}, body_head={body_head_dims}, finger_head={finger_head_dims}, "
            f"activation={activation_class.__name__}); "
            f"params: actor={n_actor:,}  critic={n_critic:,}"
        )
