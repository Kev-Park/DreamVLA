"""Sample around the standard inference mean using the checkpoint's learned std."""
import torch


def sampled_inference_policy(runner, inference_policy):
    # Calling the standard inference path preserves its observation normalization.
    actor = getattr(runner.alg, "policy", None)
    if actor is None:
        actor = runner.alg.actor_critic
    if hasattr(actor, "std"):
        std = actor.std.detach()
    elif hasattr(actor, "log_std"):
        std = actor.log_std.detach().exp()
    else:
        raise RuntimeError("Expert has no recognized learned Normal standard deviation")
    if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError("Expert checkpoint contains invalid action standard deviations")

    def sample(obs):
        mean = inference_policy(obs)
        return mean + std.to(mean) * torch.randn_like(mean)

    return sample
