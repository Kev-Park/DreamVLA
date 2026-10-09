"""Evaluation-only ablation: the trained residual actor supplies the entire token."""
import torch

from .token_adapter_wrapper import TokenAdapterVecEnvWrapper


class DirectExpertEvalWrapper(TokenAdapterVecEnvWrapper):
    """Keep actor inputs/history/fingers; bypass base addition and residual gain."""

    def __init__(self, *args, token_mode, **kwargs):
        if token_mode not in ("raw", "snap"):
            raise ValueError(token_mode)
        self.token_mode = token_mode
        super().__init__(*args, **kwargs)

    def compose(self, latent):
        return latent.to(self._dev)

    def _decode_body_29(self, body_latent):
        token = self.snap(body_latent) if self.token_mode == "snap" else body_latent
        if not torch.isfinite(token).all():
            raise RuntimeError("Non-finite expert token; evaluation cannot be scored")
        self._last_token = token.detach()
        out = self.decoder(self._build_obs_dict(token).to(torch.float32))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(self.num_envs, -1)[:, :29]


def load_direct_expert(env, encoder, decoder, checkpoint, task, seed, token_mode):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner
    import rsl_rl.modules
    import rsl_rl.runners.on_policy_runner as runner_module
    from .adapter_actor_critic import AdapterActorCritic

    rsl_rl.modules.AdapterActorCritic = AdapterActorCritic
    runner_module.AdapterActorCritic = AdapterActorCritic
    cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    cfg.seed = seed
    cfg.device = "cuda:0"
    cfg.policy.class_name = "AdapterActorCritic"
    cfg.policy.actor_hidden_dims = [256, 128]
    cfg.policy.critic_hidden_dims = [512, 256, 256]
    env = DirectExpertEvalWrapper(
        env, decoder, encoder, "cuda:0", token_mode=token_mode,
        residual_scale=0.1, residual_transform="additive_free",
        clip_actions=None, pt_mode=True, encoder_mode="g1",
    )
    runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(checkpoint)
    print(f"[direct-expert] checkpoint={checkpoint}; decoder token="
          f"{'actor[:64]' if token_mode == 'raw' else 'snap(actor[:64])'}; "
          "actor still observes reference base token; finger head unchanged", flush=True)
    return env, runner.get_inference_policy(device="cuda:0")
