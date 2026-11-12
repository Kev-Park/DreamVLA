"""Evaluation-only observation settings; no simulator imports required."""


def disable_observation_noise(env_cfg) -> list[str]:
    """Disable Isaac observation corruption before constructing the environment.

    Mutates only the supplied evaluation config. Training configs, reset/physics
    randomization, observation normalization, and GR00T sampling are untouched.
    Isaac's ObservationManager removes each term's noise when this group flag
    is false. Iterate the same config entries that the manager uses.
    """
    observations = env_cfg.observations
    groups = (
        observations.items()
        if isinstance(observations, dict)
        else vars(observations).items()
    )
    disabled = []
    for name, group in groups:
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False
            disabled.append(name)
    if not disabled:
        raise ValueError(
            "Evaluation config has no observation groups with enable_corruption"
        )
    return sorted(disabled)


def validate_noise_free_resume(summary: dict) -> None:
    """Do not mix legacy noisy/unspecified evaluations with new noise-free ones."""
    if summary.get("observation_noise_enabled") is not False:
        raise RuntimeError(
            "Refusing to resume an evaluation with observation noise enabled or unspecified. "
            "Use a new output directory for noise-free evaluation; keep the old results intact."
        )
