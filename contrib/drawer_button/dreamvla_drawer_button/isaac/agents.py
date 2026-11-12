# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1"
        # self.policy.actor_hidden_dims = [256, 128, 128]
        # self.policy.critic_hidden_dims = [256, 128, 128]
        self.policy.actor_hidden_dims = [512, 256, 256]
        self.policy.critic_hidden_dims = [512, 256, 256]


@configclass
class G1SonicFastPPORunnerCfg(G1FlatPPORunnerCfg):
    """Fast first-pass PPO settings for the pretrained SONIC action space."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sonic_open_drawer"
        self.max_iterations = 600
        self.save_interval = 25
        self.policy.init_noise_std = 0.25
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.num_learning_epochs = 4


@configclass
class G1SonicNvidiaAdaptPPORunnerCfg(G1SonicFastPPORunnerCfg):
    """Conservative warm-start settings for adapting the legacy teacher to the stock cabinet."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sonic_open_drawer_nvidia"
        self.max_iterations = 150
        self.save_interval = 10
        self.policy.init_noise_std = 0.08
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.006


@configclass
class G1SonicMultiplicativePPORunnerCfg(G1SonicFastPPORunnerCfg):
    """Fresh PPO teacher; never automatically resume the additive experiment."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sonic_open_drawer_nvidia_mul_tanh_010"
        self.resume = False


@configclass
class G1SonicMultiplicativeFreePPORunnerCfg(G1SonicMultiplicativePPORunnerCfg):
    """Fresh unbounded-multiplier control; preserve scale 0.1 and token clip."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sonic_open_drawer_nvidia_mul_free_010"
