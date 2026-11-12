"""Explicit Isaac manager signatures and train/eval scene consistency helpers."""


def close_drawer_on_reset(cfg):
    old = cfg.events.reset_cabinet.func

    def reset_cabinet(
        env, env_ids, root_offset_x=0.425, root_offset_y=0.0, root_offset_z=-0.215
    ):
        cabinet = env.scene["cabinet"]
        cabinet.write_joint_state_to_sim(
            cabinet.data.default_joint_pos[env_ids].clone(),
            cabinet.data.default_joint_vel[env_ids].clone(),
            env_ids=env_ids,
        )
        old(
            env,
            env_ids,
            root_offset_x=root_offset_x,
            root_offset_y=root_offset_y,
            root_offset_z=root_offset_z,
        )

    cfg.events.reset_cabinet.func = reset_cabinet


class NoVisual:
    def reset(self):
        pass

    def update(self):
        pass
