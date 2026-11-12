"""Shared audit/update logic for the original visual-only button panel.

No alternate procedural fixture is shipped: training RGB capture and camera
evaluation must use the same derived upstream panel in button_visual.py.
"""


def audit_visual_only(prim):
    from pxr import Usd

    forbidden = []
    for item in Usd.PrimRange(prim):
        schemas = list(item.GetAppliedSchemas())
        if any(
            "Collision" in s or "RigidBody" in s or "Articulation" in s for s in schemas
        ):
            forbidden.append(str(item.GetPath()))
    if forbidden:
        raise RuntimeError(
            "Visual asset unexpectedly contains physics: " + repr(forbidden)
        )
    return dict(collision_api_count=0, rigid_body_api_count=0, articulation_api_count=0)


class ButtonTaskVisual:
    """Subclasses attach the actual panel, wrist frame and fingertip offset."""

    def reset(self):
        for button in self.buttons:
            button.reset()

    def update(self):
        env = self.env
        robot = env.scene["robot"]
        times = env.episode_length_buf * env.step_dt + env.start_motion_times
        ref = env.motion_lib.get_motion_state(env.motion_ids, times)
        targets = ref["grab_pos"] + ref["offsets"] + env.scene.env_origins
        tips = robot.data.body_pos_w[:, self.body, :] + self.quat_apply(
            robot.data.body_quat_w[:, self.body, :], self.offset
        )
        targets = targets.detach().cpu().numpy()
        tips = tips.detach().cpu().numpy()
        ground = env.scene.env_origins[:, 2].detach().cpu().numpy()
        depths = []
        for button, target, tip, z in zip(self.buttons, targets, tips, ground):
            button.place(target, float(z))
            depths.append(button.update(tip, env.step_dt))
        return dict(target_world_m=targets.tolist(), cap_depression_m=depths)
