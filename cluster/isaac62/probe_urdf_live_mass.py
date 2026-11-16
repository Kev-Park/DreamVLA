"""Measure effective PhysX body masses for matched original/new URDF imports.

This is a diagnostic with two standalone articulations, not a task evaluation.
Completion means measurement succeeded; preservation_pass separately states
whether body names and effective masses agree. No source asset is modified.
"""
import argparse
import json
from pathlib import Path
import time
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-usd', type=Path, required=True)
    parser.add_argument('--modern-usd', type=Path, required=True)
    parser.add_argument('--pre-fix-usd', type=Path, help='Optional preserved asset to measure regression before repair')
    parser.add_argument('--state-info', type=Path, required=True,
                        help='Original converted dataset metadata defining the GR00T joint order')
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, started=time.time(), scope=__doc__, robots={})
    app = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        import torch
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext
        from dreamvla_drawer_button.isaac.robot import G1_SONIC_CFG
        sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))
        robots = {}
        inputs = [('legacy', args.legacy_usd), ('modern', args.modern_usd)]
        if args.pre_fix_usd is not None:
            inputs.append(('pre_fix', args.pre_fix_usd))
        for index, (name, path) in enumerate(inputs):
            cfg = G1_SONIC_CFG.copy()
            cfg.prim_path = '/World/'+name
            cfg.spawn.usd_path = str(path)
            cfg.actuators = {}  # Measure imported limits without actuator overrides masking them.
            cfg.init_state.pos = (index*3.0, 0., 1.)
            robots[name] = Articulation(cfg)
        sim.reset()

        def compare_live_poses(label):
            # Read actual PhysX link transforms, not USD/Gf evaluated matrices.
            # Align at each pelvis so the intentional 3 m scene separation drops out.
            from isaaclab.utils.math import convert_quat, matrix_from_quat, subtract_frame_transforms
            poses = {}
            for name, robot in robots.items():
                raw = robot.root_physx_view.get_link_transforms().clone()[0]
                q = convert_quat(raw[:, 3:7], to='wxyz')
                q = q / q.norm(dim=-1, keepdim=True)
                root = robot.body_names.index('pelvis')
                pos, quat = subtract_frame_transforms(raw[root:root+1, :3].expand_as(raw[:, :3]),
                    q[root:root+1].expand_as(q), raw[:, :3], q)
                rotation = matrix_from_quat(quat)
                poses[name] = {body: dict(position=pos[index].cpu().tolist(),
                    rotation=rotation[index].cpu().tolist()) for index, body in enumerate(robot.body_names)}
            errors = {}
            for body in poses['legacy']:
                a, b = poses['legacy'][body], poses['modern'][body]
                errors[body] = {field: float((torch.tensor(a[field])-torch.tensor(b[field])).abs().max())
                               for field in ('position', 'rotation')}
            report.setdefault('live_pose_comparisons', {})[label] = dict(poses=poses, errors=errors,
                tolerance=1e-6, preservation_pass=all(value <= 1e-6 for row in errors.values() for value in row.values()))

        compare_live_poses('initial')
        for name, robot in robots.items():
            robot.update(sim.get_physics_dt())
            masses = robot.root_physx_view.get_masses().cpu()
            inertias = robot.root_physx_view.get_inertias().cpu()
            coms = robot.root_physx_view.get_coms().cpu()
            velocity_limits = robot.root_physx_view.get_dof_max_velocities().cpu()
            assert torch.isfinite(velocity_limits).all()
            assert torch.isfinite(masses).all() and torch.isfinite(inertias).all() and torch.isfinite(coms).all()
            assert masses.shape == (1, len(robot.body_names))
            report['robots'][name] = dict(body_names=list(robot.body_names), masses=masses[0].tolist(),
                                         inertias=inertias[0].tolist(), coms=coms[0].tolist(), joint_names=list(robot.joint_names),
                                         total_mass=float(masses.sum()), velocity_limits=velocity_limits[0].tolist())
        for _ in range(16):
            sim.step(render=False)
            for robot in robots.values():
                robot.update(sim.get_physics_dt())
                assert torch.isfinite(robot.data.joint_pos).all()
        legacy, modern = (report['robots'][name] for name in ('legacy', 'modern'))
        left = dict(zip(legacy['body_names'], legacy['masses']))
        right = dict(zip(modern['body_names'], modern['masses']))
        errors = {name: abs(left[name]-right[name]) for name in left.keys() & right.keys()}
        old_limits = dict(zip(legacy['joint_names'], legacy['velocity_limits']))
        new_limits = dict(zip(modern['joint_names'], modern['velocity_limits']))
        limit_errors = {name: abs(old_limits[name]-new_limits[name]) for name in old_limits}
        if 'pre_fix' in report['robots']:
            previous = report['robots']['pre_fix']
            previous_limits = dict(zip(previous['joint_names'], previous['velocity_limits']))
            report['pre_fix_velocity_limit_errors'] = {name: abs(old_limits[name]-previous_limits[name]) for name in old_limits}
        report['imported_velocity_limits'] = dict(actuator_overrides_disabled=True,
            maximum_absolute_error=max(limit_errors.values()), errors=limit_errors,
            preservation_pass=max(limit_errors.values()) < 1e-6)
        left_ids = {name: index for index, name in enumerate(legacy['body_names'])}
        right_ids = {name: index for index, name in enumerate(modern['body_names'])}
        inertia_errors, com_errors = {}, {}
        for name in left.keys() & right.keys():
            i, j = left_ids[name], right_ids[name]
            inertia_errors[name] = float((torch.tensor(legacy['inertias'][i])-torch.tensor(modern['inertias'][j])).abs().max())
            com_errors[name] = float((torch.tensor(legacy['coms'][i][:3])-torch.tensor(modern['coms'][j][:3])).abs().max())
        report.update(measurement_completed=True, common_body_mass_errors=errors,
                      common_body_inertia_errors=inertia_errors, common_body_com_position_errors=com_errors,
                      missing_modern_bodies=sorted(left.keys()-right.keys()),
                      extra_modern_bodies=sorted(right.keys()-left.keys()),
                      preservation_pass=left.keys() == right.keys() and legacy['joint_names'] == modern['joint_names']
                      and max(errors.values()) < 1e-6 and max(inertia_errors.values()) < 1e-6 and max(com_errors.values()) < 1e-6,
                      steps=16)
        # Keep the strict raw-order result above. Independently exercise the
        # actual named interfaces used by PPO, collection and finger control.
        from types import SimpleNamespace
        from isaaclab.envs import mdp
        from dreamvla_drawer_button.isaac.motion_tracking_open_drawer_env import ActionsCfg, ObservationsCfg
        names = json.loads(args.state_info.read_text())['features']['observation.state']['names']
        assert len(names) == len(set(names)) == 43
        values = {name: (index+1)*0.001 for index, name in enumerate(names)}
        expected_q = torch.tensor([[values[name] for name in names]], device=args.device)
        io = {}
        for label, robot in robots.items():
            env = SimpleNamespace(scene={'robot': robot}, num_envs=1, device=args.device, sim=sim)
            raw = torch.tensor([[values[name] for name in robot.joint_names]], device=args.device)
            robot.write_joint_state_to_sim(raw, torch.zeros_like(raw))
            ids, resolved = robot.find_joints(names, preserve_order=True)
            assert list(resolved) == names
            q = robot.data.joint_pos[:, ids]
            assert torch.equal(q, expected_q), 'Canonical dataset state changed'
            obs_cfg = ObservationsCfg().policy.joint_pos.params['asset_cfg'].copy()
            obs_cfg.resolve(env.scene)
            policy_obs = mdp.joint_pos_rel(env, obs_cfg)
            row = dict(canonical_state=q.cpu().tolist(), ppo_joint_pos=policy_obs.cpu().tolist(), hands={})
            action_cfg = ActionsCfg()
            for side in ('left', 'right'):
                hand_names = [name for name in names if name.startswith(side+'_hand_')]
                hand_ids, resolved = robot.find_joints(hand_names, preserve_order=True)
                assert list(resolved) == hand_names and len(hand_ids) == 7
                cfg = getattr(action_cfg, side+'_hand_action')
                term = cfg.class_type(cfg, env)
                commands = []
                for value in (-1., 1.):
                    term.process_actions(torch.full((1, 1), value, device=args.device))
                    term.apply_actions()
                    commands.append(robot.data.joint_pos_target[0, hand_ids].cpu().tolist())
                row['hands'][side] = commands
            io[label] = row
        # Re-establish identical root poses before comparing named nonzero FK.
        # This uses no physics timestep, so gravity/contact evolution is excluded.
        for index, robot in enumerate(robots.values()):
            root = robot.data.default_root_state.clone()
            root[:, :3] = torch.tensor([index*3., 0., 1.], device=args.device)
            root[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=args.device)
            root[:, 7:] = 0.
            robot.write_root_state_to_sim(root)
        sim.forward()
        compare_live_poses('canonical_nonzero_joint_state')
        assert io['legacy'] == io['modern'], 'Named model/hand interfaces changed'
        report.update(completed=True, named_io_pass=True, named_io=io,
                      raw_joint_order_equal=legacy['joint_names'] == modern['joint_names'])
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2))
        if app is not None:
            # No Replicator writers or render products exist in this fixture.
            # Use the SDK's immediate shutdown after persisting measurements;
            # graceful stage teardown hung in the initial two-articulation test.
            app.close(wait_for_replicator=False, skip_cleanup=True,
                      exit_code=0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
