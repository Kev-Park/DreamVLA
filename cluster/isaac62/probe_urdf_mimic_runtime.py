"""Compare live PhysX response of original/modern imports with both mimic flags.

This owned three-link fixture uses zero gravity, identical hinge drive overrides
and no finger actuator. Robots share the same origin with inter-robot collisions
filtered, avoiding world-placement rounding as a comparison confound. It measures
importer compatibility, not research success.
"""
import argparse
import json
from pathlib import Path
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--modern-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['default', 'mimic_normal'])
    parser.add_argument('--imported-drives', action='store_true', help='Keep imported gains and limits instead of a common hinge stimulus override')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, steps=240, dt=1/120, tolerance_rad=2e-4, robots={})
    app = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationCfg, SimulationContext, UsdFileCfg
        sim = SimulationContext(SimulationCfg(dt=report['dt'], gravity=(0., 0., 0.), device=args.device))
        report['imported_drives'] = args.imported_drives
        robots, ids, traces, pose_traces, offsets = {}, {}, {}, {}, {}
        for version, root in (('legacy', args.legacy_root), ('modern', args.modern_root)):
            for case in args.cases:
                name = version+'_'+case
                offsets[name] = (0., 0., 0.)
                actuator = (ImplicitActuatorCfg(joint_names_expr=['hinge'], stiffness=None, damping=None)
                            if args.imported_drives else ImplicitActuatorCfg(joint_names_expr=['hinge'],
                                stiffness=20., damping=1., effort_limit_sim=10., velocity_limit_sim=2.))
                cfg = ArticulationCfg(prim_path='/World/'+name,
                    spawn=UsdFileCfg(usd_path=str(root/case/'option_fixture.usd')),
                    init_state=ArticulationCfg.InitialStateCfg(pos=offsets[name], joint_pos={'.*': 0.}),
                    actuators={'hinge': actuator})
                robots[name] = Articulation(cfg)
                traces[name] = []
                pose_traces[name] = []
        from pxr import Sdf, UsdPhysics
        # No floor/contact stimulus exists in this fixture. Isolate all robot
        # colliders, including instance proxies, without modifying their assets.
        group = UsdPhysics.CollisionGroup.Define(sim.stage, '/World/FixtureIsolation')
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(
            [Sdf.Path('/World/'+name) for name in robots])
        group.GetCollidersCollectionAPI().CreateExpansionRuleAttr('expandPrims')
        group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])
        report['placement_control'] = dict(shared_origin=True, collision_group=str(group.GetPath()),
            filtered_groups=[str(p) for p in group.GetFilteredGroupsRel().GetTargets()],
            asset_sources_unchanged=True)
        sim.reset()
        for name, robot in robots.items():
            indices, names = robot.find_joints(['hinge', 'finger_hinge'], preserve_order=True)
            assert names == ['hinge', 'finger_hinge'], names
            ids[name] = indices
            report['robots'][name] = dict(joint_names=list(robot.joint_names),
                body_names=list(robot.body_names),
                velocity_limits=robot.root_physx_view.get_dof_max_velocities().cpu().tolist(),
                stiffness=robot.root_physx_view.get_dof_stiffnesses().cpu().tolist(),
                damping=robot.root_physx_view.get_dof_dampings().cpu().tolist(),
                masses=robot.root_physx_view.get_masses().cpu().tolist(),
                inertias=robot.root_physx_view.get_inertias().cpu().tolist(),
                coms=robot.root_physx_view.get_coms().cpu().tolist(),
                initial_link_transforms=robot.root_physx_view.get_link_transforms().cpu().tolist())
        for step in range(report['steps']):
            target = .35 if step < 120 else -.2
            for name, robot in robots.items():
                robot.set_joint_position_target(torch.tensor([[target]], device=args.device), joint_ids=ids[name][:1])
                robot.write_data_to_sim()
            sim.step(render=False)
            for name, robot in robots.items():
                robot.update(report['dt'])
                q = robot.data.joint_pos[:, ids[name]]
                assert torch.isfinite(q).all(), name
                traces[name].append(q[0].cpu().tolist())
                poses = robot.root_physx_view.get_link_transforms()[0].clone()
                poses[:, :3] -= torch.tensor(offsets[name], device=args.device)
                body_ids = [robot.body_names.index(body) for body in ('base', 'arm', 'finger')]
                pose_traces[name].append(poses[body_ids].cpu().tolist())
        comparisons = {}
        for case in args.cases:
            old, new = (torch.tensor(traces[version+'_'+case]) for version in ('legacy', 'modern'))
            error = float((old-new).abs().max())
            left, right = (torch.tensor(pose_traces[version+'_'+case]) for version in ('legacy', 'modern'))
            position_error = float((left[..., :3]-right[..., :3]).abs().max())
            lq, rq = left[..., 3:], right[..., 3:]
            lq = lq/lq.norm(dim=-1, keepdim=True)
            rq = rq/rq.norm(dim=-1, keepdim=True)
            quaternion_error = float(torch.minimum((lq-rq).abs().amax(-1), (lq+rq).abs().amax(-1)).max())
            comparisons[case] = dict(maximum_absolute_error_rad=error,
                maximum_body_position_error_m=position_error, maximum_quaternion_component_error=quaternion_error,
                preservation_pass=error <= report['tolerance_rad'] and position_error <= 2e-4 and quaternion_error <= 1e-4)
        # Prove the fixture actually exercises the enabled/disabled constraints.
        response_gap = None
        if 'default' in args.cases and 'mimic_normal' in args.cases:
            response_gap = float((torch.tensor(traces['legacy_default'])[:, 1]
                                  - torch.tensor(traces['legacy_mimic_normal'])[:, 1]).abs().max())
        hinge_range = {name: max(row[0] for row in trace)-min(row[0] for row in trace)
                       for name, trace in traces.items()}
        report.update(measurement_completed=True, cases=args.cases, traces=traces, body_pose_traces=pose_traces,
            comparisons=comparisons, body_position_tolerance_m=2e-4, quaternion_component_tolerance=1e-4,
            enabled_disabled_finger_response_gap_rad=response_gap, hinge_range_rad=hinge_range)
        assert response_gap is None or response_gap > .01, 'Fixture did not distinguish the two mimic flag states'
        assert all(value > .1 for value in hinge_range.values()), 'Hinge stimulus did not move a fixture'
        assert all(row['preservation_pass'] for row in comparisons.values()), comparisons
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True,
                      exit_code=0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
