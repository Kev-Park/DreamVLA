"""Compare live PhysX response of original/modern imports with both mimic flags.

This owned three-link fixture uses zero gravity, identical hinge drive overrides
and no finger actuator. It measures importer compatibility, not research success.
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
        robots, ids, traces = {}, {}, {}
        for version, root in (('legacy', args.legacy_root), ('modern', args.modern_root)):
            for case in ('default', 'mimic_normal'):
                name = version+'_'+case
                cfg = ArticulationCfg(prim_path='/World/'+name,
                    spawn=UsdFileCfg(usd_path=str(root/case/'option_fixture.usd')),
                    init_state=ArticulationCfg.InitialStateCfg(pos=(len(robots)*2., 0., 0.), joint_pos={'.*': 0.}),
                    actuators={'hinge': ImplicitActuatorCfg(joint_names_expr=['hinge'],
                        stiffness=20., damping=1., effort_limit_sim=10., velocity_limit_sim=2.)})
                robots[name] = Articulation(cfg)
                traces[name] = []
        sim.reset()
        for name, robot in robots.items():
            indices, names = robot.find_joints(['hinge', 'finger_hinge'], preserve_order=True)
            assert names == ['hinge', 'finger_hinge'], names
            ids[name] = indices
            report['robots'][name] = dict(joint_names=list(robot.joint_names),
                body_names=list(robot.body_names),
                velocity_limits=robot.root_physx_view.get_dof_max_velocities().cpu().tolist())
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
        comparisons = {}
        for case in ('default', 'mimic_normal'):
            old, new = (torch.tensor(traces[version+'_'+case]) for version in ('legacy', 'modern'))
            error = float((old-new).abs().max())
            comparisons[case] = dict(maximum_absolute_error_rad=error,
                preservation_pass=error <= report['tolerance_rad'])
        # Prove the fixture actually exercises the enabled/disabled constraints.
        response_gap = float((torch.tensor(traces['legacy_default'])[:, 1]
                              - torch.tensor(traces['legacy_mimic_normal'])[:, 1]).abs().max())
        hinge_range = {name: max(row[0] for row in trace)-min(row[0] for row in trace)
                       for name, trace in traces.items()}
        report.update(measurement_completed=True, traces=traces, comparisons=comparisons,
            enabled_disabled_finger_response_gap_rad=response_gap, hinge_range_rad=hinge_range)
        assert response_gap > .01, 'Fixture did not distinguish the two mimic flag states'
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
