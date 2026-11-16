"""Convert the separately obtained G1 URDF using the original SONIC USD settings."""

import argparse
import inspect
from pathlib import Path
import sys
import traceback

from .paths import ASSET_ROOT, BACKEND_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=BACKEND_ROOT
        / "Sim2Real/resources/robots/g1_description/g1_29dof_with_hand.urdf",
    )
    p.add_argument("--output", type=Path, default=ASSET_ROOT / "sonic_g1")
    if "--help" in sys.argv or "-h" in sys.argv:
        p.print_help()
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    if not a.source.is_file():
        raise FileNotFoundError(a.source)
    if a.output.exists():
        raise FileExistsError(
            "Choose a new asset output; conversion does not overwrite assets"
        )
    a.headless = True
    app = AppLauncher(a).app
    completed = False
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        cfg = UrdfConverterCfg(
            asset_path=str(a.source.resolve()),
            usd_dir=str(a.output.resolve()),
            usd_file_name="g1_29dof_with_hand.usd",
            force_usd_conversion=True,
            make_instanceable=True,
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=False,
            collider_type="convex_hull",
            collision_from_visuals=False,
        )
        cfg.joint_drive.drive_type = "force"
        cfg.joint_drive.target_type = "position"
        cfg.joint_drive.gains.stiffness = 0.0
        cfg.joint_drive.gains.damping = 0.0
        usd_path = Path(UrdfConverter(cfg).usd_path)
        if not usd_path.is_file() or usd_path.stat().st_size == 0:
            raise RuntimeError(f"URDF conversion produced no USD: {usd_path}")
        print(usd_path, flush=True)
        completed = True
    except BaseException:
        # Fast SDK shutdown can exit before Python displays the pending error.
        traceback.print_exc()
        raise
    finally:
        if "exit_code" in inspect.signature(app.close).parameters:
            app.close(exit_code=0 if completed else 1)
        else:
            app.close()


if __name__ == "__main__":
    main()
