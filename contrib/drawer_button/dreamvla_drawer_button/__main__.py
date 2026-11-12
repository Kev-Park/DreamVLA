"""Lightweight dispatcher; heavy dependencies load only for the selected stage."""

import argparse
import runpy
import sys

COMMANDS = {
    "prepare-references": "reference_config",
    "prepare-robot-asset": "asset_robot",
    "prepare-run": "run_config",
    "train-ppo": "train_ppo",
    "record": "rollout",
    "convert": "convert",
    "finetune": "finetune",
    "serve": "serve",
    "refs-button": "generate_button",
    "refs-drawer-stand": "generate_drawer_stand",
    "refs-drawer-squat": "generate_drawer_squat",
    "sample-human": "generate_human",
    "doctor": "doctor",
    "prepare-button-asset": "asset_source",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=sorted(COMMANDS))
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        p.print_help()
        return
    args = p.parse_args(sys.argv[1:2])
    module = "dreamvla_drawer_button." + COMMANDS[args.command]
    sys.argv = [f"{p.prog} {args.command}"] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
