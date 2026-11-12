"""Write a separate Isaac 6.2 USD copy of cooked legacy volume deformables.

Run in the approved Isaac environment on a cluster, never on the local desktop.
Surface/cloth, attachments, animated solver arrays and lossy material mappings
are rejected. The original USD and referenced layers are never edited.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error('Destination already exists; choose a new file')
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, active_gpu=0, physics_gpu=0,
                             multi_gpu=False, fast_shutdown=True))
    try:
        from legacy_volume_usd import convert_copy
        receipt = convert_copy(args.source, args.destination)
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
