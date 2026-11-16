"""Separate density mapping equivalence from cross-SDK solver differences.

The mass controls use the same cooked rest mesh and requested density. Compare
their response to density-only bodies within each SDK; report cross-SDK error
separately, without treating finite trajectories as proof of equivalence.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density-run', type=Path, required=True)
    parser.add_argument('--mass-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arrays = {}
    hashes = set()
    counts = []
    for treatment, root in (('density', args.density_run), ('mass', args.mass_run)):
        for sdk in ('legacy', 'candidate'):
            for density in (1000, 1300):
                path = root/(sdk+str(density))
                receipt = json.loads((path/'receipt.json').read_text())
                assert receipt['completed'] and receipt['source_unchanged'] and not receipt.get('error')
                assert receipt['density'] == density
                hashes.add(receipt['source_sha256'])
                counts.append(receipt['source_counts'])
                states = np.load(path/'traces.npz')['states'][..., :3].astype(np.float64)
                assert np.isfinite(states).all() and len(states) == 64
                arrays[treatment, sdk, density] = states-states.mean(axis=-2, keepdims=True)
    assert len(hashes) == 1 and all(c == counts[0] for c in counts)

    def compare(left, right):
        delta = left-right
        return dict(max_error_m=float(np.abs(delta).max()), rms_error_m=float(np.sqrt(np.mean(delta**2))))

    within = {sdk+str(density): compare(arrays['density', sdk, density], arrays['mass', sdk, density])
              for sdk in ('legacy', 'candidate') for density in (1000, 1300)}
    effects = {treatment+'_'+sdk: compare(arrays[treatment, sdk, 1000], arrays[treatment, sdk, 1300])
               for treatment in ('density', 'mass') for sdk in ('legacy', 'candidate')}
    cross = {treatment+str(density): compare(arrays[treatment, 'legacy', density], arrays[treatment, 'candidate', density])
             for treatment in ('density', 'mass') for density in (1000, 1300)}
    atol = 1e-5  # Existing nodal-state position tolerance.
    passed = all(row['max_error_m'] <= atol for row in within.values()) and all(
        row['max_error_m'] > atol for row in effects.values())
    result = dict(passed=passed, scope=__doc__, nodal_position_atol_m=atol, within_sdk=within,
                  positive_density_controls=effects, cross_sdk=cross,
                  source_sha256=next(iter(hashes)), topology=counts[0],
                  cross_sdk_solver_trajectory_parity_claim=False)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    assert passed, 'Density-only/mass-control equivalence not established'


if __name__ == '__main__':
    main()
