"""Compare native and migrated bound-material density controls, without solver parity claims."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-run', type=Path, required=True)
    parser.add_argument('--target-run', type=Path, required=True)
    parser.add_argument('--scaled', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    traces, receipts = {}, {}
    densities = (1300,) if args.scaled else (1000, 1300)
    for sdk, root in (('legacy', args.legacy_run), ('candidate', args.target_run)):
        for treatment in ('density', 'mass'):
            for density in densities:
                label = (sdk+'_' if args.scaled else '')+treatment+str(density)
                path = root/label
                receipt = json.loads((path/'receipt.json').read_text())
                assert receipt['completed'] and not receipt.get('error') and receipt['source_unchanged']
                assert receipt['bound_material'] and not receipt['public_material_view_none']
                expected_scale = 1.2*.9*1.1 if args.scaled else 1.
                assert np.isclose(receipt.get('body_world_volume_scale', 1.), expected_scale)
                expected_mass = receipt['simulation_rest_volume_m3']*density
                assert np.isclose(receipt['explicit_mass_kg'], expected_mass if treatment == 'mass' else 0.)
                if sdk == 'candidate':
                    assert receipt['public_converter_without_post_edits']
                    assert 'later_nonneutral_damping' in receipt['atomic_rejections']['passed']
                    inferred = receipt['candidate_conversion']['conversion']['density_inferred_masses']
                    if treatment == 'density':
                        assert len(inferred) == 1 and np.isclose(next(iter(inferred.values())), expected_mass)
                    else:
                        assert not inferred
                states = np.load(path/'traces.npz')['states'][..., :3].astype(np.float64)
                assert len(states) == 64 and np.isfinite(states).all()
                traces[sdk, treatment, density] = states-states.mean(axis=-2, keepdims=True)
                receipts[sdk, treatment, density] = receipt
    assert len({r['source_sha256'] for r in receipts.values()}) == 1
    assert all(r['source_counts'] == next(iter(receipts.values()))['source_counts'] for r in receipts.values())
    within = {sdk+str(density): float(np.abs(traces[sdk, 'density', density]-traces[sdk, 'mass', density]).max())
              for sdk in ('legacy', 'candidate') for density in densities}
    cross = {treatment+str(density): float(np.abs(
        traces['legacy', treatment, density]-traces['candidate', treatment, density]).max())
        for treatment in ('density', 'mass') for density in densities}
    result = dict(passed=all(v <= 1e-5 for v in within.values()), position_atol_m=1e-5,
                  within_sdk_max_m=within, cross_sdk_max_m=cross,
                  scaled=args.scaled, cross_sdk_trajectory_parity_claim=False)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    assert result['passed'], 'Bound density mapping does not match explicit mass controls'


if __name__ == '__main__':
    main()
