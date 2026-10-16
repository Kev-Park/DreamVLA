"""Build complete LeRobot statistics and verify the historical consumed values.

The training factory checks every floating-point feature, including unused
features. A cache containing only model-consumed features would be regenerated.
This uses the unchanged GR00T calculation and preserves historical normalization
for every consumed dimension after checking it against the actual parquet data.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    dataset, work, groot = map(Path, sys.argv[1:])
    source = groot / 'gr00t/data/stats.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'calculate_dataset_statistics')
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(np=np, pd=pd, tqdm=tqdm, Path=Path)
    exec(compile(module, str(source), 'exec'), namespace)
    info = json.loads((dataset / 'meta/info.json').read_text())
    features = [key for key, value in info['features'].items() if 'float' in value['dtype']]
    parquets = sorted(dataset.glob('data/*/*.parquet'))
    assert len(parquets) == info['total_episodes'] == 892
    assert info['total_frames'] == 443502
    calculated = namespace['calculate_dataset_statistics'](parquets, features)
    historical = json.loads((work / 'historical_dataset_statistics.json').read_text())['unitree_g1_sonic']
    modality = json.loads((dataset / 'meta/modality.json').read_text())
    differences, consumed = [], {}
    for kind, default in [('state', 'observation.state'), ('action', 'action')]:
        for group, values in historical[kind].items():
            mapping = modality[kind][group]
            key = mapping.get('original_key', default)
            start, end = mapping['start'], mapping['end']
            for statistic, expected in values.items():
                actual = np.asarray(calculated[key][statistic][start:end])
                target = np.asarray(expected)
                assert actual.shape == target.shape and np.isfinite(actual).all()
                delta = float(np.max(np.abs(actual - target)))
                tolerance = 1e-8 if statistic in ('q01', 'q99', 'min', 'max') else 1e-5
                differences.append(dict(feature=key, group=group, statistic=statistic,
                                        max_abs_difference=delta, tolerance=tolerance))
                assert delta <= tolerance, differences[-1]
                # Retain exactly the checkpoint's values even when CPU reductions
                # differ within floating-point tolerance.
                calculated[key][statistic][start:end] = expected
                consumed.setdefault(kind, {}).setdefault(group, {})[statistic] = calculated[key][statistic][start:end]
    assert consumed == {key: historical[key] for key in ('state', 'action')}
    for key in features:
        assert set(calculated[key]) == {'mean', 'std', 'min', 'max', 'q01', 'q99'}
        assert all(np.isfinite(values).all() for values in calculated[key].values())
    current = dataset / 'meta/stats.json'
    backup = work / 'normalization_before_complete_cache.json'
    assert not backup.exists(), 'Preserve the previous verification attempt'
    backup.write_bytes(current.read_bytes())
    temporary = current.with_suffix('.verified')
    temporary.write_text(json.dumps(calculated, indent=2) + '\n')
    temporary.replace(current)
    relative = dataset / 'meta/relative_stats.json'
    assert json.loads(relative.read_text()) == historical.get('relative_action', {})
    request_path = work / 'request.json'
    request = json.loads(request_path.read_text())
    request['dataset_meta_hashes']['meta/stats.json'] = hashlib.sha256(current.read_bytes()).hexdigest()
    receipt = dict(time=time.time(),features=len(features),consumed_values_identical=True,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   stats_sha256=request['dataset_meta_hashes']['meta/stats.json'],
                   differences=differences)
    request['normalization_cache_verification'] = {key: value for key, value in receipt.items() if key != 'differences'}
    temporary = request_path.with_suffix('.verified')
    temporary.write_text(json.dumps(request, indent=2) + '\n')
    temporary.replace(request_path)
    (work / 'normalization_cache_verified.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'differences'}), flush=True)


if __name__ == '__main__':
    main()
