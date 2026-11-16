"""CPU-only rest-pose repair audit on a separately owned USD wrapper.

This does not initialize Isaac, simulate physics, render, or modify input assets.
Passing this gate does not replace the paired live PhysX pose/dynamics gates.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def load_converter():
    # Import the dependency-light converter module without importing isaaclab.sim,
    # whose package initializer requires a running simulator application.
    directory = Path(__file__).resolve().parents[2]/'Training/source/isaaclab/isaaclab/sim/converters'
    package = types.ModuleType('_rest_pose_fixture')
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__+'.urdf_converter_62', directory/'urdf_converter_62.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_case(urdf, usd, baseline, output):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, preservation_pass=False, scope=__doc__,
                  inputs=dict(urdf=str(urdf), usd=str(usd), baseline=str(baseline)))
    try:
        source, original = Usd.Stage.Open(str(usd)), Usd.Stage.Open(str(baseline))
        if not source or not original or not source.GetDefaultPrim() or not original.GetDefaultPrim():
            raise ValueError('Both inputs must be readable USD assets with default prims')
        layers = {Path(layer.realPath) for stage in (source, original) for layer in stage.GetUsedLayers()
                  if layer.realPath and Path(layer.realPath).is_file()}
        def hashes():
            return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(layers)}
        before_hashes = hashes()
        path = output/'repaired_wrapper.usda'
        stage = Usd.Stage.CreateNew(str(path))
        stage.GetRootLayer().subLayerPaths = [source.GetRootLayer().identifier]
        stage.SetDefaultPrim(stage.GetPrimAtPath(source.GetDefaultPrim().GetPath()))
        stage.GetRootLayer().Save()

        def signature(s):
            return {str(p.GetPath()): dict(type=p.GetTypeName(), schemas=p.GetAppliedSchemas(),
                attributes={a.GetName(): str(a.Get()) for a in p.GetAttributes()
                            if not a.GetName().startswith('xformOp')},
                relationships={r.GetName(): [str(t) for t in r.GetTargets()] for r in p.GetRelationships()},
                instanceable=p.IsInstanceable()) for p in s.Traverse()}

        def matrices(s):
            cache = UsdGeom.XformCache()
            bodies = [p for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len({p.GetName() for p in bodies}) != len(bodies) or not bodies:
                raise ValueError('Rigid-body names must be nonempty and unique')
            return {p.GetName(): np.asarray(cache.GetLocalToWorldTransform(p)).copy() for p in bodies}

        before = signature(stage)
        expected = matrices(original)
        converter = load_converter()
        cfg = types.SimpleNamespace(asset_path=str(urdf))
        converter._preserve_rest_world_poses(path, cfg)
        actual = matrices(stage)
        if expected.keys() != actual.keys():
            raise ValueError('Original and repaired assets have different rigid-body names')
        errors = {name: float(np.max(np.abs(expected[name]-actual[name]))) for name in expected}
        after = signature(stage)
        converter._preserve_rest_world_poses(path, cfg)
        repeated = matrices(stage)
        repeat_error = max(float(np.max(np.abs(actual[n]-repeated[n]))) for n in actual)

        # Parent edits are session-only. The saved output remains a rest asset.
        base_parent = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetDefaultPrim())
        stage.SetEditTarget(stage.GetSessionLayer())
        root = UsdGeom.Xformable(stage.GetDefaultPrim())
        root.AddTranslateOp(opSuffix='probeParent').Set(Gf.Vec3d(1., 2., 3.))
        root.AddRotateZOp(opSuffix='probeParent').Set(67.)
        parent = base_parent.GetInverse() * UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim())
        inherited = matrices(stage)
        parent_error = max(float(np.max(np.abs(inherited[n]-np.asarray(Gf.Matrix4d(actual[n])*parent))))
                           for n in actual)
        after_hashes = hashes()
        report.update(completed=True, wrapper=str(path), body_count=len(actual),
            rest_matrix_errors=errors, maximum_rest_matrix_error=max(errors.values()),
            all_nontransform_properties_and_paths_exact=before == after,
            repeat_application_error=repeat_error, parent_transform_error=parent_error,
            input_layers_sha256=before_hashes, input_layers_unchanged=before_hashes == after_hashes)
        report['preservation_pass'] = (max(errors.values()) <= 1e-6 and before == after
            and repeat_error <= 1e-12 and parent_error <= 1e-12 and before_hashes == after_hashes)
        return report
    finally:
        (output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--usd', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_case(args.urdf, args.usd, args.baseline, args.output)
    print(json.dumps(report))
    if not report['preservation_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
