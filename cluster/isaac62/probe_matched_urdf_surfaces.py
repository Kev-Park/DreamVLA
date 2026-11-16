"""Compare actual original/target USD surfaces and joint anchors by body name.

This is a matched-import preservation check, separate from the ideal-URDF
geometry check. It does not measure SDK-specific cooked collision hulls.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-usd', type=Path, required=True)
    parser.add_argument('--modern-usd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial import cKDTree

    def extract(path):
        stage = Usd.Stage.Open(str(path))
        assert stage and stage.GetDefaultPrim(), path
        cache = UsdGeom.XformCache()
        bodies = {p.GetName(): p for p in stage.Traverse()
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        root_inv = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(bodies['pelvis'])).T)

        def world(prim):
            return root_inv @ np.asarray(cache.GetLocalToWorldTransform(prim)).T

        surfaces = defaultdict(list)
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            assert body, str(prim.GetPath())
            matrix = world(prim)
            # The 5.1 importer authors mesh CollisionAPI on an ancestor
            # Xform; 6.2 authors it on the Mesh itself. Include the geometry
            # beneath either representation without treating visuals as hulls.
            collision_owner = prim
            while collision_owner and collision_owner != body:
                if collision_owner.HasAPI(UsdPhysics.CollisionAPI):
                    break
                collision_owner = collision_owner.GetParent()
            is_collision = bool(collision_owner and collision_owner != body)
            params = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                      if a.GetName() in ('radius', 'height', 'axis', 'size',
                                        'physics:collisionEnabled', 'physics:approximation')}
            if is_collision:
                params.update({a.GetName(): str(a.Get()) for a in collision_owner.GetAttributes()
                               if a.GetName() in ('physics:collisionEnabled', 'physics:approximation')})
            points = None
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                p = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                points = p @ matrix[:3, :3].T + matrix[:3, 3]
                assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            key = (body.GetName(), prim.GetTypeName(), is_collision)
            surfaces[key].append(dict(path=str(prim.GetPath()), matrix=matrix,
                                      params=params, points=points,
                                      collision_api_on_geometry=prim.HasAPI(UsdPhysics.CollisionAPI)))
        joints = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            anchors = []
            for rel, pos, rot in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                                  (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())):
                targets = rel.GetTargets()
                assert len(targets) <= 1
                body = stage.GetPrimAtPath(targets[0]) if targets else None
                local = Gf.Matrix4d(1)
                local.SetRotate(Gf.Quatd(rot.Get()))
                local.SetTranslateOnly(Gf.Vec3d(pos.Get()))
                matrix = (world(body) if body else np.eye(4)) @ np.asarray(local).T
                anchors.append((body.GetName() if body else None, matrix))
            joints[prim.GetName()] = anchors
        return surfaces, joints

    report = dict(completed=False, scope=__doc__, tolerance=1e-6, surfaces=[], joints=[])
    try:
        (left, lj), (right, rj) = extract(args.legacy_usd), extract(args.modern_usd)
        report['surface_groups'] = {
            label: [{'body': k[0], 'type': k[1], 'collision': k[2], 'count': len(v)}
                    for k, v in sorted(groups.items())]
            for label, groups in [('legacy', left), ('modern', right)]}
        assert left.keys() == right.keys(), 'Surface groups differ'
        assert lj.keys() == rj.keys(), 'Joint names differ'
        for key in sorted(left):
            remaining = list(right[key])
            assert len(left[key]) == len(remaining), key
            for a in left[key]:
                # Namespaces and suffixes changed between importers. Match by
                # actual geometry location, then independently verify its data.
                center = lambda x: x['points'].mean(0) if x['points'] is not None else x['matrix'][:3, 3]
                index = min(range(len(remaining)),
                            key=lambda i: np.linalg.norm(center(a)-center(remaining[i])))
                b = remaining.pop(index)
                if a['points'] is not None:
                    assert b['points'] is not None
                    error = max(float(cKDTree(a['points']).query(b['points'])[0].max()),
                                float(cKDTree(b['points']).query(a['points'])[0].max()))
                else:
                    error = float(np.max(np.abs(a['matrix']-b['matrix'])))
                parameter_errors = {}
                parameters_match = a['params'].keys() == b['params'].keys()
                for name in a['params'].keys() & b['params'].keys():
                    if name in ('radius', 'height', 'size'):
                        delta = abs(float(a['params'][name])-float(b['params'][name]))
                        parameter_errors[name] = delta
                        parameters_match &= np.isfinite(delta) and delta <= 1e-6
                    else:
                        parameters_match &= a['params'][name] == b['params'][name]
                report['surfaces'].append(dict(body=key[0], type=key[1], collision=key[2],
                    legacy=a['path'], modern=b['path'], geometry_error=error,
                    collision_api_on_geometry={
                        'legacy': a['collision_api_on_geometry'],
                        'modern': b['collision_api_on_geometry']},
                    parameters_exact=a['params'] == b['params'],
                    parameter_values={'legacy': a['params'], 'modern': b['params']},
                    parameter_errors=parameter_errors, parameters_match=bool(parameters_match)))
        for name in sorted(lj):
            errors = []
            for a, b in zip(lj[name], rj[name]):
                assert a[0] == b[0], (name, a[0], b[0])
                errors.append(float(np.max(np.abs(a[1]-b[1]))))
            report['joints'].append(dict(name=name, anchor_errors=errors))
        report['preservation_pass'] = all(r['geometry_error'] <= 1e-6 and r['parameters_match']
                                           for r in report['surfaces']) and all(
            max(r['anchor_errors']) <= 1e-6 for r in report['joints'])
        report['completed'] = True
        assert report['preservation_pass'], 'Matched surface/anchor preservation failed'
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
