"""Audit converted rest transforms, joint axes and collision geometry against URDF.

Includes instance proxies. Convex mesh inputs are compared as point sets because
the PhysX convex-hull contract uses their convex hull, not source triangle order.
This does not establish equality of SDK-specific cooked hulls or dynamics.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--usd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from scipy.spatial import cKDTree
    from pxr import Usd, UsdGeom, UsdPhysics

    report = dict(completed=False, scope=__doc__, links=[], axes=[], collisions=[], failures=[])

    def origin(element):
        matrix = np.eye(4)
        if element is not None:
            x, y, z = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
            sx, sy, sz = np.sin([x, y, z])
            cx, cy, cz = np.cos([x, y, z])
            matrix[:3, :3] = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array(
                [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            matrix[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        return matrix

    def apply(matrix, points):
        return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]

    def distance(a, b):
        return max(float(cKDTree(a).query(b)[0].max()), float(cKDTree(b).query(a)[0].max()))

    def gate(kind, row, errors):
        row.update(errors)
        report[kind].append(row)
        if any(not np.isfinite(v) or v > 1e-6 for v in errors.values()):
            report['failures'].append(dict(kind=kind, **row))

    try:
        robot = ET.parse(args.urdf).getroot()
        stage = Usd.Stage.Open(str(args.usd))
        assert stage and stage.GetDefaultPrim()
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        links = {link.get('name'): link for link in robot.findall('link')}
        joints = robot.findall('joint')
        children = {joint.find('child').get('link') for joint in joints}
        roots = set(links)-children
        assert len(roots) == 1
        root = roots.pop()
        expected = {root: np.eye(4)}
        pending = list(joints)
        while pending:
            ready = [joint for joint in pending if joint.find('parent').get('link') in expected]
            assert ready, 'Disconnected or cyclic URDF'
            for joint in ready:
                expected[joint.find('child').get('link')] = expected[joint.find('parent').get('link')] @ origin(joint.find('origin'))
                pending.remove(joint)
        bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        # This fixture exercises the unmerged original G1 contract. The matched
        # SDK5.1 importer retains even empty fixed sensor links as rigid bodies.
        assert set(bodies) == set(links), (set(bodies)-set(links), set(links)-set(bodies))
        link_prims = bodies
        cache = UsdGeom.XformCache()
        world = lambda p: np.asarray(cache.GetLocalToWorldTransform(p)).T
        base_inv = np.linalg.inv(world(bodies[root]))
        for name, prim in link_prims.items():
            gate('links', dict(name=name), dict(transform_error=float(np.max(np.abs(base_inv @ world(prim)-expected[name])))))
        by_name = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
        unit_axes = dict(X=np.array([1., 0, 0]), Y=np.array([0., 1, 0]), Z=np.array([0., 0, 1]))
        for joint in joints:
            if joint.get('type') not in ('revolute', 'prismatic'):
                continue
            name = joint.get('name')
            actual_joint = UsdPhysics.Joint(by_name[name])
            body0 = stage.GetPrimAtPath(actual_joint.GetBody0Rel().GetTargets()[0])
            rotation = actual_joint.GetLocalRot0Attr().Get()
            from pxr import Gf
            basis = np.stack([Gf.Rotation(rotation).TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)], axis=1)
            axis = by_name[name].GetAttribute('physics:axis').Get()
            actual_axis = (base_inv @ world(body0))[:3, :3] @ basis @ unit_axes[axis]
            desired = expected[joint.find('parent').get('link')] @ origin(joint.find('origin'))
            expected_axis = desired[:3, :3] @ np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            expected_axis /= np.linalg.norm(expected_axis)
            anchor = apply(base_inv @ world(body0), np.asarray(actual_joint.GetLocalPos0Attr().Get()))
            gate('axes', dict(name=name), dict(axis_error=float(np.max(np.abs(actual_axis-expected_axis))),
                 anchor_error=float(np.max(np.abs(anchor-desired[:3, 3])))))
        grouped = defaultdict(list)
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            assert body and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            grouped[body.GetName()].append(prim)
        for name, link in links.items():
            source_collisions = link.findall('collision')
            actual_collisions = grouped[name]
            assert len(actual_collisions) == len(source_collisions), (name, len(actual_collisions), len(source_collisions))
            available = list(actual_collisions)
            for index, collision in enumerate(source_collisions):
                geom = collision.find('geometry')[0]
                kind = geom.tag
                candidates = [p for p in available if p.GetTypeName() == {'mesh': 'Mesh', 'sphere': 'Sphere', 'cylinder': 'Cylinder', 'box': 'Cube'}[kind]]
                assert candidates, (name, kind)
                desired = origin(collision.find('origin'))
                # Match repeated primitive shapes by center, not authored suffix.
                relative = lambda p: np.linalg.inv(world(bodies[name])) @ world(p)
                prim = min(candidates, key=lambda p: np.linalg.norm(relative(p)[:3, 3]-desired[:3, 3]))
                available.remove(prim)
                matrix = relative(prim)
                errors = {}
                row = dict(link=name, index=index, type=kind, path=str(prim.GetPath()))
                if kind == 'mesh':
                    mesh_path = args.urdf.parent/geom.get('filename')
                    data = mesh_path.read_bytes()
                    count = int.from_bytes(data[80:84], 'little')
                    assert len(data) == 84+50*count, 'Expected original binary STL'
                    triangles = np.frombuffer(data, dtype=np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]), offset=84)
                    points = triangles['vertices'].reshape(-1, 3).astype(np.float64)
                    points *= np.fromstring(geom.get('scale', '1 1 1'), sep=' ')
                    source_points = apply(desired, points)
                    actual_points = apply(matrix, UsdGeom.Mesh(prim).GetPointsAttr().Get())
                    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull'
                    errors['point_set_error_m'] = distance(source_points, actual_points)
                    row.update(source_sha256=hashlib.sha256(data).hexdigest(), source_triangles=count, usd_points=len(actual_points))
                elif kind == 'box':
                    corners = np.array(list(itertools.product((-0.5, 0.5), repeat=3)))
                    expected_points = apply(desired, corners*np.fromstring(geom.get('size'), sep=' '))
                    actual_points = apply(matrix, corners*UsdGeom.Cube(prim).GetSizeAttr().Get())
                    errors['corner_error_m'] = distance(expected_points, actual_points)
                else:
                    errors['center_error_m'] = float(np.max(np.abs(matrix[:3, 3]-desired[:3, 3])))
                    scales = np.linalg.norm(matrix[:3, :3], axis=0)
                    if kind == 'sphere':
                        radii = scales*UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                        errors['radius_error_m'] = float(np.max(np.abs(radii-float(geom.get('radius')))))
                    else:
                        cylinder = UsdGeom.Cylinder(prim)
                        axis_index = 'XYZ'.index(cylinder.GetAxisAttr().Get())
                        axis = matrix[:3, axis_index]/scales[axis_index]
                        desired_axis = desired[:3, 2]
                        errors['axis_error'] = min(float(np.linalg.norm(axis-desired_axis)), float(np.linalg.norm(axis+desired_axis)))
                        errors['height_error_m'] = abs(scales[axis_index]*cylinder.GetHeightAttr().Get()-float(geom.get('length')))
                        errors['radius_error_m'] = float(np.max(np.abs(np.delete(scales, axis_index)*cylinder.GetRadiusAttr().Get()-float(geom.get('radius')))))
                gate('collisions', row, errors)
            assert not available
        report['counts'] = dict(links=len(report['links']), axes=len(report['axes']), collisions=dict(Counter(row['type'] for row in report['collisions'])))
        report['completed'] = not report['failures']
        assert report['completed'], report['failures']
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
