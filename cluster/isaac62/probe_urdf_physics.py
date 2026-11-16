"""Read-only converted-USD mass, inertia and joint-limit audit against its URDF.

This checks authored rigid-body properties, not collision geometry equivalence
or dynamic trajectory parity. Run with the existing simulator environment.
"""
import argparse
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
    from pxr import Gf, Usd, UsdPhysics

    def rpy(text):
        x, y, z = np.fromstring(text, sep=' ')
        sx, sy, sz = np.sin([x, y, z])
        cx, cy, cz = np.cos([x, y, z])
        return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array(
            [[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    report = dict(completed=False, scope=__doc__, urdf=str(args.urdf), usd=str(args.usd),
                  bodies=[], joints=[], failures=[])
    try:
        robot = ET.parse(args.urdf).getroot()
        stage = Usd.Stage.Open(str(args.usd))
        assert stage and stage.GetDefaultPrim()
        prims = {}
        for prim in stage.Traverse():
            prims.setdefault(prim.GetName(), []).append(prim)
        for link in robot.findall('link'):
            inertial = link.find('inertial')
            if inertial is None:
                continue
            name = link.attrib['name']
            matches = [p for p in prims.get(name, []) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            assert len(matches) == 1, (name, [str(p.GetPath()) for p in matches])
            mass = UsdPhysics.MassAPI(matches[0])
            expected_mass = float(inertial.find('mass').attrib['value'])
            origin = inertial.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rotation = rpy(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0')
            i = inertial.find('inertia').attrib
            expected_inertia = np.array([[float(i['ixx']), float(i['ixy']), float(i['ixz'])],
                                         [float(i['ixy']), float(i['iyy']), float(i['iyz'])],
                                         [float(i['ixz']), float(i['iyz']), float(i['izz'])]])
            expected_inertia = rotation @ expected_inertia @ rotation.T
            axes = Gf.Rotation(mass.GetPrincipalAxesAttr().Get())
            basis = np.stack([axes.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)], axis=1)
            actual_inertia = basis @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ basis.T
            row = dict(name=name, mass_error=abs(float(mass.GetMassAttr().Get())-expected_mass),
                       com_error=float(np.max(np.abs(np.asarray(mass.GetCenterOfMassAttr().Get())-xyz))),
                       inertia_error=float(np.max(np.abs(actual_inertia-expected_inertia))))
            report['bodies'].append(row)
            if row['mass_error'] > 1e-5 or row['com_error'] > 1e-6 or row['inertia_error'] > 1e-6:
                report['failures'].append(dict(kind='body', **row))
        for joint in robot.findall('joint'):
            if joint.attrib['type'] not in ('revolute', 'prismatic'):
                continue
            name = joint.attrib['name']
            matches = [p for p in prims.get(name, []) if p.IsA(UsdPhysics.Joint)]
            assert len(matches) == 1, (name, [str(p.GetPath()) for p in matches])
            revolute = joint.attrib['type'] == 'revolute'
            usd_joint = (UsdPhysics.RevoluteJoint if revolute else UsdPhysics.PrismaticJoint)(matches[0])
            limits = joint.find('limit').attrib
            factor = np.pi/180 if revolute else 1.0
            row = dict(name=name,
                       lower_error=abs(float(usd_joint.GetLowerLimitAttr().Get())*factor-float(limits['lower'])),
                       upper_error=abs(float(usd_joint.GetUpperLimitAttr().Get())*factor-float(limits['upper'])))
            report['joints'].append(row)
            if row['lower_error'] > 1e-6 or row['upper_error'] > 1e-6:
                report['failures'].append(dict(kind='joint', **row))
        assert report['bodies'] and report['joints']
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
