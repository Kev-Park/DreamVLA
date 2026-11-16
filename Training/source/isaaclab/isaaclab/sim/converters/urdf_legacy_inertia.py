# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA urdf-importer-extension, commit
# 9908f0341ee4c375a94d594c4eeb318dea3f4cf2: UrdfParser.cpp and math/core.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Original URDF natural-drive inertia calculation, independent of the SDK.

Preserve the original traversal, transform order, Frobenius inertia magnitude
and float32 arithmetic. Replacing this with an axis-projected effective inertia
would change the original drive recipe. Calculation precedes fixed-link merging.
"""
import math
import struct
import xml.etree.ElementTree as ET


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def _add(a, b):
    return tuple(f32(x+y) for x, y in zip(a, b))


def _scale(a, s):
    return tuple(f32(x*s) for x in a)


def _dot(a, b):
    return f32(f32(f32(a[0]*b[0])+f32(a[1]*b[1]))+f32(a[2]*b[2]))


def _cross(a, b):
    return tuple(f32(f32(a[j]*b[k])-f32(a[k]*b[j])) for j, k in ((1, 2), (2, 0), (0, 1)))


def _rotate(q, v):
    a = _scale(v, f32(f32(f32(2.*q[3])*q[3])-1.))
    b = _scale(_scale(_cross(q[:3], v), q[3]), 2.)
    c = _scale(_scale(q[:3], _dot(q[:3], v)), 2.)
    return _add(_add(a, b), c)


def _qmul(a, b):
    x, y, z, w = a
    u, v, s, t = b
    return (f32(f32(f32(f32(w*u)+f32(t*x))+f32(y*s))-f32(v*z)),
            f32(f32(f32(f32(w*v)+f32(t*y))+f32(z*u))-f32(s*x)),
            f32(f32(f32(f32(w*s)+f32(t*z))+f32(x*v))-f32(u*y)),
            f32(f32(f32(f32(w*t)-f32(x*u))-f32(y*v))-f32(z*s)))


def _compose(a, b):
    return _add(_rotate(a[1], b[0]), a[0]), _qmul(a[1], b[1])


def _inverse(t):
    q = (-t[1][0], -t[1][1], -t[1][2], t[1][3])
    return _scale(_rotate(q, t[0]), -1.), q


def _origin(element):
    origin = element.find('origin') if element is not None else None
    p = tuple(f32(float(x)) for x in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split())
    r, pitch, y = (f32(float(x)) for x in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split())
    cr, cp, cy = [f32(math.cos(f32(x*.5))) for x in (r, pitch, y)]
    sr, sp, sy = [f32(math.sin(f32(x*.5))) for x in (r, pitch, y)]
    mul3 = lambda a, b, c: f32(f32(a*b)*c)
    q = (f32(mul3(cy, sr, cp)-mul3(sy, cr, sp)),
         f32(mul3(cy, cr, sp)+mul3(sy, sr, cp)),
         f32(mul3(sy, cr, cp)-mul3(cy, sr, sp)),
         f32(mul3(cy, cr, cp)+mul3(sy, sr, sp)))
    if len(p) != 3 or not all(math.isfinite(x) for x in (*p, *q)):
        raise ValueError('Invalid URDF inertial/joint origin')
    return p, q


def compute_link_rest_poses(asset_path):
    """Original float32 link-to-robot transforms before fixed-link merging.

    The original importer places link prims directly below the robot root. Its
    accumulated, non-normalized float32 quaternions are part of the authored
    USD contract, including their rounding near identity rotations.
    """
    robot = ET.parse(asset_path).getroot()
    names = [link.get('name') for link in robot.findall('link')]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError('URDF link names must be nonempty and unique')
    links = set(names)
    children, parents = {}, {}
    for joint in robot.findall('joint'):
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        if parent not in links or child not in links or child in parents:
            raise ValueError('URDF joints must form a tree over the declared links')
        parents[child] = parent
        children.setdefault(parent, []).append((child, _origin(joint)))
    roots = links - set(parents)
    if len(roots) != 1:
        raise ValueError('URDF must have exactly one root link')
    poses = {}
    pending = [(next(iter(roots)), ((0., 0., 0.), (0., 0., 0., 1.)))]
    while pending:
        name, pose = pending.pop()
        if name in poses:
            raise ValueError('URDF joint graph contains a cycle')
        poses[name] = pose
        pending.extend((child, _compose(pose, local)) for child, local in children.get(name, []))
    if set(poses) != links:
        raise ValueError('URDF contains disconnected links or a disconnected cycle')
    return poses


def compute_equivalent_inertias(asset_path):
    """Return original pre-merge equivalent inertia for every URDF joint."""
    root = ET.parse(asset_path).getroot()
    links, joints = {}, {}
    for link in root.findall('link'):
        inertial = link.find('inertial')
        tensor = inertial.find('inertia') if inertial is not None else None
        mass = inertial.find('mass') if inertial is not None else None
        values = {key: f32(float(tensor.get(key, '0'))) if tensor is not None else 0.
                  for key in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')}
        links[link.get('name')] = dict(origin=_origin(inertial),
            mass=f32(float(mass.get('value'))) if mass is not None else 0.,
            inertia=((values['ixx'], values['ixy'], values['ixz']),
                     (values['ixy'], values['iyy'], values['iyz']),
                     (values['ixz'], values['iyz'], values['izz'])))
    for element in root.findall('joint'):
        name = element.get('name')
        if name in joints:
            raise ValueError(f'Duplicate URDF joint {name}')
        joints[name] = dict(parent=element.find('parent').get('link'), child=element.find('child').get('link'),
                            origin=_origin(element), children=[])
    child_map = {joint['child']: name for name, joint in joints.items()}
    for name in sorted(joints):  # Original std::map insertion/traversal order.
        joint = joints[name]
        joint['parent_joint'] = child_map.get(joint['parent'], '')
        if joint['parent_joint']:
            joints[joint['parent_joint']]['children'].append(name)
    identity = ((0., 0., 0.), (0., 0., 0., 1.))

    def accumulate(start, forward):
        total = [[0.]*3 for _ in range(3)]
        pending, visited = [(start, identity, forward)], set()
        while pending:
            name, transform, ahead = pending.pop()
            if name in visited or not name:
                continue
            visited.add(name)
            joint = joints[name]
            link = links[joint['child'] if ahead else joint['parent']]
            if ahead:
                distance = _compose(_compose(transform, joint['origin']), link['origin'])[0]
            else:
                distance = _compose(_compose(_inverse(joint['origin']), transform), link['origin'])[0]
            d2 = _dot(distance, distance)
            for i in range(3):
                for j in range(3):
                    shift = f32((d2 if i == j else 0.)-f32(distance[i]*distance[j]))
                    value = f32(link['inertia'][i][j]+f32(link['mass']*shift))
                    total[i][j] = f32(total[i][j]+value)
            if not ahead:
                pending.append((joint['parent_joint'], _compose(_inverse(joint['origin']), transform), False))
            if ahead or name != start:
                for child in joint['children']:
                    pending.append((child, _compose(transform, joint['origin']), True))
        squared = 0.
        for column in range(3):
            for row in range(3):
                squared = f32(squared+f32(total[row][column]*total[row][column]))
        return f32(math.sqrt(squared))

    result = {}
    for name in sorted(joints):
        left, right = accumulate(name, False), accumulate(name, True)
        result[name] = f32(f32(left*right)/f32(left+right)) if left+right > 0 else 0.
    return result
