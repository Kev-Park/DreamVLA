# SPDX-License-Identifier: Apache-2.0
"""Metadata-driven migration of cooked legacy volume deformables.

Field mappings and mesh layout follow NVIDIA's Apache-2.0-licensed
omni.physxassetvalidator.scripts.deformableSchemaChecker. Discovery reads authored
API tokens, because removed schemas are absent from GetAppliedSchemas in 6.2.
The installed SDK is never patched. Call only on an isolated flattened stage.
"""
import hashlib
import json
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def authored_apis(prim):
    op = prim.GetMetadata('apiSchemas')
    return set(op.GetAppliedItems()) if op else set()


def _value(prim, name, default=None):
    attr = prim.GetAttribute(name)
    result = attr.Get() if attr else None
    return default if result is None else result


def _set(prim, name, value):
    if value is None:
        return
    attr = prim.GetAttribute(name)
    if not attr or not attr.Set(value):
        raise RuntimeError(f'Cannot author converted property {name} on {prim.GetPath()}')


def _bind_pose(mesh, points):
    prim = mesh.GetPrim()
    prim.ApplyAPI('OmniPhysicsDeformablePoseAPI', 'default')
    prim.CreateAttribute('deformablePose:default:omniphysics:points', Sdf.ValueTypeNames.Point3fArray).Set(points)
    prim.CreateAttribute('deformablePose:default:omniphysics:purposes', Sdf.ValueTypeNames.TokenArray).Set(['bindPose'])


def convert_cooked_volumes(stage, sdk):
    if not stage.GetRootLayer().anonymous or len(stage.GetUsedLayers()) > 2:
        raise ValueError('Migration requires an isolated anonymous flattened stage')
    materials = [p for p in stage.Traverse() if 'PhysxDeformableBodyMaterialAPI' in authored_apis(p)]
    bodies = [p for p in stage.Traverse() if 'PhysxDeformableBodyAPI' in authored_apis(p)]
    for prim in stage.Traverse():
        unsupported = authored_apis(prim) & {'PhysxDeformableSurfaceAPI', 'PhysxParticleClothAPI'}
        if unsupported or prim.GetTypeName() == 'PhysxPhysicsAttachment':
            raise NotImplementedError(f'Volume-only migration cannot preserve {prim.GetPath()}: {unsupported}')
    for prim in bodies:
        density = _value(prim, 'physics:density')
        mass = _value(prim, 'physics:mass')
        if density is not None and (mass is None or mass <= 0):
            raise NotImplementedError('Body-only density without explicit mass cannot be dropped')
        if any(a.GetNumTimeSamples() for a in prim.GetAttributes() if a.GetName().startswith('physxDeformable:')):
            raise NotImplementedError('Animated legacy solver arrays require a separate conversion')
    for prim in materials:
        if _value(prim, 'physxDeformableBodyMaterial:dampingScale', 1.) != 1.:
            raise NotImplementedError('Non-neutral legacy dampingScale has no current equivalent')
        for schema in ('OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI', 'PhysxDeformableMaterialAPI'):
            prim.ApplyAPI(schema)
        defaults = {'density':None, 'dynamicFriction':.25, 'poissonsRatio':.45,
                    'youngsModulus':50000000.*UsdGeom.GetStageMetersPerUnit(stage)/UsdPhysics.GetStageKilogramsPerUnit(stage)}
        for key, default in defaults.items():
            _set(prim, 'omniphysics:'+key, _value(prim, 'physxDeformableBodyMaterial:'+key, default))
        _set(prim, 'omniphysics:staticFriction', _value(prim, 'physxDeformableBodyMaterial:dynamicFriction', .25))
        _set(prim, 'physxDeformableMaterial:elasticityDamping', _value(prim, 'physxDeformableBodyMaterial:elasticityDamping', .005))
        sdk.clean_old_material_apis_and_attrs(prim)
    for prim in bodies:
        path = prim.GetPath()
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f'Legacy volume must be a cooked Mesh: {path}')
        data = {a.GetName():a.Get() for a in prim.GetAttributes()}
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        rest = data.get('physxDeformable:restPoints')
        if rest is None or len(rest) != len(points):rest = points
        # SDK helpers preserve geometry, bindings and transforms. No discovery
        # guard or global SDK state is changed by calling these operations.
        skin = sdk.clone_and_clean_skin_mesh(prim)
        _bind_pose(skin, rest)
        if not sdk.retype_mesh_to_xform(stage, path):
            raise RuntimeError(f'Could not retype legacy Mesh at {path}')
        for schema in ('OmniPhysicsDeformableBodyAPI', 'OmniPhysicsBodyAPI', 'PhysxBaseDeformableBodyAPI', 'MaterialBindingAPI'):
            prim.ApplyAPI(schema)
        mapping = {'deformableEnabled':'omniphysics:deformableBodyEnabled',
                   'kinematicEnabled':'omniphysics:kinematicEnabled',
                   'solverPositionIterationCount':'physxDeformableBody:solverPositionIterationCount',
                   'vertexVelocityDamping':'physxDeformableBody:linearDamping',
                   'sleepDamping':'physxDeformableBody:settlingDamping',
                   'sleepThreshold':'physxDeformableBody:sleepThreshold',
                   'settlingThreshold':'physxDeformableBody:settlingThreshold',
                   'maxDepenetrationVelocity':'physxDeformableBody:maxDepenetrationVelocity',
                   'selfCollision':'physxDeformableBody:selfCollision',
                   'selfCollisionFilterDistance':'physxDeformableBody:selfCollisionFilterDistance',
                   'enableCCD':'physxDeformableBody:enableSpeculativeCCD',
                   'disableGravity':'physxDeformableBody:disableGravity'}
        for old, new in mapping.items():
            _set(prim, new, data.get('physxDeformable:'+old, .005 if old=='vertexVelocityDamping' else None))
        _set(prim, 'omniphysics:mass', data.get('physics:mass'))
        owner = prim.GetRelationship('physxDeformable:simulationOwner')
        if owner and owner.GetTargets():prim.GetRelationship('omniphysics:simulationOwner').SetTargets(owner.GetTargets())
        meshes = {}
        for kind, child in (('simulation','simMesh'), ('collision','collMesh')):
            current = data['physxDeformable:'+kind+'Points']
            rest = data.get('physxDeformable:'+kind+'RestPoints')
            if rest is None or len(rest) != len(current):rest = current
            flat = data['physxDeformable:'+kind+'Indices']
            if not len(current) or not len(flat) or len(flat)%4:
                raise ValueError(f'Invalid cooked {kind} topology on {path}')
            indices = [tuple(flat[i:i+4]) for i in range(0,len(flat),4)]
            mesh = UsdGeom.TetMesh.Define(stage,path.AppendChild(child))
            mesh.CreatePointsAttr(current);mesh.CreateTetVertexIndicesAttr(indices)
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide);_bind_pose(mesh,rest)
            if kind=='simulation':
                mesh.GetPrim().ApplyAPI('OmniPhysicsVolumeDeformableSimAPI')
                _set(mesh.GetPrim(),'omniphysics:restShapePoints',rest)
                _set(mesh.GetPrim(),'omniphysics:restTetVtxIndices',indices)
                velocities = data.get('physxDeformable:simulationVelocities')
                if velocities is not None and len(velocities)==len(current):mesh.CreateVelocitiesAttr(velocities)
            else:
                mesh.CreateSurfaceFaceVertexIndicesAttr(UsdGeom.TetMesh.ComputeSurfaceFaces(mesh,Usd.TimeCode.Default()))
                mesh.GetPrim().ApplyAPI('PhysicsCollisionAPI');mesh.GetPrim().ApplyAPI('PhysxCollisionAPI')
                for key in ('contactOffset','restOffset'):
                    _set(mesh.GetPrim(),'physxCollision:'+key,data.get('physxCollision:'+key))
            meshes[kind] = mesh
        sdk.clean_old_deformable_apis_and_attrs(prim)
    return dict(body_count=len(bodies),material_count=len(materials),discovery='authored API metadata')


def convert_copy(source, destination):
    """Convert a cooked volume USD to a new file; requires a running Isaac app."""
    from pxr import Usd
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('omni.physx.asset_validator')
    from omni.physxassetvalidator.scripts.deformableSchemaChecker import DeformableSchemaChecker
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or source.resolve()==destination.resolve():
        raise ValueError('Conversion requires a new output file')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    original=Usd.Stage.Open(str(source))
    if original is None:raise ValueError('Cannot open input USD')
    # The SDK rewrites editable layer stacks. Flatten into an anonymous layer
    # so referenced assets and the source file cannot receive those edits.
    stage=Usd.Stage.Open(original.Flatten())
    # Keep authored versus registered API discovery visible: removed USD schemas
    # may remain in file metadata while GetAppliedSchemas omits them.
    for label, current in (("original", original), ("flattened", stage)):
        rows=[]
        for prim in current.Traverse():
            authored=prim.GetMetadata("apiSchemas")
            if "Deformable" in str(authored):
                rows.append(dict(path=str(prim.GetPath()), authored=str(authored),
                                 registered=list(prim.GetAppliedSchemas())))
        print(json.dumps({"schema_discovery":label,"rows":rows}),flush=True)
    old=[]
    for prim in stage.Traverse():
        schemas=authored_apis(prim)
        if 'PhysxDeformableBodyMaterialAPI' in schemas:
            attr=prim.GetAttribute('physxDeformableBodyMaterial:dampingScale')
            if attr and attr.HasAuthoredValueOpinion() and attr.Get()!=1.:
                raise NotImplementedError('Non-neutral dampingScale is removed by the SDK converter')
        if 'PhysxDeformableBodyAPI' not in schemas:continue
        density=prim.GetAttribute('physics:density');mass=prim.GetAttribute('physics:mass')
        if density and density.HasAuthoredValueOpinion() and (not mass or not mass.Get() or mass.Get()<0):
            raise NotImplementedError('SDK conversion drops body-only density without explicit mass')
        def size(name):
            value=prim.GetAttribute(name).Get()
            return len(value) if value is not None else 0
        row=dict(path=str(prim.GetPath()),simulation_nodes=size('physxDeformable:simulationPoints'),
                 simulation_elements=size('physxDeformable:simulationIndices')//4,
                 collision_nodes=size('physxDeformable:collisionPoints'),
                 collision_elements=size('physxDeformable:collisionIndices')//4)
        if min(row[k] for k in row if k!='path')<=0:
            raise ValueError('Legacy asset must contain cooked simulation and collision meshes')
        old.append(row)
    if not old:raise ValueError('No legacy volume deformable found')
    conversion=convert_cooked_volumes(stage,DeformableSchemaChecker)
    for row in old:
        prim=stage.GetPrimAtPath(row['path'])
        if not prim.HasAPI('OmniPhysicsDeformableBodyAPI') or 'PhysxDeformableBodyAPI' in authored_apis(prim):
            raise RuntimeError('SDK failed to convert legacy body '+row['path'])
    if not stage.GetRootLayer().Export(str(destination)):
        raise RuntimeError('Could not export converted copy')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Source USD changed during isolated conversion')
    return dict(source_sha256=digest,legacy_bodies=old,converter='metadata-driven volume conversion with installed SDK geometry helpers', conversion=conversion)
