"""Exercise loss-prevention guards on anonymous USD stages in a running app."""


def run_checks():
    from pxr import Sdf, Usd, UsdGeom
    from legacy_volume_usd import convert_cooked_volumes

    passed = []

    def reject(name, build, expected):
        stage = Usd.Stage.CreateInMemory()
        build(stage)
        before = stage.GetRootLayer().ExportToString()
        try:
            convert_cooked_volumes(stage, None)
        except NotImplementedError as error:
            assert expected in str(error), (name, str(error))
        else:
            raise AssertionError('Expected rejection: ' + name)
        assert before == stage.GetRootLayer().ExportToString(), name
        passed.append(name)

    def schema(stage, name):
        prim = UsdGeom.Mesh.Define(stage, '/Legacy').GetPrim()
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([name]))
        return prim

    def damping(stage):
        prim = schema(stage, 'PhysxDeformableBodyMaterialAPI')
        prim.CreateAttribute('physxDeformableBodyMaterial:dampingScale', Sdf.ValueTypeNames.Float).Set(2.)

    def density(stage):
        prim = schema(stage, 'PhysxDeformableBodyAPI')
        prim.CreateAttribute('physics:density', Sdf.ValueTypeNames.Float).Set(1000.)

    def animated(stage):
        prim = schema(stage, 'PhysxDeformableBodyAPI')
        attr = prim.CreateAttribute('physxDeformable:simulationPoints', Sdf.ValueTypeNames.Point3fArray)
        attr.Set([(0., 0., 0.)], Usd.TimeCode(1))

    reject('nonneutral_damping', damping, 'dampingScale')
    reject('density_without_mass', density, 'density')
    reject('animated_solver_arrays', animated, 'Animated')
    for api in ('PhysxDeformableSurfaceAPI', 'PhysxParticleClothAPI'):
        reject(api, lambda stage, api=api: schema(stage, api), 'Volume-only')
    reject('attachment', lambda stage: stage.DefinePrim('/Attachment', 'PhysxPhysicsAttachment'), 'Volume-only')
    return dict(passed=passed, anonymous_stages_unchanged=True)
