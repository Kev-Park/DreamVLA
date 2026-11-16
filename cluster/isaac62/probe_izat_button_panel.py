"""Real USD panel composition and independent inherited-clone pose/animation."""
import json
import os
from pathlib import Path
import sys


def main():
    assert os.name == 'posix' and sys.version_info[:2] == (3, 12)
    import numpy as np
    from pxr import Usd, UsdGeom
    from dreamvla_drawer_button.button_visual import VisualButton
    from dreamvla_drawer_button.button_geometry import audit_visual_only
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World/env_0')
    UsdGeom.Xform.Define(stage, '/World/env_1').GetPrim().GetInherits().AddInherit('/World/env_0')
    first = VisualButton(stage, '/World/env_0/VisualButton')
    second = VisualButton(stage, '/World/env_1/VisualButton')
    second_origin = UsdGeom.XformCache().GetLocalToWorldTransform(second.prim).ExtractTranslation()
    first.place([1, 2, 3])
    assert np.array_equal(UsdGeom.XformCache().GetLocalToWorldTransform(second.prim).ExtractTranslation(), second_origin)
    second.place([-1, 2, 3])
    assert np.array_equal(UsdGeom.XformCache().GetLocalToWorldTransform(first.prim).ExtractTranslation(), [1, 2, 3])
    second_cap = np.array(second.cap.Get())
    depth = first.update([1, 2, 3])
    assert depth > 0
    assert np.array_equal(second.cap.Get(), second_cap)
    first.reset()
    assert np.array_equal(first.cap.Get(), first.cap_initial)
    before = stage.GetRootLayer().ExportToString()
    try:
        VisualButton(stage, '/World/env_1/VisualButton')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate local panel was overwritten')
    assert before == stage.GetRootLayer().ExportToString()
    result = dict(completed=True, independent_clone_placement=True,
                  independent_cap_animation=True, duplicate_rejected_without_mutation=True,
                  physics=[audit_visual_only(button.prim) for button in (first, second)])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
