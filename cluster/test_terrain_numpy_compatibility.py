"""Exercise actual terrain geometry with the target NumPy/SciPy/Trimesh stack.

Only extract pure mesh functions to avoid launching Kit in a CPU contract test.
This is not a terrain-in-simulation or rendering gate.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.spatial.transform as tf
import trimesh


class TerrainGeometry(unittest.TestCase):
    def test_star_terrain_all_angles_are_finite_and_within_configured_bounds(self):
        folder = (Path(__file__).resolve().parents[1] /
                  'Training/source/isaaclab/isaaclab/terrains/trimesh')
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        for filename, names in [('utils.py', {'make_plane', 'make_border'}),
                                ('mesh_terrains.py', {'star_terrain'})]:
            tree = ast.parse((folder/filename).read_text(encoding='utf-8'))
            nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
        scope = dict(math=math, np=np, tf=tf, trimesh=trimesh)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     'actual_terrain_functions', 'exec'), scope)
        for bars in (4, 5, 7):
            with self.subTest(bars=bars):
                cfg = SimpleNamespace(num_bars=bars, bar_height_range=(.2, .4),
                                      bar_width_range=(.2, .5), size=(8., 8.), platform_width=1.)
                meshes, origin = scope['star_terrain'](.5, cfg)
                self.assertEqual(len(meshes), bars+6)
                np.testing.assert_array_equal(origin, [4., 4., 0.])
                for mesh in meshes:
                    self.assertGreater(len(mesh.faces), 0)
                    self.assertTrue(np.isfinite(mesh.vertices).all())
                    self.assertTrue((mesh.vertices[:, :2] >= -1e-6).all())
                    self.assertTrue((mesh.vertices[:, :2] <= 8.+1e-6).all())
                    self.assertGreaterEqual(float(mesh.vertices[:, 2].min()), -.3-1e-6)
                    self.assertLessEqual(float(mesh.vertices[:, 2].max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
