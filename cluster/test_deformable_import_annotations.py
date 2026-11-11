"""Removed optional soft-body annotations must not block rigid-task imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class OptionalAssetImport(unittest.TestCase):
    def test_data_class_can_load_without_removed_legacy_tensor_types(self):
        path = (Path(__file__).resolve().parents[1] /
                'Training/source/isaaclab/isaaclab/assets/deformable_object/deformable_object_data.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        # Execute the actual class and its annotation policy without importing Kit.
        # The 6.2 tensor API provides DeformableBodyView, not SoftBodyView.
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) or
                 isinstance(node, ast.ImportFrom) and node.module == '__future__']
        scope = {'physx': SimpleNamespace(DeformableBodyView=object),
                 'torch': SimpleNamespace(Tensor=object)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        self.assertIn('DeformableObjectData', scope)
        # This is an import gate only; it does not validate deformable simulation.


if __name__ == '__main__':
    unittest.main()
