"""Keep saved motion JSON compatible across NumPy 1.x and 2.x."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np


SOURCE = (Path(__file__).resolve().parents[1] /
          'Training/source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib/skeleton.py')
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
names = {'NumpyEncoder', 'json_numpy_obj_hook'}
nodes = [node for node in tree.body
         if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
assert len(nodes) == len(names)
scope = {'np': np, 'json': json}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)


class MotionSerialization(unittest.TestCase):
    def roundtrip(self, data):
        return json.loads(json.dumps(data, cls=scope['NumpyEncoder']),
                          object_hook=scope['json_numpy_obj_hook'])

    def test_motion_arrays_keep_values_shapes_and_dtypes(self):
        # The on-disk ndarray schema must remain readable by legacy motion loaders.
        for dtype in (np.float32, np.float64, np.int32, np.uint64):
            for shape in ((2, 3), (0, 3), ()):
                with self.subTest(dtype=dtype, shape=shape):
                    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
                    result = self.roundtrip({'motion': data})['motion']
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, data.dtype)
                    np.testing.assert_array_equal(result, data)

    def test_scalar_metadata_remains_plain_json_numbers(self):
        for value in (np.int8(-3), np.uint64(2**63 + 7), np.float16(.5), np.float32(.25)):
            with self.subTest(type=type(value)):
                result = self.roundtrip({'value': value})['value']
                self.assertEqual(result, value.item())
                self.assertIs(type(result), int if isinstance(value, np.integer) else float)

    def test_legacy_ndarray_document_still_loads(self):
        result = json.loads('{"__ndarray__":[[1,2],[3,4]],"dtype":"float32","shape":[2,2]}',
                            object_hook=scope['json_numpy_obj_hook'])
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
