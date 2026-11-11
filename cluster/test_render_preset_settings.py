"""Rendering preset paths and values must survive the 6.2 dependency cleanup."""
import ast
from collections.abc import Mapping
from pathlib import Path
import tomllib
from typing import Any
import unittest


ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'Training/source/isaaclab/isaaclab/utils/dict.py'
node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
            if isinstance(n, ast.FunctionDef) and n.name == 'flatten_dict')
scope = {'Mapping': Mapping, 'Any': Any}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
flatten = scope['flatten_dict']


class RenderingPresets(unittest.TestCase):
    def test_nested_values_keep_types_and_array_settings(self):
        values = {'rtx': {'enabled': False, 'count': 2, 'gain': .25, 'color': [1., 0., 0.]},
                  'rtx-transient': {'label': 'balanced'}}
        self.assertEqual(flatten(values),
                         {'rtx.enabled': False, 'rtx.count': 2, 'rtx.gain': .25,
                          'rtx.color': [1., 0., 0.], 'rtx-transient.label': 'balanced'})

    def test_actual_presets_keep_every_authored_leaf_setting(self):
        files = list((ROOT/'Training/apps/rendering_modes').glob('*.kit'))
        self.assertTrue(files)
        for file in files:
            # These presets author dotted scalar keys, so the source provides an
            # independent expected path set instead of reimplementing recursion.
            authored = {line.split('=', 1)[0].strip() for line in file.read_text().splitlines()
                        if '=' in line and not line.lstrip().startswith('#')}
            with file.open('rb') as stream:
                result = flatten(tomllib.load(stream))
            with self.subTest(preset=file.name):
                self.assertEqual(set(result), authored)
                self.assertIs(result['rtx.shadows.enabled'], True)
                self.assertIsInstance(result['rtx.post.dlss.execMode'], int)


if __name__ == '__main__':
    unittest.main()
