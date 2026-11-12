"""Runtime labels must select the correct Kit settings namespace, not new assets."""
import ast
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'Training/source/isaaclab/isaaclab/app/app_launcher.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AppLauncher')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_resolve_kit_args')
scope = dict(importlib=__import__('importlib'), sys=sys)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), scope)


class RuntimeLabel(unittest.TestCase):
    def resolve(self, version, default=True, args=None):
        owner = SimpleNamespace(_using_default_experience=default)
        with patch.object(sys, 'argv', ['probe']), patch('importlib.metadata.version', return_value=version):
            scope['_resolve_kit_args'](owner, args or {})
            self.assertEqual(sys.argv[1:], owner._kit_args)
            return owner._kit_args

    def test_target_default_uses_runtime_namespace(self):
        self.assertEqual(self.resolve('6.2.0.0'), ['--/app/version=6.2.0'])

    def test_legacy_and_custom_experience_unchanged(self):
        self.assertEqual(self.resolve('5.1.0.0'), [])
        self.assertEqual(self.resolve('6.2.0.0', default=False), [])

    def test_explicit_override_keeps_last_argument_precedence(self):
        self.assertEqual(self.resolve('6.2.0.0', args={'kit_args': '--/app/version=custom --verbose'}),
                         ['--/app/version=6.2.0', '--/app/version=custom', '--verbose'])

    def test_binary_install_without_package_metadata_is_unchanged(self):
        owner = SimpleNamespace(_using_default_experience=True)
        with patch.object(sys, 'argv', ['probe']), patch('importlib.metadata.version',
                    side_effect=importlib.metadata.PackageNotFoundError('isaacsim')):
            scope['_resolve_kit_args'](owner, {})
            self.assertEqual(owner._kit_args, [])


if __name__ == '__main__':
    unittest.main()
