"""Original rest-pose tree composition and malformed graph rejection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]/'Training/source/isaaclab/isaaclab/sim/converters/urdf_legacy_inertia.py'
SPEC = importlib.util.spec_from_file_location('legacy_inertia', SOURCE)
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def calculate(text):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'robot.urdf'
        path.write_text('<robot name="fixture">'+text+'</robot>')
        return legacy.compute_link_rest_poses(path)


class RestPoses(unittest.TestCase):
    def test_chain_accumulates_original_float32_translation(self):
        values = calculate('''<link name="root"/><link name="a"/><link name="b"/>
          <joint name="a" type="fixed"><parent link="root"/><child link="a"/><origin xyz="0.1 0 0"/></joint>
          <joint name="b" type="fixed"><parent link="a"/><child link="b"/><origin xyz="0.2 0 0"/></joint>''')
        self.assertEqual(values['root'], ((0., 0., 0.), (0., 0., 0., 1.)))
        self.assertEqual(values['b'][0], (legacy.f32(legacy.f32(.1)+legacy.f32(.2)), 0., 0.))

    def test_disconnected_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disconnected'):
            calculate('''<link name="root"/><link name="a"/><link name="b"/>
              <joint name="a"><parent link="a"/><child link="b"/></joint>
              <joint name="b"><parent link="b"/><child link="a"/></joint>''')

    def test_duplicate_parent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'tree'):
            calculate('''<link name="root"/><link name="a"/>
              <joint name="a"><parent link="root"/><child link="a"/></joint>
              <joint name="b"><parent link="root"/><child link="a"/></joint>''')


if __name__ == '__main__':
    unittest.main()
