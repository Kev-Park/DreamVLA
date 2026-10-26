"""Small filesystem tests for archive repair safety; no simulation or remote jobs."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from walk_archive_recovery import inventory, plan, promote


class ArchiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.destination = self.root / 'destination'
        self.staging = self.root / 'staging'
        self.backup = self.root / 'backup'
        for path in (self.source, self.destination, self.staging):
            path.mkdir()
        for name, content in [('complete.hdf5', b'complete'),
                              ('damaged.hdf5', b'correct full recording'),
                              ('missing.hdf5', b'previously absent')]:
            (self.source / name).write_bytes(content)
        self.expected = inventory(self.source)
        (self.destination / 'complete.hdf5').write_bytes(b'complete')
        (self.destination / 'damaged.hdf5').write_bytes(b'truncated')
        self.inspected = inventory(self.destination)

    def stage_repairs(self):
        for row in plan(self.expected, self.inspected):
            (self.staging / row['path']).write_bytes((self.source / row['path']).read_bytes())

    def test_repairs_missing_and_damaged_preserving_old_and_complete_files(self):
        self.stage_repairs()
        complete_inode = (self.destination / 'complete.hdf5').stat().st_ino
        result = promote(self.destination, self.staging, self.backup,
                         self.expected, self.inspected)
        self.assertEqual(inventory(self.destination), self.expected)
        self.assertEqual((self.backup / 'damaged.hdf5').read_bytes(), b'truncated')
        self.assertEqual((self.destination / 'complete.hdf5').stat().st_ino, complete_inode)
        self.assertFalse((self.backup / 'complete.hdf5').exists())
        self.assertEqual(result['phase'], 'verified')

    def test_bad_staging_cannot_modify_destination(self):
        self.stage_repairs()
        (self.staging / 'damaged.hdf5').write_bytes(b'bad transfer')
        with self.assertRaises(ValueError):
            promote(self.destination, self.staging, self.backup, self.expected, self.inspected)
        self.assertEqual(inventory(self.destination), self.inspected)
        self.assertFalse(self.backup.exists())

    def test_changed_destination_is_preserved_for_review(self):
        self.stage_repairs()
        (self.destination / 'damaged.hdf5').write_bytes(b'changed after inspection')
        with self.assertRaises(ValueError):
            promote(self.destination, self.staging, self.backup, self.expected, self.inspected)
        self.assertEqual((self.destination / 'damaged.hdf5').read_bytes(), b'changed after inspection')
        self.assertFalse(self.backup.exists())

    def test_unknown_files_are_not_overwritten(self):
        (self.destination / 'unknown.hdf5').write_bytes(b'keep this')
        with self.assertRaises(ValueError):
            plan(self.expected, inventory(self.destination))

    def test_parent_path_is_rejected(self):
        bad = [{'path': '../outside.hdf5', 'bytes': 1,
                'sha256': hashlib.sha256(b'x').hexdigest()}]
        with self.assertRaises(ValueError):
            plan(bad, [])

    def test_matching_archive_needs_no_data_transfer(self):
        self.assertEqual(plan(self.expected, self.expected), [])


if __name__ == '__main__':
    unittest.main()
