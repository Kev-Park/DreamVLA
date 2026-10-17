"""Incomplete or duplicated collection must never become a training dataset."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'adroit'))
from build_walk200_round3_dataset import group_coverage


class GroupCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.checkpoint = '/home/example/kevin/checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967'
        manifest = self.work / 'adroit_round3_inputs.ready.json'
        self.write(manifest, {'student_checkpoint_relative': 'checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967'})
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for group in range(4):
            directory = self.work / 'adroit_round3_groups' / str(group)
            expected = list(range(group * 50, (group + 1) * 50))
            common = {'group': group, 'motion_ids': expected, 'checkpoint': self.checkpoint,
                      'slurm_job_id': str(100 + group)}
            self.write(directory / 'group.launch.json', dict(common, source_manifest_sha256=sha))
            self.write(directory / 'group.done.json', dict(common, student_attempts=50, errors=0))
            for shard in range(2):
                text = ''
                for motion in expected[shard::2]:
                    text += (f'[INFO] motion {motion}_r0/1\n'
                             f'[rescue] motion {motion}: pass A 4-rule\n'
                             f'[rescue] WROTE rescue 1 -> sonic_adapter__Pick_sim2__motion_{motion:03d}_r0.hdf5\n')
                text += '[INFO] Collection finished. errored=0\n'
                (directory / f'collect_r3_shard{shard}.log').write_text(text)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_exact_complete_coverage(self):
        result = group_coverage(self.work)
        self.assertEqual(result['written_ids'], list(range(200)))
        self.assertEqual(result['slurm_jobs'], ['100', '101', '102', '103'])

    def test_valid_no_grasp_skip_still_counts_as_attempt(self):
        log = self.work / 'adroit_round3_groups/0/collect_r3_shard0.log'
        log.write_text(log.read_text().replace(
            '[rescue] WROTE rescue 1 -> sonic_adapter__Pick_sim2__motion_000_r0.hdf5',
            '[rescue] SKIP motion 0: no reference grasp frame'))
        result = group_coverage(self.work)
        self.assertEqual(result['skipped_ids'], [0])
        self.assertEqual(len(result['written_ids']), 199)

    def test_duplicate_disposition_is_rejected(self):
        log = self.work / 'adroit_round3_groups/0/collect_r3_shard0.log'
        log.write_text(log.read_text() + '[rescue] SKIP motion 0: no reference grasp frame\n')
        with self.assertRaises(AssertionError):
            group_coverage(self.work)

    def test_missing_group_or_aborted_group_is_rejected(self):
        path = self.work / 'adroit_round3_groups/3/group.done.json'
        prior = path.read_text()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            group_coverage(self.work)
        path.write_text(prior)
        self.write(path.parent / 'group.abort.json', {'error': 'worker failed'})
        with self.assertRaises(AssertionError):
            group_coverage(self.work)

    def test_foreign_student_and_changed_input_manifest_are_rejected(self):
        for name in ['group.launch.json', 'group.done.json']:
            path = self.work / 'adroit_round3_groups/0' / name
            row = json.loads(path.read_text())
            row['checkpoint'] = '/different/run/checkpoint-5000'
            self.write(path, row)
        with self.assertRaises(AssertionError):
            group_coverage(self.work)
        path = self.work / 'adroit_round3_inputs.ready.json'
        row = json.loads(path.read_text())
        row['changed'] = True
        self.write(path, row)
        with self.assertRaises(AssertionError):
            group_coverage(self.work)

    def test_worker_error_or_incomplete_summary_is_rejected(self):
        log = self.work / 'adroit_round3_groups/0/collect_r3_shard0.log'
        log.write_text(log.read_text().replace('errored=0', 'errored=1'))
        with self.assertRaises(ValueError):
            group_coverage(self.work)


if __name__ == '__main__':
    unittest.main()
