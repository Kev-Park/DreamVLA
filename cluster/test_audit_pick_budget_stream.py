import hashlib
import io
import json
import tarfile
import unittest

import numpy as np
from audit_pick_budget_stream import audit_stream


class TrajectoryAuditTests(unittest.TestCase):
    def stream(self, missing=False, corrupt=False):
        f = io.BytesIO()
        root = np.tile([0.01, 0, 0], (3, 1))
        quat = np.tile([1., 0, 0, 0], (3, 1))
        np.savez(f, body_names=np.array(['pelvis']), body_pos=root[:, None],
                 ref_body_pos=np.zeros((3, 1, 3)), root_pos=root, ref_root_pos=np.zeros((3, 3)),
                 root_quat=quat, obj_pos=np.zeros((3, 3)), obj_quat=quat,
                 wrist=np.tile(np.eye(4), (3, 1, 1)))
        content = f.getvalue(); run = 'test_run'
        names = [f'{run}/b0_ep{i}_m{i}_traj.npz' for i in range(100)]
        entry = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        report = {'episodes': 100, 'successes': 0, 'world_mpjpe_cm': 1.,
                  'root_aligned_mpjpe_cm': 0., 'per_episode': [
                      {'motion_id': i, 'success': False, 'world_mpjpe_cm': 1.,
                       'root_aligned_mpjpe_cm': 0., 'frames': 3, 'bodies': 1} for i in range(100)]}
        manifest = {'runs': {run: {'report': report, 'box': {'lo': [1, 1, 1], 'hi': [2, 2, 2]}}},
                    'files': {name: entry for name in names}}
        wire = io.BytesIO()
        with tarfile.open(fileobj=wire, mode='w|') as archive:
            data = json.dumps(manifest).encode()
            member = tarfile.TarInfo('manifest.json'); member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            for i, name in enumerate(names[:-1] if missing else names):
                data = b'X' + content[1:] if corrupt and i == 0 else content
                member = tarfile.TarInfo(name); member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        wire.seek(0)
        return wire

    def test_known_translation_and_root_alignment(self):
        report = audit_stream(self.stream())
        self.assertEqual(report['verified_trajectories'], 100)
        self.assertEqual(report['runs'][0]['world_mpjpe_cm'], 1.)
        self.assertEqual(report['runs'][0]['root_aligned_mpjpe_cm'], 0.)
        self.assertEqual(report['trajectory_copies_stored'], 0)

    def test_rejects_missing_and_corrupt_episodes(self):
        for kwargs in [{'missing': True}, {'corrupt': True}]:
            with self.assertRaises(AssertionError):
                audit_stream(self.stream(**kwargs))


if __name__ == '__main__':
    unittest.main()
