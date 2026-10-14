"""Offline regressions for checkpoint staging with cached immutable shards."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import walk_biped_eval_bridge as bridge


class CachedTransferCapacityTest(unittest.TestCase):
    def test_capacity_and_destination_integrity(self):
        cases = [
            ('cache credit permits transfer', 7, True, True, False),
            ('insufficient physical space', 5, True, False, False),
            ('wrong hash gets no credit', 7, False, False, False),
            ('uncached files need full space', 11, False, True, False),
            ('exact floor remains rejected', 6, True, False, False),
            ('existing destination preserved', 100, True, False, True),
        ]
        for name, extra, matching, should_pass, preexisting in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                home = Path(folder)
                base = home/'kevin'
                cache = base/'checkpoints/walk10_v3_dagger57_20261013_r3/checkpoint-111'
                cache.mkdir(parents=True)
                first = cache/'model-00001-of-00003.safetensors'
                first.write_bytes(b'same' if matching else b'nope')
                (cache/'model-00002-of-00003.safetensors').write_bytes(b'old!')
                expected = [dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                            for path, data in [('model-00001-of-00003.safetensors', b'same'),
                                               ('model-00002-of-00003.safetensors', b'new!'),
                                               ('config.json', b'{}')]]
                relative = 'checkpoints/walk50_v3_dagger57_20261013_r3/checkpoint-999'
                target = base/relative
                if preexisting:
                    target.mkdir(parents=True)
                    (target/first.name).write_bytes(b'keep')

                def local_remote_call(host, script, timeout):
                    self.assertEqual(host, 'biped')
                    self.assertEqual(timeout, 1800)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        exec(compile(script, 'remote_capacity_check', 'exec'), {})
                    return json.loads(output.getvalue())

                with patch.object(bridge, 'call', local_remote_call), \
                     patch.object(Path, 'home', return_value=home), \
                     patch.object(shutil, 'disk_usage', return_value=SimpleNamespace(free=20*2**30+extra)):
                    if not should_pass:
                        with self.assertRaises(AssertionError):
                            bridge.reuse_inference_shards('biped', relative, expected)
                    else:
                        reused = bridge.reuse_inference_shards('biped', relative, expected)
                        self.assertEqual(len(reused), int(matching))
                        if matching:
                            self.assertEqual((target/first.name).stat().st_ino, first.stat().st_ino)
                if preexisting:
                    self.assertEqual((target/first.name).read_bytes(), b'keep')
                elif not should_pass:
                    self.assertFalse(target.exists(), 'Failed checks must not leave partial links')


if __name__ == '__main__':
    unittest.main()
