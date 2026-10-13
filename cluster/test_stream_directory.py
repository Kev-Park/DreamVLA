import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from stream_directory import inventory,prepare,send,receive,rooted,sha


class StreamTests(unittest.TestCase):
    def fixture(self,base):
        home=Path(base);src=home/'kevin/source';src.mkdir(parents=True)
        (src/'nested').mkdir();(src/'nested/data.bin').write_bytes(b'payload'*200)
        (src/'info.json').write_text('{}')
        return home,src

    def test_round_trip_preserves_bytes_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            home,src=self.fixture(temp)
            with patch.object(Path,'home',return_value=home):
                manifest=inventory('source','roundtrip',False)
                prep=prepare('destination',manifest);wire=io.BytesIO()
                with patch.object(sys,'stdout',SimpleNamespace(buffer=wire)):send('roundtrip',prep['omitted'])
                wire.seek(0)
                with patch.object(sys,'stdin',SimpleNamespace(buffer=wire)):receipt=receive('destination','roundtrip')
                self.assertEqual(receipt['verified_files'],2)
                self.assertEqual((home/'kevin/destination/nested/data.bin').read_bytes(),(src/'nested/data.bin').read_bytes())
                with self.assertRaises(AssertionError):prepare('destination',manifest)
                with self.assertRaises(ValueError):rooted('../escape')

    def test_corruption_never_publishes_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            home,_=self.fixture(temp)
            with patch.object(Path,'home',return_value=home):
                manifest=inventory('source','corrupt',False);prep=prepare('destination',manifest)
                wire=io.BytesIO()
                with patch.object(sys,'stdout',SimpleNamespace(buffer=wire)):send('corrupt',prep['omitted'])
                damaged=io.BytesIO(wire.getvalue().replace(b'payloadpayload',b'Xayloadpayload',1))
                with patch.object(sys,'stdin',SimpleNamespace(buffer=damaged)),self.assertRaises(AssertionError):receive('destination','corrupt')
                self.assertFalse((home/'kevin/destination').exists())

    @unittest.skipIf(os.name=='nt','Receiving host uses Linux symlinks')
    def test_inference_omits_optimizer_and_reuses_identical_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            home,src=self.fixture(temp)
            (src/'optimizer.pt').write_bytes(b'optimizer')
            model=src/'model.safetensors';model.write_bytes(b'weights')
            store=home/'kevin/checkpoints/_shared_weights/sha256';store.mkdir(parents=True)
            canonical=store/(sha(model)+'.safetensors');canonical.write_bytes(model.read_bytes())
            with patch.object(Path,'home',return_value=home):
                manifest=inventory('source','shared',True);self.assertNotIn('optimizer.pt',manifest['files'])
                prep=prepare('destination',manifest);self.assertEqual(prep['omitted'],['model.safetensors'])
                wire=io.BytesIO()
                with patch.object(sys,'stdout',SimpleNamespace(buffer=wire)):send('shared',prep['omitted'])
                wire.seek(0)
                with patch.object(sys,'stdin',SimpleNamespace(buffer=wire)):receive('destination','shared')
                self.assertTrue((home/'kevin/destination/model.safetensors').is_symlink())
                self.assertEqual((home/'kevin/destination/model.safetensors').resolve(),canonical)


if __name__=='__main__':unittest.main()
