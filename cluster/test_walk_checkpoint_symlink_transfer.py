import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from walk_adroit_bridge import inventory_script


class CheckpointSymlinkTransferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.root = self.home / 'kevin'
        self.relative = 'checkpoints/test/checkpoint-1'
        self.checkpoint = self.root / self.relative
        self.checkpoint.mkdir(parents=True)
        self.body = b'inference fixture with an immutable canonical identity'
        self.digest = hashlib.sha256(self.body).hexdigest()
        self.canonical = self.root / 'checkpoints/_shared_weights/sha256' / (self.digest + '.safetensors')
        self.canonical.parent.mkdir(parents=True)
        self.canonical.write_bytes(self.body)
        self.link = self.checkpoint / 'model.safetensors'

    def link_to(self, target):
        try:self.link.symlink_to(target)
        except OSError as error:
            if os.name == 'nt' and getattr(error,'winerror',None) == 1314:
                self.skipTest('Run this filesystem test in WSL;Windows symlink privilege unavailable')
            raise

    def inventory(self, inference=True):
        return subprocess.run([sys.executable,'-c',inventory_script(self.relative,inference)],
            env=dict(os.environ,HOME=str(self.home),USERPROFILE=str(self.home)),
            capture_output=True,text=True)

    def test_regular_files_keep_existing_inventory_behavior(self):
        self.link.write_bytes(self.body)
        value=self.inventory(False)
        self.assertEqual(value.returncode,0,value.stderr)
        self.assertEqual(json.loads(value.stdout),[{'path':'model.safetensors','bytes':len(self.body),'sha256':self.digest}])

    def test_canonical_inference_link_is_hashed_as_its_payload(self):
        self.link_to(self.canonical)
        value=self.inventory()
        self.assertEqual(value.returncode,0,value.stderr)
        self.assertEqual(json.loads(value.stdout)[0]['sha256'],self.digest)

    def test_dataset_inventory_still_rejects_links(self):
        self.link_to(self.canonical)
        value=self.inventory(False)
        self.assertNotEqual(value.returncode,0)
        self.assertIn('Unexpected inventory symlink',value.stderr)

    def test_links_outside_owned_store_are_rejected(self):
        other=self.home/'foreign.safetensors';other.write_bytes(self.body)
        self.link_to(other)
        value=self.inventory()
        self.assertNotEqual(value.returncode,0)
        self.assertIn('leaves the owned canonical store',value.stderr)

    def test_metadata_links_remain_rejected(self):
        self.link=self.checkpoint/'statistics.json';self.link_to(self.canonical)
        value=self.inventory()
        self.assertNotEqual(value.returncode,0)
        self.assertIn('Unexpected inventory symlink',value.stderr)

    def test_corrupted_canonical_content_cannot_transfer(self):
        self.link_to(self.canonical);self.canonical.write_bytes(b'corrupt')
        value=self.inventory()
        self.assertNotEqual(value.returncode,0)
        self.assertIn('differs from its SHA256 identity',value.stderr)

    @unittest.skipIf(os.name=='nt','Exercise GNU tar on WSL/Linux')
    def test_tar_export_materializes_independent_verified_files(self):
        self.link_to(self.canonical)
        archive=self.home/'inference.tar';destination=self.home/'destination';destination.mkdir()
        subprocess.run(['tar','--dereference','-C',str(self.root),'-cf',str(archive),'--',self.relative+'/model.safetensors'],check=True)
        subprocess.run(['tar','-C',str(destination),'-xf',str(archive)],check=True)
        result=destination/self.relative/'model.safetensors'
        self.assertFalse(result.is_symlink())
        self.assertEqual(hashlib.sha256(result.read_bytes()).hexdigest(),self.digest)


if __name__=='__main__':unittest.main()
