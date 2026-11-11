import hashlib
import io
import json
import unittest
from pathlib import Path
import tempfile
import urllib.error

from upload_adroit_archive_session import acknowledged, transfer, validate_session, source_path


URI = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=test'
DATA = b'x' * (256 * 1024)
DIGEST = hashlib.md5(DATA).hexdigest()


def incomplete(count=None):
    return urllib.error.HTTPError(URI, 308, 'Incomplete',
                                  {} if count is None else {'Range': f'bytes=0-{count-1}'}, None)


def quota_error():
    return urllib.error.HTTPError(URI,403,'Forbidden',{},io.BytesIO(json.dumps(
        {'error':{'errors':[{'reason':'rateLimitExceeded'}]}}).encode()))


class UploadTests(unittest.TestCase):
    def run_upload(self, events, **options):
        requests = []
        def opener(request, **kwargs):
            requests.append(request)
            event = events.pop(0)
            if isinstance(event, Exception):
                raise event
            return io.BytesIO(json.dumps(event).encode())
        result = transfer(io.BytesIO(DATA), URI, len(DATA), DIGEST, lambda _: None,
                          opener=opener, sleep=lambda _: None, chunk=len(DATA), **options)
        return result, requests

    def result(self, **changes):
        return dict(id='verified', size=str(len(DATA)), md5Checksum=DIGEST, **changes)

    def test_fresh_upload_queries_before_sending(self):
        result, requests = self.run_upload([incomplete(), self.result()])
        self.assertEqual(requests[0].data, b'')
        self.assertEqual(requests[1].data, DATA)
        self.assertNotIn('Authorization', requests[1].headers)
        self.assertEqual(result['id'], 'verified')

    def test_resume_sends_only_unacknowledged_bytes(self):
        _, requests = self.run_upload([incomplete(512), self.result()])
        self.assertEqual(requests[1].data, DATA[512:])

    def test_unknown_outcome_queries_before_resend(self):
        _, requests = self.run_upload([incomplete(), TimeoutError(), incomplete(1024), self.result()])
        self.assertEqual(requests[2].data, b'')
        self.assertEqual(requests[3].data, DATA[1024:])

    def test_already_completed_does_not_send_content(self):
        _, requests = self.run_upload([self.result()])
        self.assertEqual(len(requests), 1)

    def test_digest_mismatch_fails(self):
        wrong = self.result();wrong['md5Checksum'] = '0' * 32
        with self.assertRaises(ValueError):self.run_upload([wrong])

    def test_provider_sha256_is_verified_when_required(self):
        sha=hashlib.sha256(DATA).hexdigest()
        result,_=self.run_upload([self.result(sha256Checksum=sha)],expected_sha256=sha)
        self.assertEqual(result['sha256Checksum'],sha)

    def test_provider_sha256_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            self.run_upload([self.result(sha256Checksum='0'*64)],expected_sha256=hashlib.sha256(DATA).hexdigest())

    def test_missing_provider_sha256_cannot_authorize_verification(self):
        with self.assertRaisesRegex(ValueError,'missing provider checksum'):
            self.run_upload([self.result()],expected_sha256=hashlib.sha256(DATA).hexdigest())

    def test_non_retryable_http_failure_is_sanitized(self):
        with self.assertRaisesRegex(RuntimeError, '^Upload HTTP 403; source retained$'):
            self.run_upload([urllib.error.HTTPError(URI, 403, 'Forbidden', {}, None)])

    def test_quota_before_upload_waits_then_queries_same_session(self):
        notices=[]
        _,requests=self.run_upload([quota_error(),incomplete(512),self.result()],
            quota_wait=lambda *args:notices.append(args))
        self.assertEqual(requests[0].data,b'')
        self.assertEqual(requests[1].data,b'')
        self.assertEqual(requests[2].data,DATA[512:])
        self.assertGreaterEqual(notices[0][2],60)

    def test_quota_during_content_queries_before_any_resend(self):
        _,requests=self.run_upload([incomplete(),quota_error(),incomplete(1024),self.result()])
        self.assertEqual(requests[2].data,b'')
        self.assertEqual(requests[3].data,DATA[1024:])

    def test_quota_wait_is_bounded(self):
        with self.assertRaisesRegex(RuntimeError,'quota unavailable repeatedly'):
            self.run_upload([quota_error(),quota_error(),quota_error()],quota_retries=2)

    def test_other_forbidden_reason_is_not_retried(self):
        error=urllib.error.HTTPError(URI,403,'Forbidden',{},io.BytesIO(json.dumps(
            {'error':{'errors':[{'reason':'insufficientPermissions'}]}}).encode()))
        with self.assertRaisesRegex(RuntimeError,'Upload HTTP 403; source retained'):
            self.run_upload([error])

    def test_repeated_unknown_outcomes_are_bounded(self):
        events = [incomplete()]
        for _ in range(5):events.extend([TimeoutError(), incomplete()])
        with self.assertRaisesRegex(RuntimeError, 'failed repeatedly'):
            self.run_upload(events)

    def test_complete_ack_still_requires_final_metadata(self):
        _, requests = self.run_upload([incomplete(len(DATA)), self.result()])
        self.assertTrue(all(r.data == b'' for r in requests))

    def test_acknowledgements_are_bounded(self):
        for value in ('bytes=1-5', 'bytes=0-999999', 'invalid'):
            with self.assertRaises(ValueError):acknowledged({'Range': value}, len(DATA))

    def test_endpoint_cannot_redirect_capability(self):
        for uri in (URI.replace('https:', 'http:'), URI.replace('www.googleapis.com', 'example.com')):
            with self.assertRaises(ValueError):validate_session(uri)

    def test_source_must_stay_in_authorized_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                source_path(Path(directory), {'relative':'../../elsewhere.hdf5'})

    def test_source_identity_must_match_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'cold_storage/pick342_historical_archive_20261020/a.hdf5'
            path.parent.mkdir(parents=True);path.write_bytes(DATA);stat=path.stat()
            row=dict(relative='a.hdf5',bytes=len(DATA),mtime_ns=stat.st_mtime_ns,device=stat.st_dev,inode=stat.st_ino)
            self.assertEqual(source_path(root,row),path)
            row['bytes']+=1
            with self.assertRaises(ValueError):source_path(root,row)


if __name__ == '__main__':
    unittest.main()
