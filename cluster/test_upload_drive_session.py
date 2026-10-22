import hashlib
import io
import json
import unittest
import urllib.error

from upload_drive_session import acknowledged, transfer, validate_session


URI = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=test'
DATA = b'x' * (256 * 1024)
DIGEST = hashlib.md5(DATA).hexdigest()


def incomplete(count=None):
    return urllib.error.HTTPError(URI, 308, 'Incomplete',
                                  {} if count is None else {'Range': f'bytes=0-{count-1}'}, None)


class UploadTests(unittest.TestCase):
    def run_upload(self, events):
        requests = []
        def opener(request, **kwargs):
            requests.append(request)
            event = events.pop(0)
            if isinstance(event, Exception):
                raise event
            return io.BytesIO(json.dumps(event).encode())
        result = transfer(io.BytesIO(DATA), URI, len(DATA), DIGEST, lambda _: None,
                          opener=opener, sleep=lambda _: None, chunk=len(DATA))
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

    def test_non_retryable_http_failure_is_sanitized(self):
        with self.assertRaisesRegex(RuntimeError, '^Upload HTTP 403; source retained$'):
            self.run_upload([urllib.error.HTTPError(URI, 403, 'Forbidden', {}, None)])

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


if __name__ == '__main__':
    unittest.main()
