import unittest
from unittest.mock import patch
from types import SimpleNamespace

from walk_adroit_quota import parse_quota, read_quota


class QuotaTests(unittest.TestCase):
    def test_raw_blocks_and_asterisk_are_accounted(self):
        row = 'server:/scratch/network 419430400* 410000000 419430400 1793000000 10 100 110'
        value = parse_quota(row)
        self.assertEqual(value['hard_limit_bytes'], 400 * 1024**3)
        self.assertEqual(value['headroom_bytes'], 0)

    def test_missing_scratch_is_not_unlimited_space(self):
        with self.assertRaises(ValueError):
            parse_quota('server:/home 1 10 20')

    def test_limit_exit_code_still_produces_attention(self):
        report = SimpleNamespace(returncode=1, stdout='server:/scratch/network 419430400 410000000 419430400', stderr='')
        with patch('walk_adroit_quota.subprocess.run', return_value=report):
            self.assertIn('attention', read_quota())

    def test_sufficient_headroom_has_no_attention(self):
        report = SimpleNamespace(returncode=0, stdout='server:/scratch/network 300000000 410000000 419430400', stderr='')
        with patch('walk_adroit_quota.subprocess.run', return_value=report):
            self.assertNotIn('attention', read_quota())


if __name__ == '__main__':
    unittest.main()
