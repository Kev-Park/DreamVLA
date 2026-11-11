"""Dataset metadata validation without loading the converter environment."""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name('merge_lerobot.py')
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
names = {'select_episodes', 'validate_catalog', 'validate_modalities'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == len(names)
scope = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)


class DatasetCatalog(unittest.TestCase):
    def fixture(self):
        episodes = [{'episode_index': 0, 'length': 3, 'tasks': ['pick']},
                    {'episode_index': 1, 'length': 5, 'tasks': ['pick']}]
        stats = [{'episode_index': 0, 'stats': {'action.token': {'min': [1]}}},
                 {'episode_index': 1, 'stats': {'action.token': {'min': [2]}}}]
        info = {'total_episodes': 2, 'total_frames': 8}
        return episodes, stats, info

    def test_subset_retains_requested_order_and_original_label_statistics(self):
        episodes, stats, info = self.fixture()
        selected, lookup = scope['validate_catalog'](episodes, stats, info, [1, 0])
        self.assertEqual([e['episode_index'] for e in selected], [1, 0])
        self.assertEqual(lookup[1], stats[1]['stats'])

    def test_unselected_catalog_is_sorted_without_silent_duplicate_episodes(self):
        episodes, _, _ = self.fixture()
        ordered = scope['select_episodes'](list(reversed(episodes)), None)
        self.assertEqual([e['episode_index'] for e in ordered], [0, 1])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            scope['select_episodes']([episodes[0], episodes[0]], None)

    def test_invalid_or_missing_episode_selections_are_rejected(self):
        episodes, _, _ = self.fixture()
        for indices in ([], [1, 1], [True], [-1], [2]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                scope['select_episodes'](episodes, indices)

    def test_catalog_totals_and_lengths_must_describe_complete_source(self):
        for change in ('length', 'total_frames', 'total_episodes'):
            episodes, stats, info = self.fixture()
            if change == 'length':
                episodes[0]['length'] = 0
            else:
                info[change] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                scope['validate_catalog'](episodes, stats, info)

    def test_statistics_cannot_be_missing_duplicated_or_for_another_episode(self):
        for change in ('missing', 'duplicate', 'wrong_id'):
            episodes, stats, info = self.fixture()
            if change == 'missing':
                stats.pop()
            elif change == 'duplicate':
                stats.append(stats[0])
            else:
                stats[0]['episode_index'] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                scope['validate_catalog'](episodes, stats, info)

    def test_modality_mismatches_and_partial_metadata_are_rejected(self):
        first = {'state': {'joint_pos': {'start': 0, 'end': 29}}}
        scope['validate_modalities']([first, first])
        scope['validate_modalities']([None, None])
        for values in ([first, None], [None, first], [first, {'state': {}}]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'modality'):
                scope['validate_modalities'](values)


if __name__ == '__main__':
    unittest.main()
