"""Protocol tests against the user-provided reference, not model accuracy tests."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from robustness.data import corrupt, prepare, arrays, verify
from robustness import corruption_reference as reference


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.RandomState(91).uniform(1, 1000, (17, 12, 19, 3)).astype('float32')

    def test_reference_equivalence_and_unchanged_features(self):
        for seed in (1, 2024):
            for rate in (.1, .2, .3):
                for kind, function in [('node', reference.node_missing),
                                       ('node_gap', reference.node_contiguous_gap),
                                       ('noise', reference.gaussian_noise)]:
                    with self.subTest(seed=seed, rate=rate, kind=kind):
                        before = self.x.copy()
                        actual, mask, nodes = corrupt(self.x, kind, rate, seed)
                        expected = function(self.x, rate=rate, seed=seed)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(self.x, before)
                        np.testing.assert_array_equal(actual[..., 1:], self.x[..., 1:])
                        np.testing.assert_array_equal(actual[..., 0][~mask], self.x[..., 0][~mask])
                        self.assertTrue(np.all(actual >= 0))

    def test_node_ids_shared_across_different_split_lengths(self):
        previous = set()
        for rate in (.1, .2, .3):
            selections = []
            for s in (17, 9, 5):
                _, mask, nodes = corrupt(self.x[:s], 'node', rate, 2024)
                self.assertEqual(len(nodes), int(19 * rate))
                self.assertTrue(mask[:, :, nodes].all())
                selections.append(nodes)
            np.testing.assert_array_equal(selections[0], selections[1])
            np.testing.assert_array_equal(selections[0], selections[2])
            self.assertTrue(previous.issubset(set(selections[0])))
            previous = set(selections[0])

    def test_gap_blocks_and_counts(self):
        for rate in (.1, .2, .3):
            _, mask, _ = corrupt(self.x, 'node_gap', rate)
            blocks = mask.reshape(17, 4, 3, 19)
            np.testing.assert_array_equal(blocks[:, :, 0], blocks[:, :, 1])
            np.testing.assert_array_equal(blocks[:, :, 1], blocks[:, :, 2])
            expected = int(17 * 4 * rate) * 3
            self.assertTrue(np.all(mask.sum(axis=(0, 1)) == expected))
            self.assertFalse(np.array_equal(mask[:, :, 0], mask[:, :, 1]))

    def test_noise_selection_count(self):
        for rate in (.1, .2, .3):
            actual, mask, _ = corrupt(self.x, 'noise', rate)
            self.assertEqual(int(mask.sum()), int(17 * 12 * 19 * rate))
            np.testing.assert_array_equal(actual, corrupt(self.x, 'noise', rate)[0])

    def test_clean_and_validation(self):
        out, mask, _ = corrupt(self.x, 'clean', 0)
        np.testing.assert_array_equal(out, self.x)
        self.assertFalse(mask.any())
        with self.assertRaises(ValueError):
            corrupt(self.x[:, :11], 'node_gap', .1)
        with self.assertRaises(ValueError):
            corrupt(self.x, 'noise', -.1)

    def test_full_bundle_boundaries_labels_scaler_and_hashes(self):
        test_root = Path(__file__).resolve().parent / 'results'
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            root = Path(folder)
            assert root.resolve().is_relative_to(test_root.resolve())
            source = root / 'PEMS08'
            source.mkdir()
            raw = np.empty((240, 170, 3), dtype='float32')
            raw[..., 0] = np.arange(240)[:, None] + np.arange(170)[None, :] + 1
            raw[..., 1] = np.arange(240)[:, None] / 288
            raw[..., 2] = 0
            raw.tofile(source / 'data.dat')
            (source / 'desc.json').write_text(json.dumps(dict(shape=list(raw.shape),
                feature_description=['traffic flow', 'time of day', 'day of week'])))
            bundle = prepare(root, 'PEMS08', root / 'bundle', 'node', .2)
            meta = verify(bundle)
            self.assertFalse(meta['key']['smoke'])
            selected = meta['splits']['train']['missing_nodes']
            for split, begin, end in [('train', 0, 144), ('val', 144, 192), ('test', 192, 240)]:
                values = arrays(bundle, split)
                self.assertEqual(len(values['x']), end - begin - 23)
                self.assertEqual(values['starts'][0], begin)
                self.assertEqual(values['starts'][-1] + 24, end)
                self.assertEqual(meta['splits'][split]['missing_nodes'], selected)
                self.assertTrue((values['x'][:, :, selected] == 0).all())
                for index, start in enumerate(values['starts']):
                    np.testing.assert_array_equal(values['y'][index], raw[start + 12:start + 24, :, :1])
                from robustness.hyperd_data import BundleDataset
                dataset = BundleDataset(bundle, 'valid' if split == 'val' else split)
                item = dataset[0]
                np.testing.assert_array_equal(item['inputs'][..., :1], values['x'][0])
                np.testing.assert_array_equal(item['target'][..., :1], values['y'][0])
            train_x = arrays(bundle, 'train')['x']
            self.assertAlmostEqual(meta['scaler']['mean'], float(np.mean(train_x, dtype='float64')))
            self.assertAlmostEqual(meta['scaler']['std'], float(np.std(train_x, dtype='float64')))
            from robustness.hyperd_data import make_priors
            prior_dir = root / 'priors'
            prior_dir.mkdir()
            for name in make_priors(bundle, prior_dir):
                prior = np.load(name)
                self.assertTrue(np.isfinite(prior).all())
                self.assertTrue(np.all(prior[:, selected] <= 0))
            np.testing.assert_array_equal(np.fromfile(source / 'data.dat', dtype='float32').reshape(raw.shape), raw)
            # Cached requests cannot silently reuse another corruption setting.
            with self.assertRaises(ValueError):
                prepare(root, 'PEMS08', root / 'bundle', 'node', .3)
            with (bundle / 'test_x.npy').open('r+b') as stream:
                stream.seek(-4, 2)
                stream.write(b'1234')
            with self.assertRaises(ValueError):
                verify(bundle)
            del values, dataset, train_x


if __name__ == '__main__':
    unittest.main()
