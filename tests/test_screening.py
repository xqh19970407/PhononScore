"""Integration checks against the fixed figure pools and recorded paper numbers."""
import unittest

import numpy as np
import pandas as pd

from phononscore.screening import (DEFAULT_DATA, check_reference, evaluate_pool,
                                  load_manifest, load_pool, validate_pool, verify_and_extract_cifs)


class FixedScreeningPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.frames = {s['pool_id']: load_pool(DEFAULT_DATA, s) for s in cls.manifest['pools']}

    def test_displayed_metrics_and_selected_material_ids(self):
        recorded = {
            ('MP20_05pct', 'phononscore'): (24, .3676021078256232, .8496587171052631),
            ('MP20_05pct', 'phononscore_dft'): (16, .1982791132025687, .7585485197368422),
            ('MP20_05pct', 'alignn'): (3, .1864003144290628, .8421011513157894),
            ('DFT_05pct', 'phononscore'): (2, .14684129521544872, .7162947368421052),
            ('DFT_05pct', 'phononscore_dft'): (4, .3825606907747905, .8755368421052632),
            ('DFT_05pct', 'alignn'): (0, .11241401748195057, .7140210526315789),
        }
        selected = []
        for spec in self.manifest['pools']:
            metrics, _, _, ids, _ = evaluate_pool(self.frames[spec['pool_id']], spec)
            check_reference(metrics, DEFAULT_DATA)
            selected.append(ids)
            for row in metrics.itertuples(index=False):
                hits, ap, auc = recorded[(row.pool_id, row.method)]
                self.assertEqual(row.hits, hits)
                self.assertAlmostEqual(row.ap, ap, places=12)
                self.assertAlmostEqual(row.roc_auc, auc, places=12)
        generated = pd.concat(selected).sort_values(['pool_id', 'method', 'rank'])
        released = pd.read_csv(DEFAULT_DATA / 'selected_candidates.csv').sort_values(['pool_id', 'method', 'rank'])
        self.assertEqual(len(generated), 111)
        self.assertEqual(list(generated.record_id), list(released.record_id))
        self.assertEqual(list(generated.stable), list(released.stable))
        stable_targets = pd.read_csv(DEFAULT_DATA / 'stable_targets.csv')
        truth_ids = set().union(*(set(frame.loc[frame.stable == 1, 'record_id']) for frame in self.frames.values()))
        self.assertEqual(len(stable_targets), 185)
        self.assertEqual(set(stable_targets.record_id), truth_ids)

    def test_row_reordering_preserves_material_rankings_and_metrics(self):
        for spec in self.manifest['pools']:
            frame = self.frames[spec['pool_id']]
            shuffled = validate_pool(frame.sample(frac=1, random_state=20261105), spec)
            expected, _, _, expected_ids, _ = evaluate_pool(frame, spec)
            actual, _, _, actual_ids, _ = evaluate_pool(shuffled, spec)
            self.assertEqual(list(actual_ids.record_id), list(expected_ids.record_id))
            np.testing.assert_allclose(actual[['ap', 'roc_auc', 'hits', 'precision', 'recall', 'enrichment']],
                                       expected[['ap', 'roc_auc', 'hits', 'precision', 'recall', 'enrichment']], atol=1e-12)

    def test_all_3700_cif_contents_match_released_id_mapping(self):
        self.assertEqual(sum(map(len, self.frames.values())), 3700)
        verify_and_extract_cifs(DEFAULT_DATA, list(self.frames.values()))


if __name__ == '__main__':
    unittest.main()
