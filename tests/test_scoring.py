"""Regression checks against recorded paper predictions and pool invariance."""
import copy
import csv
from pathlib import Path
import unittest

from phononscore.scoring import add_fixed_scores, load_scoring_config


class FixedScoringTests(unittest.TestCase):
    def setUp(self):
        self.config = load_scoring_config()
        with (Path(__file__).parent / 'fixtures/fixed_score_examples.csv').open() as stream:
            self.rows = list(csv.DictReader(stream))

    def test_recorded_paper_scores(self):
        for model in self.config['models']:
            rows = [r.copy() for r in self.rows if r['model'] == model]
            add_fixed_scores(rows, model_name=model, config=self.config)
            for row in rows:
                self.assertAlmostEqual(row['standardized_score'], float(row['expected_score']), places=12)

    def test_single_pool_reordered_and_duplicate_agree(self):
        for model in self.config['models']:
            rows = [r.copy() for r in self.rows if r['model'] == model]
            add_fixed_scores(rows, model_name=model, config=self.config)
            expected = {r['id']: r['standardized_score'] for r in rows}
            for pool in ([r.copy()] for r in reversed(rows)):
                add_fixed_scores(pool, model_name=model, config=self.config)
                self.assertEqual(pool[0]['standardized_score'], expected[pool[0]['id']])
            repeated = list(reversed(copy.deepcopy(rows))) + copy.deepcopy(rows)
            add_fixed_scores(repeated, model_name=model, config=self.config)
            self.assertTrue(all(r['standardized_score'] == expected[r['id']] for r in repeated))

    def test_nonfinite_input_is_rejected(self):
        row = self.rows[0].copy(); row['prediction'] = float('nan')
        with self.assertRaises(ValueError):
            add_fixed_scores([row], model_name=row['model'], config=self.config)


if __name__ == '__main__':
    unittest.main()
