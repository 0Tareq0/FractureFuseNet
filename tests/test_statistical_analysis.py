import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

TEST_DIR = Path(__file__).resolve().parent
PHASE1_DIR = TEST_DIR.parent
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from scripts.verify_cv_hypothesis import (
    holm_bonferroni,
    benjamini_hochberg,
    compute_effect_sizes,
    group_permutation_test,
    compute_summary_stats
)


class TestStatisticalAnalysis(unittest.TestCase):
    """Unit tests for statistical methods, effect sizes, and cluster corrections."""

    def setUp(self):
        # Locate data directory
        candidates = [
            PHASE1_DIR / "data",
            TEST_DIR.parent / "data",
            Path.cwd() / "phase 1" / "data",
            Path.cwd() / "data"
        ]
        self.data_dir = None
        for c in candidates:
            if (c / "dataset_metadata_cleaned.csv").exists():
                self.data_dir = c
                break

    def test_holm_bonferroni_adjustment(self):
        """Holm-Bonferroni adjusted p-values must be >= raw p-values and <= 1.0."""
        raw_p = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.20])
        adj_p = holm_bonferroni(raw_p)
        self.assertEqual(len(adj_p), len(raw_p))
        self.assertTrue((adj_p >= raw_p).all(), "Adjusted p-values must be >= raw p-values")
        self.assertTrue((adj_p <= 1.0).all(), "Adjusted p-values must not exceed 1.0")
        # Step-down monotonicity for sorted values:
        # 0.001 * 6 = 0.006; 0.01 * 5 = 0.05; 0.03 * 4 = 0.12
        self.assertAlmostEqual(adj_p[0], 0.006, places=4)
        self.assertAlmostEqual(adj_p[1], 0.050, places=4)
        self.assertAlmostEqual(adj_p[2], 0.120, places=4)

    def test_benjamini_hochberg_adjustment(self):
        """Benjamini-Hochberg FDR adjusted p-values must be monotonic and >= raw p-values."""
        raw_p = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.20])
        adj_p = benjamini_hochberg(raw_p)
        self.assertEqual(len(adj_p), len(raw_p))
        self.assertTrue((adj_p >= raw_p).all(), "BH p-values must be >= raw p-values")
        self.assertTrue((adj_p <= 1.0).all(), "BH p-values must not exceed 1.0")
        # BH is less conservative than Holm:
        holm_p = holm_bonferroni(raw_p)
        self.assertTrue((adj_p <= holm_p).all(), "BH p-values must be <= Holm p-values")

    def test_effect_sizes_bounds(self):
        """Effect sizes must satisfy mathematical bounds."""
        g1 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        eff = compute_effect_sizes(g1, g2)
        
        # Perfect separation:
        self.assertAlmostEqual(eff["cliff_delta"], 1.0, places=4)
        self.assertAlmostEqual(eff["cles_auc"], 1.0, places=4)
        self.assertAlmostEqual(eff["rank_biserial_r"], 1.0, places=4)
        self.assertTrue(eff["cohen_d"] > 2.0)

        # Reversed separation:
        eff_rev = compute_effect_sizes(g2, g1)
        self.assertAlmostEqual(eff_rev["cliff_delta"], -1.0, places=4)
        self.assertAlmostEqual(eff_rev["cles_auc"], 0.0, places=4)
        self.assertAlmostEqual(eff_rev["rank_biserial_r"], -1.0, places=4)
        self.assertTrue(eff_rev["cohen_d"] < -2.0)

        # Identical distributions:
        eff_id = compute_effect_sizes(g1, g1)
        self.assertAlmostEqual(eff_id["cliff_delta"], 0.0, places=4)
        self.assertAlmostEqual(eff_id["cles_auc"], 0.5, places=4)
        self.assertAlmostEqual(eff_id["cohen_d"], 0.0, places=4)

    def test_group_permutation_test_deterministic(self):
        """Permutation test should be reproducible given a fixed seed."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        p1 = group_permutation_test(g1, g2, n_perms=1000, seed=42)
        p2 = group_permutation_test(g1, g2, n_perms=1000, seed=42)
        self.assertEqual(p1, p2, "Permutation test must be deterministic with fixed seed")
        self.assertTrue(0.0 <= p1 <= 1.0, "p-value must be in [0, 1]")

    def test_summary_stats(self):
        """Summary statistics should return valid dictionary with IQR and medians."""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        st = compute_summary_stats(arr)
        self.assertEqual(st["n"], 5)
        self.assertEqual(st["median"], 30.0)
        self.assertEqual(st["min"], 10.0)
        self.assertEqual(st["max"], 50.0)
        self.assertTrue(st["iqr"] > 0)


if __name__ == "__main__":
    unittest.main()
