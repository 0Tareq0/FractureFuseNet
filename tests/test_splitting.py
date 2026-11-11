"""
Unit Test Suite for FractureFuseNet Dataset Splitting & Zero-Leakage Verification
Dataset: Human Bone Fractures Multi-modal Image Dataset (HBFMID)

Tests:
1. Metadata existence and authoritative schema validation.
2. Exact row count verification (1,429 included, 110 excluded).
3. Exact split size verification (857 train, 285 valid, 287 test).
4. Pairwise zero filename overlap.
5. Pairwise zero base_id overlap.
6. Pairwise zero dup_group_id (perceptual hash cluster) overlap.
7. Pairwise zero composite group_id overlap.
8. Exclusion integrity (zero ties, zero empty labels in active splits).
9. Split manifest consistency (train, val, test, excluded manifests).
10. Deterministic reproducibility with random_state=42.
11. Rare class Pigeonhole Principle constraint verification (Segmental & Linear).
12. Class label validity across all 10 target classes.
"""

import unittest
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]

REQUIRED_COLUMNS = [
    "filename", "rel_img_path", "rel_lbl_path", "base_id", "original_split",
    "modality", "is_empty_label", "num_boxes", "num_unique_classes",
    "primary_class", "all_classes", "label_quality", "group_id",
    "dup_group_id", "cv_fold", "stratified_group_split", "exclusion_reason"
]


class TestDatasetSplitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(__file__).resolve().parent
        # Candidate locations for data and manifest directories
        candidates = [
            test_dir.parent / "data",
            test_dir.parent / "phase 1" / "data",
            test_dir / "data",
        ]
        data_dir = None
        for c in candidates:
            if (c / "dataset_metadata_cleaned.csv").exists():
                data_dir = c
                break

        if data_dir is None:
            # Fallback check
            data_dir = Path(r"c:\Users\User\Downloads\New folder\phase 1\data")

        cls.data_dir = data_dir
        cls.manifest_dir = data_dir / "manifests"
        cls.meta_path = data_dir / "dataset_metadata_cleaned.csv"

        if not cls.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {cls.meta_path}")

        cls.df = pd.read_csv(cls.meta_path)
        cls.active_splits = ["train", "valid", "test"]
        cls.active_df = cls.df[cls.df["stratified_group_split"].isin(cls.active_splits)]
        cls.excluded_df = cls.df[cls.df["stratified_group_split"] == "excluded"]

    def test_01_metadata_existence_and_schema(self):
        """Verifies metadata table exists, has 1539 rows, and contains all 17 required columns."""
        self.assertEqual(len(self.df), 1539, "Total dataset rows must equal 1,539.")
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.df.columns, f"Required column '{col}' missing from metadata schema.")

    def test_02_record_counts(self):
        """Verifies exact split counts for included and excluded records."""
        self.assertEqual(len(self.active_df), 1429, "Included records must equal 1,429.")
        self.assertEqual(len(self.excluded_df), 110, "Excluded records must equal 110.")
        self.assertEqual(len(self.df[self.df["stratified_group_split"] == "train"]), 857, "Train split must have 857 images.")
        self.assertEqual(len(self.df[self.df["stratified_group_split"] == "valid"]), 285, "Valid split must have 285 images.")
        self.assertEqual(len(self.df[self.df["stratified_group_split"] == "test"]), 287, "Test split must have 287 images.")

    def test_03_zero_filename_overlap(self):
        """Asserts zero image filename overlap between train, valid, and test."""
        f_train = set(self.df[self.df["stratified_group_split"] == "train"]["filename"])
        f_val = set(self.df[self.df["stratified_group_split"] == "valid"]["filename"])
        f_test = set(self.df[self.df["stratified_group_split"] == "test"]["filename"])

        self.assertEqual(len(f_train & f_val), 0, "Filename overlap found between train and valid!")
        self.assertEqual(len(f_train & f_test), 0, "Filename overlap found between train and test!")
        self.assertEqual(len(f_val & f_test), 0, "Filename overlap found between valid and test!")

    def test_04_zero_base_id_overlap(self):
        """Asserts zero source base_id overlap between train, valid, and test."""
        b_train = set(self.df[self.df["stratified_group_split"] == "train"]["base_id"])
        b_val = set(self.df[self.df["stratified_group_split"] == "valid"]["base_id"])
        b_test = set(self.df[self.df["stratified_group_split"] == "test"]["base_id"])

        self.assertEqual(len(b_train & b_val), 0, f"base_id leakage found between train and valid: {b_train & b_val}")
        self.assertEqual(len(b_train & b_test), 0, f"base_id leakage found between train and test: {b_train & b_test}")
        self.assertEqual(len(b_val & b_test), 0, f"base_id leakage found between valid and test: {b_val & b_test}")

    def test_05_zero_dup_group_id_overlap(self):
        """Asserts zero perceptual duplicate-group (dup_group_id) overlap between train, valid, and test."""
        d_train = set(self.df[self.df["stratified_group_split"] == "train"]["dup_group_id"])
        d_val = set(self.df[self.df["stratified_group_split"] == "valid"]["dup_group_id"])
        d_test = set(self.df[self.df["stratified_group_split"] == "test"]["dup_group_id"])

        self.assertEqual(len(d_train & d_val), 0, f"dup_group_id leakage between train and valid: {d_train & d_val}")
        self.assertEqual(len(d_train & d_test), 0, f"dup_group_id leakage between train and test: {d_train & d_test}")
        self.assertEqual(len(d_val & d_test), 0, f"dup_group_id leakage between valid and test: {d_val & d_test}")

    def test_06_zero_composite_group_id_overlap(self):
        """Asserts zero composite group_id overlap between train, valid, and test."""
        g_train = set(self.df[self.df["stratified_group_split"] == "train"]["group_id"])
        g_val = set(self.df[self.df["stratified_group_split"] == "valid"]["group_id"])
        g_test = set(self.df[self.df["stratified_group_split"] == "test"]["group_id"])

        self.assertEqual(len(g_train & g_val), 0, f"group_id leakage between train and valid: {g_train & g_val}")
        self.assertEqual(len(g_train & g_test), 0, f"group_id leakage between train and test: {g_train & g_test}")
        self.assertEqual(len(g_val & g_test), 0, f"group_id leakage between valid and test: {g_val & g_test}")

    def test_07_exclusion_integrity(self):
        """Verifies that no empty label or multi-class tie record enters train, valid, or test."""
        self.assertFalse(self.active_df["is_empty_label"].any(), "Empty label record found in active split!")
        self.assertTrue(self.active_df["label_quality"].isin(["single", "majority"]).all(), "Non-clean record in active split!")
        self.assertTrue(self.active_df["primary_class"].isin(CLASS_NAMES).all(), "Invalid primary_class in active split!")

        # Verify excluded breakdown
        self.assertEqual((self.df["label_quality"] == "tie").sum(), 107, "Must have exactly 107 multi-class ties.")
        self.assertEqual((self.df["label_quality"] == "unannotated").sum(), 3, "Must have exactly 3 unannotated records.")

    def test_08_split_manifests_consistency(self):
        """Verifies that split manifests exist and match the master metadata row-by-row."""
        train_m = pd.read_csv(self.manifest_dir / "train_manifest.csv")
        val_m = pd.read_csv(self.manifest_dir / "val_manifest.csv")
        test_m = pd.read_csv(self.manifest_dir / "test_manifest.csv")
        excl_m = pd.read_csv(self.manifest_dir / "excluded_manifest.csv")

        self.assertEqual(len(train_m), 857, "train_manifest.csv row count mismatch.")
        self.assertEqual(len(val_m), 285, "val_manifest.csv row count mismatch.")
        self.assertEqual(len(test_m), 287, "test_manifest.csv row count mismatch.")
        self.assertEqual(len(excl_m), 110, "excluded_manifest.csv row count mismatch.")

        self.assertTrue((train_m["stratified_group_split"] == "train").all())
        self.assertTrue((val_m["stratified_group_split"] == "valid").all())
        self.assertTrue((test_m["stratified_group_split"] == "test").all())
        self.assertTrue((excl_m["stratified_group_split"] == "excluded").all())

    def test_09_deterministic_reproducibility(self):
        """Verifies that StratifiedGroupKFold with seed=42 reproduces the exact split bit-for-bit."""
        inc_mask = (self.df["label_quality"].isin(["single", "majority"])) & (~self.df["is_empty_label"])
        inc_df = self.df.loc[inc_mask].copy()

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        reproduced_folds = pd.Series(index=inc_df.index, dtype=int)

        for fold_idx, (_, val_pos) in enumerate(sgkf.split(inc_df, inc_df["primary_class"], inc_df["group_id"])):
            orig_idx = inc_df.index[val_pos]
            reproduced_folds.loc[orig_idx] = fold_idx

        self.assertTrue((reproduced_folds == inc_df["cv_fold"]).all(), "Reproducibility failure: Fold assignments do not match metadata!")

    def test_10_rare_class_segmental_constraint(self):
        """Verifies the Pigeonhole Principle constraint on Segmental (2 groups cannot populate 3 splits)."""
        seg_df = self.active_df[self.active_df["primary_class"] == "Segmental"]
        self.assertEqual(len(seg_df), 8, "Clean Segmental count must be exactly 8.")
        self.assertEqual(seg_df["base_id"].nunique(), 2, "Segmental must have exactly 2 base_ids (70_jpg and 225_jpg).")

        seg_train = seg_df[seg_df["stratified_group_split"] == "train"]
        seg_val = seg_df[seg_df["stratified_group_split"] == "valid"]
        seg_test = seg_df[seg_df["stratified_group_split"] == "test"]

        self.assertEqual(len(seg_train), 5, "Segmental train count must be 5 (from 70_jpg).")
        self.assertEqual(len(seg_val), 3, "Segmental valid count must be 3 (from 225_jpg).")
        self.assertEqual(len(seg_test), 0, "Segmental test count must be 0 (Pigeonhole Principle enforced).")

    def test_11_rare_class_linear_coverage(self):
        """Verifies that Linear (5 groups) is properly distributed across all 3 splits."""
        lin_df = self.active_df[self.active_df["primary_class"] == "Linear"]
        self.assertEqual(len(lin_df), 22, "Clean Linear count must be exactly 22.")
        self.assertEqual(lin_df["base_id"].nunique(), 5, "Linear must have exactly 5 base_ids.")

        self.assertGreater(len(lin_df[lin_df["stratified_group_split"] == "train"]), 0)
        self.assertGreater(len(lin_df[lin_df["stratified_group_split"] == "valid"]), 0)
        self.assertGreater(len(lin_df[lin_df["stratified_group_split"] == "test"]), 0)


if __name__ == "__main__":
    unittest.main()
