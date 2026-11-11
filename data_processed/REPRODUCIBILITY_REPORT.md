# FractureFuseNet: Dataset Splitting Reproducibility Report

**Project:** FractureFuseNet (Project 12)
**Random Seed:** 42
**CV Folds:** 5
**Stratification Method:** `sklearn.model_selection.StratifiedGroupKFold`
**Grouping Variable:** `group_id` (connected components of `base_id` $\cup$ `dup_group_id`)
**Target Allocation:** 60% Train, 20% Valid, 20% Test (Folds 0,2,4 -> Train, Fold 1 -> Valid, Fold 3 -> Test)

## 1. Cryptographic File Integrity Checksums (SHA-256 & MD5)

| File Name | Path | Rows | SHA-256 Checksum | MD5 Checksum |
| :--- | :--- | :---: | :--- | :--- |
| `dataset_metadata_cleaned.csv` | `phase 1/data/` | 1539 | `8a75dc64ebe826b3a08ba6a1053a458f5e65528b9a61b17e6bae550d814f6771` | `9ba744d59398b5baa079031d4e746728` |
| `train_manifest.csv` | `phase 1/data/manifests/` | 857 | `58a5ca911c2fedbe3ecaa073403d69465fd420e68df68ae8ff11865b9cdee18d` | `bcf794169678ad5c1c1f499459bf7659` |
| `val_manifest.csv` | `phase 1/data/manifests/` | 285 | `df2f328539934186b9dd27adc425e5f2511451c86a457b97c6a9a1499318e26a` | `96df085f8f867c26c3a04410293fc3c5` |
| `test_manifest.csv` | `phase 1/data/manifests/` | 287 | `516815dd3fdc1a981991425bfcbfe8eebbfc45b06bf23d66aa7aef4fc29a0825` | `a97f099436b2ebde31686511fe830d3e` |
| `excluded_manifest.csv` | `phase 1/data/manifests/` | 110 | `dee2114e9434738d2c2ee1263f1559392f4c72d57160964a14d60ff3973685e8` | `87f163614c8aeb33030fd3fae5aeadf2` |

## 2. Environment & Dependency Specifications

- Python Version: `3.11.0`
- NumPy Version: `2.4.6`
- Pandas Version: `2.3.3`
- Scikit-Learn Version: `1.4.0+`

## 3. Bit-for-Bit Deterministic Reproduction Instructions

To reproduce the exact split bit-for-bit from the cleaned metadata:

```powershell
python "phase 1/scripts/stratified_split.py"
```

The script checks `RANDOM_SEED=42`, re-executes `StratifiedGroupKFold`, verifies index alignment, and validates all 6 automated assertions before re-exporting identical manifests.
