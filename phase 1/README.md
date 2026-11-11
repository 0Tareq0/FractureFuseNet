# FractureFuseNet: Multi-Modal Late Fusion of Deep Radiographic Features and Classical Edge/Hough Fracture-Line Detection

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: 16/16 Passed](https://img.shields.io/badge/Tests-16%2F16%20Passed-brightgreen.svg)]()
[![Zero Leakage Verified](https://img.shields.io/badge/Data%20Leakage-0.00%25-brightgreen.svg)]()

**Course / Project:** CSE438 Project 12  
**Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID / Mendeley Data `xwfs6xbk47`)  
**Phase:** Phase 1 — Data Auditing, Leakage-Free Splitting, Classical Feature Engineering & Statistical Hypothesis Testing  

---

## 1. Project Overview

**FractureFuseNet** investigates a hybrid dual-branch computer vision architecture for fine-grained bone fracture classification that fuses:
1. **Branch A (Deep Radiographic Representation):** Convolutional feature maps extracted from resized ($224 \times 224$) radiographs capturing global bone geometry and anatomical context.
2. **Branch B (Classical Edge & Line Representation):** A 64-dimensional structural descriptor derived from Contrast Limited Adaptive Histogram Equalization (CLAHE), multi-threshold Canny edge detection, Probabilistic Hough line analysis, and Sobel gradient orientation statistics.

This repository contains the complete, reproducible Phase 1 data preprocessing, auditing, leakage-free splitting, classical feature extraction, and statistical hypothesis verification pipeline.

---

## 2. Repository Structure

```
FractureFuseNet/
├── configs/
│   └── phase1_config.yaml             # Central configuration (paths, seeds, preprocessing params)
├── data_processed/                    # Processed metadata, manifests, and reports
│   ├── dataset_metadata_cleaned.csv   # Authoritative metadata (1,539 rows, 17 columns)
│   ├── manifests/                     # Cryptographically hashed split manifests
│   │   ├── train_manifest.csv         # 857 train images (59.97%)
│   │   ├── val_manifest.csv           # 285 valid images (19.94%)
│   │   ├── test_manifest.csv          # 287 test images (20.08%)
│   │   ├── excluded_manifest.csv      # 110 excluded images (107 ties, 3 unannotated)
│   │   └── split_manifest_summary.json# Split manifest statistics and checksums
│   ├── edge_feature_scaler.joblib     # StandardScaler fitted strictly on N=857 train images
│   ├── edge_feature_scaler_provenance.json # Scaler provenance & train set hash manifest
│   ├── unaugmented_edge_features.npz  # Cached unaugmented 64-dim classical features
│   ├── train_pixel_stats.json         # Exact train-only RGB normalization statistics
│   └── *.csv / *.json                 # Statistical and overlap reports
├── figures/                           # High-resolution publication figures
│   ├── split_class_distribution.png   # Class distribution across Train/Valid/Test splits
│   ├── split_percentage_distribution.png # Percentage adherence to 60:20:20 target
│   ├── split_group_distribution.png   # Independent source groups per class
│   ├── cv_fold_class_distribution.png # 5-fold cross-validation distribution
│   ├── group_level_cv_hypothesis_distributions.png # 4-panel cluster boxplot (N=496)
│   ├── pseudoreplication_sensitivity_comparison.png # P-value inflation forest plot
│   ├── classwise_edge_metrics.png     # Class-wise classical feature distributions
│   └── modality_feature_comparison.png# X-ray vs. MRI physical feature disparities
├── scripts/
│   ├── create_metadata.py             # Raw annotation parser & perceptual duplicate grouping
│   ├── stratified_split.py            # 5-fold StratifiedGroupKFold 60:20:20 splitter
│   ├── eda.py                         # Train-only pixel statistics & EDA generator
│   ├── verify_cv_hypothesis.py        # 64-dim feature extraction & statistical test suite
│   └── generate_pdf.py                # Headless PDF summary report re-renderer
├── src/
│   ├── __init__.py
│   └── dataset.py                     # Multi-modal BoneFractureDataset, FocalLoss, DataLoaders
├── tests/
│   ├── test_splitting.py              # 11 automated unit tests for splitting integrity
│   └── test_statistical_analysis.py   # 5 automated unit tests for statistical methods
├── requirements.txt                   # Verified dependency specifications
├── run_phase1_pipeline.py             # Complete automated pipeline execution runner
├── FINAL_PHASE1_VALIDATION_REPORT.md  # Comprehensive execution certification report
├── PHASE_1_SUMMARY.md                 # Detailed Phase 1 technical documentation
└── PHASE_1_SUMMARY.pdf                # Publication-quality compiled PDF report
```

---

## 3. Quick Start & Installation

### Step 1: Clone or Navigate to the Repository
```bash
cd "path/to/FractureFuseNet"
```

### Step 2: Install Dependencies
Ensure you are using Python 3.10 or 3.11:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Complete Pipeline
To execute the automated pipeline checks and validate DataLoaders:
```bash
python run_phase1_pipeline.py
```

To force a full re-computation from the raw dataset (rebuilding metadata, splits, manifests, features, scalers, and statistical reports from scratch):
```bash
python run_phase1_pipeline.py --fresh
```

### Step 4: Run Automated Unit Tests
```bash
python -m unittest discover tests -v
```
*All 16 unit tests across splitting integrity and statistical methods must pass.*

---

## 4. Key Pipeline Results & Scientific Findings

### A. Leakage-Free 60:20:20 Grouped Partitioning
* **Total Audited Records:** 1,539 (1,337 X-ray, 202 MRI)
* **Clean Single-Label Records:** 1,429 ($92.85\%$)
* **Excluded Records:** 110 ($7.15\%$, 107 multi-class ties + 3 empty annotation files)
* **Split Allocation:**
  * **Train:** 857 images ($59.97\%$), 143 base IDs, 141 composite groups, 10/10 classes
  * **Valid:** 285 images ($19.94\%$), 47 base IDs, 46 composite groups, 10/10 classes
  * **Test:** 287 images ($20.08\%$), 48 base IDs, 48 composite groups, 9/10 classes\*
* **Pairwise Source-Group Leakage:** **0.00%** (0 base IDs, 0 perceptual duplicate groups, and 0 composite groups cross splits, compared to $40.82\%$ leakage in the original Roboflow partition).

\**`Segmental` fracture has only 2 independent source groups in the dataset (`70_jpg` and `225_jpg`). By the Pigeonhole Principle ($n=2, k=3$), it cannot populate 3 disjoint splits without leakage. It is present in Train (5 images) and Valid (3 images), with 0 in the fixed Test split. However, across 5-fold cross-validation, 100% of the 8 images are evaluated out-of-fold.*

### B. Scaler Provenance & Train-Only Normalization
* **Train-Only Pixel Normalization (`train_pixel_stats.json`):**
  * Overall Train ($N=857$): `mean = [0.2555, 0.2555, 0.2555]`, `std = [0.2594, 0.2594, 0.2596]`
* **Train-Only StandardScaler (`edge_feature_scaler.joblib`):**
  * Fitted strictly on the $N=857$ unaugmented training images (`n_samples_seen_ == 857`).
  * Cryptographic provenance recorded in `edge_feature_scaler_provenance.json` with training split SHA-256 hash.

### C. Cluster-Aware Statistical Inference & Anti-Spurious Caution
* **Pseudoreplication Rectification:** Running tests on $N=1,227$ individual augmented images falsely inflated degrees of freedom by $8.2\times$ (p-values inflated by up to $1.5 \times 10^7\times$). All hypothesis tests were corrected by evaluating cluster medians across $N=496$ independent perceptual duplicate clusters.
* **10,000-Iteration Permutation Tests:** Confirmed that Canny edge density ($p_{\text{perm}} = 1.00 \times 10^{-4}$), Hough line count ($p_{\text{perm}} = 1.00 \times 10^{-4}$), average line length ($p_{\text{perm}} = 0.0336$), and Sobel gradient magnitude ($p_{\text{perm}} = 1.00 \times 10^{-4}$) exhibit significant distributional separation after Holm-Bonferroni and Benjamini-Hochberg corrections.
* **Critical Interpretation:** Healthy radiographs exhibit significantly **HIGHER edge density and line counts** than fractured radiographs (Cliff's $\delta = -0.698$ and $-0.632$). This is caused by anatomical composition (multi-joint hand/wrist views with dense cortical margins vs. long bone shafts), **NOT** because edge detectors isolate the fracture line. Classical features provide whole-image structural context, not localized fracture segmentations.

### D. Modality Confounding (X-ray vs. MRI)
* MRI scans account for $13.13\%$ of the dataset but $64.56\%$ of `Comminuted` fractures, while 5 classes (`Healthy`, `Oblique`, `Segmental`, `Spiral`, `Transverse`) have **zero** MRI images.
* Even within `Comminuted` fractures, MRI images produce more than double the edge density of X-rays ($4.61\%$ vs $2.14\%$, $p = 8.87 \times 10^{-9}$).
* Consequently, the primary benchmark is strictly established on **X-ray**, with MRI reserved for secondary exploratory transfer benchmarking.

---

## 5. Configuration Guide (`configs/phase1_config.yaml`)

All parameters are centrally governed via YAML:
* `dataset.raw_dir`: Path to the raw HBFMID dataset root.
* `project.random_seed`: Seed for StratifiedGroupKFold and permutation tests (default: `42`).
* `preprocessing`: Parameters for CLAHE (`clipLimit: 2.0`), Canny (`30, 100` / `75, 200`), and Hough (`threshold: 40`, `minLineLength: 25`, `maxLineGap: 10`).
* `splitting`: Number of folds (`5`), fold mapping (`train: [0, 2, 4]`, `valid: [1]`, `test: [3]`).
* `statistics`: Number of permutation iterations (`10000`), FWER $\alpha$ (`0.05`), FDR $q$ (`0.05`).
* `dataloader`: Batch size (`16`), workers (`0`).

---

## 6. Citation & Dataset Acknowledgment

If using this codebase or data pipeline, please cite:
* **Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID), Mendeley Data, doi: `10.17632/xwfs6xbk47.1`.
* **Project:** *FractureFuseNet: Hybrid Late Fusion of Deep Radiographic Representations and Classical Edge-Based Fracture-Line Descriptors for Bone Fracture Classification*, CSE438, 2026.
