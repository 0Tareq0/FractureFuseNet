# Phase 1 Technical Report: Data Auditing, Leakage-Free Splitting & Multimodal Preprocessing
**Project 12:** FractureFuseNet: Multi-Modal Late Fusion of Deep Radiographic Features and Classical Edge-Based Fracture-Line Detection  
**Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID)  
**Course:** CSE438  
**Date:** November 10, 2026  

---

## 1. Project Overview

**FractureFuseNet** investigates a hybrid dual-branch architecture for bone fracture classification that combines:
1. **Branch A (Deep Radiographic Representation):** Convolutional feature maps extracted from resized ($224 \times 224$) radiographs to capture global anatomical structure and bone morphology.
2. **Branch B (Classical Edge & Line Representation):** A 64-dimensional structural descriptor computed via Contrast Limited Adaptive Histogram Equalization (CLAHE), multi-threshold Canny edge detection, Probabilistic Hough Transform line analysis, and Sobel gradient orientation statistics.

This Phase 1 report documents the complete audit, annotation cleaning, duplicate and data-leakage resolution, multimodal preprocessing standardization, dataset-wide statistical hypothesis verification, and end-to-end PyTorch pipeline validation.

---

## 2. Dataset Description

* **Source Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID, Mendeley Data `xwfs6xbk47`, exported via Roboflow in YOLOv8 format).
* **Total Image Records:** **1,539** images ($640 \times 640$ pixels, 3-channel RGB encoding grayscale radiographs) and **1,539** corresponding `.txt` annotation files.
* **Original Roboflow Partition:**
  * `train`: 1,347 images ($3\times$ offline augmented from 449 source images using random horizontal flip, vertical flip, $\pm 5^\circ$ rotation, $\pm 2^\circ$ shear, and $\pm 10\%$ brightness adjustment).
  * `valid`: 128 images (unaugmented).
  * `test`: 64 images (unaugmented).
* **Imaging Modalities (Identified via Filename Audit):**
  * **X-ray (`xray`):** 1,337 images ($86.87\%$)
  * **MRI (`mri`):** 202 images ($13.13\%$, identified by `_mri_` in filenames)
* **Target Classes (10):** `Comminuted` (0), `Greenstick` (1), `Healthy` (2), `Linear` (3), `Oblique Displaced` (4), `Oblique` (5), `Segmental` (6), `Spiral` (7), `Transverse Displaced` (8), and `Transverse` (9).

---

## 3. Dataset Auditing

Programmatic verification (`phase 1/scripts/eda.py` and `phase 1/scripts/create_metadata.py`) across all 1,539 image-annotation pairs established the following baseline integrity metrics:

| Audit Check | Measured Result | Verification Status |
| :--- | :---: | :---: |
| **Total Image Files Audited** | 1,539 | Verified |
| **Total Annotation Files Audited** | 1,539 | Verified |
| **Corrupt or Unreadable Images** | 0 | Passed |
| **Image Dimensions** | $640 \times 640$ (1,539 / 1,539) | Passed |
| **Color Encoding** | 3-channel RGB (1,539 / 1,539) | Passed |
| **Total Bounding Boxes Parsed** | 1,771 | Verified |
| **Invalid Class IDs ($\notin [0, 9]$)** | 0 | Passed |
| **Out-of-Bounds Box Coordinates ($\notin [0.0, 1.0]$)** | 0 | Passed |
| **Exact MD5 Byte-Level Duplicates** | 0 | Passed |

---

## 4. Annotation Cleaning & Inclusion Policy

Across the 1,539 records, 220 images ($14.29\%$) contain more than one bounding box, and 113 images ($7.34\%$) contain bounding boxes spanning more than one class. Rather than silently discarding records or arbitrarily selecting the first bounding box in multi-class files, every record was audited and categorized into four `label_quality` tiers:

| `label_quality` Tier | Record Count | Percentage | Description & Policy Decision |
| :--- | :---: | :---: | :--- |
| **`single`** | **1,423** | $92.46\%$ | Contains $\ge 1$ bounding box, all belonging to a single unambiguous fracture/healthy class. **Included.** |
| **`majority`** | **6** | $0.39\%$ | Multi-class annotations with a strict majority class (`52_jpg.rf.*`: 2 boxes `Transverse Displaced`, 1 box `Oblique Displaced`). **Included** under majority label. |
| **`tie`** | **107** | $6.95\%$ | Multi-class annotations with an exact tie in box counts (e.g., 1 box `Segmental` + 1 box `Comminuted`). Assigned `primary_class = 'Ambiguous_Tie'` and **Excluded** from single-label splits to prevent arbitrary first-box labeling bias. |
| **`unannotated`** | **3** | $0.19\%$ | 0-byte empty annotation files (`114_jpg.rf.*` in `train`). Assigned `primary_class = 'Unannotated'` and **Excluded**. |
| **Total Audited** | **1,539** | $100.00\%$ | **1,429 Included ($92.85\%$) \| 110 Excluded ($7.15\%$)** |

All 110 excluded records and all 113 multi-class records are preserved in the master metadata table (`phase 1/data/dataset_metadata_cleaned.csv`) and logged individually in:
* `phase 1/data/excluded_records_audit_log.csv` (110 records)
* `phase 1/data/annotation_investigation_report.csv` (113 records)

---

## 5. Duplicate Detection & Source Grouping Analysis

Two complementary grouping mechanisms were implemented in `phase 1/scripts/create_metadata.py`:
1. **Filename Prefix Grouping (`base_id`):** Extracted via `filename.split('.rf.')[0]`, yielding **245** unique `base_id` prefixes across the 1,539 images (238 across the 1,429 included records).
2. **Dihedral Perceptual Hash Clustering (`dup_group_id`):** Because Roboflow applied random horizontal flips ($50\%$), vertical flips ($50\%$), and rotations ($\pm 5^\circ$), we computed 64-bit `pHash` and 64-bit `dHash` across all 4 dihedral orientations (original, horizontal flip, vertical flip, and $180^\circ$ rotation) and clustered images via Union-Find at a conservative combined Hamming distance threshold of $\le 10$ bits (justified by the bimodal pairwise distance distribution in `phase 1/figures/duplicate_distance_histogram.png`).

### Key Empirical Findings on Grouping:
* **Perceptual Duplicate Clusters (`dup_group_id`):** **657** distinct clusters across all 1,539 images (closely matching the 641 raw source radiographs reported in the Mendeley dataset documentation).
* **Why `base_id` Alone Does Not Equal One Source Image:** **158 out of 245 `base_id` prefixes ($64.49\%$)** contain more than one distinct perceptual cluster (`dup_group_id`). When source images from separate class folders (each numbered `1.jpg, 2.jpg, ...`) were imported into Roboflow, distinct radiographs with the same number received the same `base_id` prefix (for example, `225_jpg` contains 3 `Segmental` images and 9 `Oblique Displaced` images).
* **Perceptual Clusters Merging Different `base_id`s:** **4 out of 657 perceptual clusters** span 2 different `base_id` prefixes (`245 - 3 = 242` connected components).

---

## 6. Data Leakage Investigation in the Original Roboflow Split

Evaluating the original Roboflow partition (`train`: 1,347, `valid`: 128, `test`: 64) against both grouping definitions revealed substantial cross-split leakage:
* **`base_id` Cross-Split Overlap:** **100 out of 245 `base_id`s ($40.82\%$)** appear in more than one split in the original Roboflow partition.
* **Perceptual Duplicate Cluster (`dup_group_id`) Cross-Split Leakage:** **84 out of 657 perceptual clusters ($12.79\%$)** span multiple splits in the original partition:
  * `train` + `valid`: 48 clusters
  * `train` + `test`: 31 clusters
  * `valid` + `test`: 2 clusters
  * `train` + `valid` + `test`: 3 clusters
* **Missing Test Classes in Original Split:** The original 64-image `test` split contains **0 `Linear`** images and **0 `Segmental`** images.

---

## 7. Corrected Splitting Methodology (`stratified_group_split`)

To guarantee that **neither `base_id` nor any confirmed perceptual duplicate cluster (`dup_group_id`) ever crosses splits**, `phase 1/scripts/stratified_split.py` constructs a composite connected-component identifier (`group_id`) via Union-Find over the union of `base_id` equivalence and `dup_group_id` equivalence:
* Across all 1,539 records: **242** composite `group_id`s.
* Across the 1,429 included records: **235** composite `group_id`s (spanning 238 `base_id`s and 612 perceptual duplicate groups).

Using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the 1,429 included records (while strictly preserving original DataFrame indices):
* **Train Split (`cv_fold` $\in \{0, 2, 4\}$):** **857 images ($59.97\%$)** across **143 `base_id`s** / **141 composite `group_id`s**.
* **Validation Split (`cv_fold` $= 1$):** **285 images ($19.94\%$)** across **47 `base_id`s** / **46 composite `group_id`s**.
* **Test Split (`cv_fold` $= 3$):** **287 images ($20.08\%$)** across **48 `base_id`s** / **48 composite `group_id`s**.

### Automated Zero-Leakage Verification Results:
* `base_id` overlap between `train`, `valid`, and `test`: **0**
* `dup_group_id` overlap between `train`, `valid`, and `test`: **0**
* `group_id` overlap between `train`, `valid`, and `test`: **0**
* Image filename overlap between `train`, `valid`, and `test`: **0**
* Excluded (`tie` / `unannotated`) records in `train`, `valid`, or `test`: **0**

---

## 8. Class & Modality Distribution

### 8.1 Split Distribution Across the 10 Target Classes (1,429 Included Records)

| Class ID | Class Name | Train Images | Valid Images | Test Images | Total Included Images | Train Composite Groups | Valid Composite Groups | Test Composite Groups | Total Composite Groups |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **Comminuted** | 94 | 32 | 32 | **158** | 30 | 12 | 12 | 54 |
| 1 | **Greenstick** | 50 | 16 | 17 | **83** | 8 | 1 | 3 | 12 |
| 2 | **Healthy** | 41 | 12 | 11 | **64** | 6 | 2 | 2 | 10 |
| 3 | **Linear** | 12 | 3 | 7 | **22** | 3 | 1 | 1 | 5 |
| 4 | **Oblique Displaced** | 185 | 61 | 61 | **307** | 30 | 10 | 10 | 50 |
| 5 | **Oblique** | 31 | 9 | 12 | **52** | 5 | 1 | 2 | 8 |
| 6 | **Segmental** | 5 | 3 | **0*** | **8** | 1 | 1 | **0*** | **2** |
| 7 | **Spiral** | 43 | 16 | 15 | **74** | 7 | 3 | 2 | 12 |
| 8 | **Transverse Displaced** | 340 | 114 | 114 | **568** | 58 | 19 | 19 | 96 |
| 9 | **Transverse** | 56 | 19 | 18 | **93** | 9 | 3 | 3 | 15 |
| — | **Total Included** | **857** | **285** | **287** | **1,429** | **141** | **46** | **48** | **235** |

*\*Note on `Segmental` Constraint:* In the entire cleaned dataset, `Segmental` has only **8 single-label images belonging to 2 `base_id` groups** (`70_jpg` with 5 images and `225_jpg` with 3 images; the other 13 images containing `Segmental` boxes in the raw dataset are multi-class exact ties in `116_jpg` and `134_jpg`). By the Pigeonhole Principle, 2 disjoint `base_id` groups can populate at most 2 of the 3 splits without violating zero-leakage. `70_jpg` (5 images) is assigned to `train` and `225_jpg` (3 images) is assigned to `valid` (and both are evaluated in 5-fold cross-validation via `cv_fold`). Note also that some composite groups contain distinct source radiographs from more than one class due to shared `base_id` prefixes, so the sum of per-class composite group counts (264) exceeds the 235 unique composite groups.

### 8.2 Modality Distribution Across Classes (1,429 Included + 110 Excluded Records)

| Class / Category | X-ray Images | MRI Images | Total Images | MRI Fraction (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Comminuted** | 56 | **102** | 158 | **64.56%** |
| **Greenstick** | 77 | 6 | 83 | 7.23% |
| **Healthy** | 64 | 0 | 64 | 0.00% |
| **Linear** | 19 | 3 | 22 | 13.64% |
| **Oblique** | 52 | 0 | 52 | 0.00% |
| **Oblique Displaced** | 282 | 25 | 307 | 8.14% |
| **Segmental** | 8 | 0 | 8 | 0.00% |
| **Spiral** | 74 | 0 | 74 | 0.00% |
| **Transverse** | 93 | 0 | 93 | 0.00% |
| **Transverse Displaced** | 502 | 66 | 568 | 11.62% |
| *Excluded (`Ambiguous_Tie`)* | *107* | *0* | *107* | *0.00%* |
| *Excluded (`Unannotated`)* | *3* | *0* | *3* | *0.00%* |
| **Total** | **1,337** | **202** | **1,539** | **13.13%** |

---

## 9. Classical CV Feature Extraction & Dataset-Wide Hypothesis Evaluation

### 9.1 Corrected 64-Dimensional Classical Feature Vector (`phase 1/src/dataset.py`)
Every radiograph is processed via CLAHE (`clipLimit=2.0, tileGridSize=(8,8)`) and $3\times 3$ Gaussian smoothing, followed by extraction of 64 structural features:
* **Indices `[0..31]` (32 dims):** $4 \times 4$ spatial grid edge densities for low-threshold Canny ($30, 100$) and high-threshold Canny ($75, 200$).
* **Indices `[32..40]` (9 dims):** Probabilistic Hough Transform (`rho=1, theta=pi/180, threshold=40, minLineLength=25, maxLineGap=10`) descriptors:
  * `[32]`: $\log(1 + \text{line\_count}) / \log(1001)$ (prevents saturation above 500 lines).
  * `[33..35]`: Mean, standard deviation, and maximum line length normalized by image diagonal $D = \sqrt{H^2 + W^2}$.
  * `[36..37]`: Axial circular statistics over line angles $\theta \in [0, 180^\circ)$ via angle doubling ($2\theta$): circular mean orientation (`[36]`) and circular dispersion $1 - R$ (`[37]`).
  * `[38..40]`: Horizontal ($<30^\circ$ or $\ge 150^\circ$), vertical ($60^\circ$–$120^\circ$), and oblique line fractions.
* **Indices `[41..51]` (11 dims):** Global Sobel gradient magnitude mean, std, and max (`[41..43]`), plus an 8-bin magnitude-weighted orientation probability histogram (`[44..51]`) with safe zero-gradient handling.
* **Indices `[52..55]` (4 dims):** Global low-threshold Canny edge density (`[52]`), global high-threshold Canny edge density (`[53]`), normalized Shannon orientation entropy $H \in [0, 1]$ (`[54]`), and $\log(1 + \text{total\_line\_length} / D)$ (`[55]`). *(Raw brightness and contrast were removed to eliminate modality/exposure shortcuts.)*
* **Indices `[56..59]` (4 dims):** $2 \times 2$ quadrant high-threshold Canny edge densities (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
* **Indices `[60..63]` (4 dims):** Sobel gradient magnitude 90th, 95th, and 99th percentiles (`[60..62]`), and stabilized high-to-low Canny edge ratio $\frac{\sum \text{edges\_high}}{\sum \text{edges\_low} + 1}$ (`[63]`).

### 9.2 Corrected Statistical Hypothesis Verification (`phase 1/scripts/verify_cv_hypothesis.py`)

A critical scientific audit identified that running statistical tests on individual augmented images violates the independent observation assumption (pseudoreplication, inflating degrees of freedom by $8.2\times$). The analysis was therefore re-engineered at the **perceptual duplicate cluster level (`dup_group_id`)**, aggregating augmented views to their cluster medians ($N=496$ independent X-ray clusters: 27 Healthy, 469 Fracture):

#### A. Cluster-Level Primary X-ray Hypothesis Testing ($N = 496$ Independent Clusters)
| Classical CV Metric | Healthy ($n=27$) Median (IQR) | Fracture ($n=469$) Median (IQR) | Cliff's $\delta$ | CLES (AUC) | $p_{\text{raw}}$ | $p_{\text{perm}} (10\text{k})$ | $p_{\text{Holm}}$ | Sig. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **High Canny Edge Density (%)** | 2.96 (1.52) | 1.87 (1.29) | **-0.698** | 0.151 | $1.06 \times 10^{-9}$ | $1.00 \times 10^{-4}$ | $9.51 \times 10^{-9}$ | *** |
| **Low Canny Edge Density (%)** | 7.47 (3.53) | 4.88 (2.81) | **-0.648** | 0.176 | $1.45 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | $1.16 \times 10^{-7}$ | *** |
| **Detected Hough Line Count** | 188.50 (102.50) | 80.00 (82.00) | **-0.632** | 0.184 | $3.31 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | $2.32 \times 10^{-7}$ | *** |
| **Average Line Length (px)** | 48.60 (9.83) | 57.38 (17.38) | **+0.408** | 0.704 | $3.65 \times 10^{-4}$ | $3.36 \times 10^{-2}$ | $1.10 \times 10^{-3}$ | ** |
| **Mean Sobel Gradient Mag** | 33.30 (13.62) | 24.86 (10.95) | **-0.623** | 0.189 | $5.19 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | $2.60 \times 10^{-7}$ | *** |
| **Spatial Heterogeneity (%)** | 2.51 (1.02) | 1.67 (0.83) | **-0.512** | 0.244 | $7.73 \times 10^{-6}$ | $9.00 \times 10^{-4}$ | $3.09 \times 10^{-5}$ | *** |
| **Orientation Entropy** | 0.96 (0.02) | 0.96 (0.03) | -0.187 | 0.407 | $1.02 \times 10^{-1}$ | $2.83 \times 10^{-1}$ | $2.05 \times 10^{-1}$ | ns |
| **Circular Dispersion** | 0.56 (0.07) | 0.57 (0.07) | -0.096 | 0.452 | $4.01 \times 10^{-1}$ | $9.52 \times 10^{-1}$ | $4.01 \times 10^{-1}$ | ns |

*Multiple-comparison corrections: Holm-Bonferroni step-down (FWER $\alpha=0.05$) and Benjamini-Hochberg (FDR $q=0.05$).*

#### B. Pseudoreplication Audit & P-Value Inflation
Comparing naive image-level ($N=1,227$) with cluster-level ($N=496$) inference reveals extreme Type I error inflation:
* High Edge Density: image-level $p = 6.89 \times 10^{-17}$ vs. cluster-level $p = 1.06 \times 10^{-9}$ ($1.5 \times 10^7\times$ inflation).
* Hough Line Count: image-level $p = 7.83 \times 10^{-14}$ vs. cluster-level $p = 3.31 \times 10^{-8}$ ($4.2 \times 10^5\times$ inflation).

#### C. Cluster-Level 9-Class Kruskal-Wallis Omnibus Tests (X-ray Modality, $N = 488$ Clusters)
| Classical CV Metric | Kruskal-Wallis $H$ | $p$-value | Effect Size ($\varepsilon^2$) |
| :--- | :---: | :---: | :---: |
| **High Canny Edge Density (%)** | 82.37 | $1.63 \times 10^{-14}$ | 0.167 |
| **Mean Sobel Gradient Mag** | 80.07 | $4.74 \times 10^{-14}$ | 0.163 |
| **Low Canny Edge Density (%)** | 78.27 | $1.09 \times 10^{-13}$ | 0.159 |
| **Average Line Length (px)** | 77.64 | $1.46 \times 10^{-13}$ | 0.158 |
| **Hough Line Count / Density** | 75.90 | $3.26 \times 10^{-13}$ | 0.154 |
| **Spatial Edge Heterogeneity (%)** | 60.39 | $3.91 \times 10^{-10}$ | 0.123 |

**Critical Scientific Interpretation:**  
Healthy radiographs exhibit significantly **HIGHER edge density and line count** than fractured radiographs (Cliff's $\delta = -0.698$ and $-0.632$). This occurs because the healthy radiographs in HBFMID represent multi-joint skeletal views (dense carpal/metacarpal margins), whereas fracture radiographs often focus on long bone shafts with large empty backgrounds. **Statistical significance therefore does NOT prove that the edge detector isolates fracture lines specifically; it proves that classical features capture whole-image anatomical and structural geometry**, validating their inclusion as a complementary branch in FractureFuseNet.

---

## 10. Multimodal Preprocessing & Shared Augmentation Architecture

In `phase 1/src/dataset.py`, `BoneFractureDataset` enforces strict geometric alignment and train-only parameter fitting:

```
[ Raw Radiograph (640x640 RGB) ]
                │
                ▼
[ Shared Geometric Transform (apply_shared_geometric_transform) ]
  * Train: RandomHorizontalFlip (p=0.5) + RandomRotation (±15°, bilinear)
  * Valid/Test: Identity (no random geometric distortion)
                │
        ┌───────┴───────────────────────────────────────────┐
        ▼                                                   ▼
[ Branch A: Deep Radiographic Stream ]          [ Branch B: Classical Edge/Line Stream ]
  * Resize to 224 × 224                           * Grayscale conversion (640 × 640)
  * ToTensor() in [0.0, 1.0]                      * CLAHE (2.0, 8×8) + 3×3 Gaussian blur
  * Train-Split Calibrated Normalization:         * Multi-threshold Canny + HoughLinesP + Sobel
      Mean = [0.2555, 0.2555, 0.2555]             * 64-dim structural vector (0 NaN / 0 Inf)
      Std  = [0.2594, 0.2594, 0.2596]             * Train-Only StandardScaler (N=857 train imgs)
  * Output Tensor: (Batch, 3, 224, 224)           * Output Tensor: (Batch, 64)
```

* **Train-Only Image Normalization (`phase 1/data/train_pixel_stats.json`):** Computed strictly over the $N=857$ training images using exact dataset variance $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$:
  * Overall Train ($N=857$): `mean = [0.2555, 0.2555, 0.2555]`, `std = [0.2594, 0.2594, 0.2596]`
  * X-ray Train ($N=743$): `mean = [0.2517, 0.2517, 0.2515]`, `std = [0.2567, 0.2566, 0.2568]`
  * MRI Train ($N=114$): `mean = [0.2801, 0.2802, 0.2811]`, `std = [0.2756, 0.2758, 0.2761]`
* **Train-Only Classical Feature Standardization (`phase 1/data/edge_feature_scaler.joblib`):** `StandardScaler` fitted strictly on the $N=857$ unaugmented training feature vectors (`n_samples_seen_ == 857`), achieving zero-mean ($\max |\mu| = 2.53 \times 10^{-7}$) and unit variance ($\sigma = 1.0000$) on train, and applied without refitting to validation and test sets.

---

## 11. Pipeline Verification Results

Executing `python "phase 1/run_phase1_pipeline.py"` performs 12 automated assertions and verifies all three DataLoaders (`batch_size = 16`):

| Verification Step | Measured Result | Status |
| :--- | :--- | :---: |
| **1. Dataset Auto-Discovery** | Located `Bone Fractures Detection` root | PASSED |
| **2. Authoritative Schema Check** | 1,539 rows, 17 standardized columns | PASSED |
| **3. File Path & Slash Validation** | 1,539 images & 1,539 labels exist; 0 backslashes | PASSED |
| **4. Split Exclusion Integrity** | 1,429 included (`single`=1,423, `majority`=6); 110 excluded (`tie`=107, `unannotated`=3) | PASSED |
| **5. Zero Source-Group Leakage** | 0 `base_id`, 0 `dup_group_id`, and 0 `group_id` overlaps across splits | PASSED |
| **6. Zero-Gradient Stability** | `np.zeros((640,640))` & constant images produce finite `(64,)` vectors | PASSED |
| **7. Shared Geometric Symmetry** | Horizontal flip preserves exact pixel mirror symmetry & swaps TL/TR quadrant densities | PASSED |
| **8. Train-Only Scaler Audit** | `edge_feature_scaler.joblib` verified with `n_samples_seen_ == 857` | PASSED |
| **9. Train DataLoader Check** | 54 batches; batch shapes: Image `(16, 3, 224, 224)`, Classical `(16, 64)`, Labels `(16,)` | PASSED |
| **10. Valid DataLoader Check** | 18 batches; batch shapes: Image `(16, 3, 224, 224)`, Classical `(16, 64)`, Labels `(16,)` | PASSED |
| **11. Test DataLoader Check** | 18 batches; batch shapes: Image `(16, 3, 224, 224)`, Classical `(16, 64)`, Labels `(16,)` | PASSED |
| **12. NaN / Inf & Loss Check** | 0 NaNs, 0 Infs across all tensors and `FocalLoss(gamma=2.0)` forward pass | PASSED |

---

## 12. Limitations & Methodological Caveats

1. **Extreme Scarcity of `Segmental` ($n=8$ images, 2 `base_id` groups) and `Linear` ($n=22$ images, 5 `base_id` groups):**
   * Because `Segmental` has only 2 `base_id` groups in the cleaned single-label dataset (`70_jpg` and `225_jpg`), it is mathematically impossible to populate all 3 disjoint splits (`train`, `valid`, `test`) without leaking a source group across splits. `Segmental` is present in `train` (5 images) and `valid` (3 images), and has 0 images in the fixed `test` split (though it is evaluated in 5-fold cross-validation across `cv_fold`). Phase 2 evaluation must report macro-F1 both across the 9 test-present classes and across all 5 cross-validation folds with explicit `zero_division=0` and sample-size (`n=...`) caveats.
2. **Modality Confounding (MRI vs. X-ray):**
   * MRI scans account for $13.13\%$ (202/1,539) of the dataset but $64.56\%$ (102/158) of `Comminuted` fractures, while 6 classes (`Healthy`, `Oblique`, `Segmental`, `Spiral`, `Transverse`) contain $0\%$ MRI scans. Phase 2 models must report evaluation metrics both overall and stratified within X-ray-only and MRI-only subsets.
3. **Lack of True Patient Metadata:**
   * Public HBFMID annotations do not include clinical patient identifiers. Grouping via `group_id = base_id` $\cup$ `dup_group_id` prevents all filename-level and perceptual near-duplicate leakage, but cannot detect if two completely different anatomical views from the same patient were assigned different source numbers prior to Roboflow export.

---

## 13. Phase 2 Roadmap

With Phase 1 audited and verified, Phase 2 will proceed in a separate `phase 2/` directory:
1. **Baseline 1 (Deep-Only):** Pretrained CNN backbones (EfficientNet-B0 / ResNet-50) trained on the $(3, 224, 224)$ image stream using class-weighted Cross-Entropy / Focal Loss.
2. **Baseline 2 (Classical CV-Only):** Random Forest, XGBoost, and MLP classifiers trained solely on the standardized 64-dimensional classical edge/line vector.
3. **Proposed FractureFuseNet (Feature-Level & Decision-Level Late Fusion):** Dual-branch fusion combining the deep representation with a projected classical edge embedding ($64 \to 128$), benchmarked against standalone baselines and a random-edge control ablation across both the fixed split and 5-fold cross-validation, reporting overall and modality-stratified (X-ray vs. MRI) metrics.
