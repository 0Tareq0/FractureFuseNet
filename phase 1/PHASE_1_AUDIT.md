# Phase 1 Comprehensive Research & Code Audit Report
**Project Title:** FractureFuseNet: Multi-Modal Late Fusion of Deep Radiographic Features and Classical Edge-Based Fracture-Line Detection  
**Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID)  
**Audit Scope:** `phase 1/`, `phase 1/data/`, `phase 1/scripts/`, `phase 1/src/`, `phase 1/run_phase1_pipeline.py`, `data_processed/`, `scripts/`, `src/`, raw YOLO annotations, and Phase 1 documentation.

---

## Executive Audit Summary

A complete code and data audit identified **11 technical, methodological, and architectural defects** across the initial Phase 1 implementation. All original files and metadata tables were backed up (`phase 1/data/dataset_metadata_cleaned.old.csv`, `data_processed/dataset_metadata_cleaned.old.csv`, and `phase 1/archive/`) prior to applying targeted corrections. Every identified defect has been corrected and verified via automated assertions in `phase 1/run_phase1_pipeline.py`.

---

## Detailed Audit Findings & Resolution Log

### Issue 1: DataFrame Index Misalignment Bug Causing Split Corruption & Data Leakage
1. **Identified Problem:** `clean_df = df[clean_mask].copy().reset_index(drop=True)` reset the filtered DataFrame index to `0..1428` before assigning fold and split labels back to the 1,539-row master DataFrame `df` via `df.loc[clean_df.index, 'split'] = clean_df['split']`.
2. **File and Code Location:** `scripts/build_stratified_folds.py` (lines 123–138; archived in `phase 1/archive/build_stratified_folds.py`).
3. **Why It Is Problematic:** Resetting the index caused `df.loc[clean_df.index]` to overwrite the first 1,429 rows (`0..1428`) of `df` rather than the original filtered row indices. Consequently, 106 valid `single`-class radiographs at the end of `df` were falsely marked `excluded`, while 103 ambiguous `tie` records and all 3 `unannotated` (empty label) records were injected into `train`, `valid`, and `test`. Furthermore, shifting row assignments across index boundaries caused severe leakage across splits (113 `group_id` overlaps between `train` and `valid`, and 109 between `train` and `test`).
4. **Recommended Correction:** Never call `.reset_index(drop=True)` on the filtered subset before mapping labels back to the master DataFrame. Preserve original DataFrame indices (`inc_df = df.loc[included_mask].copy()`) and map fold indices via `orig_indices = inc_df.index[val_pos]`.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/scripts/stratified_split.py` (and synchronized to `scripts/stratified_split.py`).
6. **How the Correction Was Tested:** Verified by automated assertions in `phase 1/scripts/stratified_split.py` and `phase 1/run_phase1_pipeline.py` checking: (a) exact filename-by-filename index alignment, (b) 0 `tie` or `unannotated` records in `train`/`valid`/`test`, and (c) 0 `base_id`, `dup_group_id`, or `group_id` overlaps across splits.

---

### Issue 2: Incompatible Metadata Schemas Across Scripts and Dataset Loader
1. **Identified Problem:** Multiple conflicting metadata schemas existed across scripts. `dataset_metadata_cleaned.old.csv` used `['filename', 'rel_img_path', 'rel_lbl_path', 'base_id', 'original_split', 'is_empty_label', 'num_boxes', 'num_unique_classes', 'primary_class', 'all_classes', 'leakage_free_split', 'stratified_group_split']`, whereas `build_stratified_folds.py` overwrote `dataset_metadata_cleaned.csv` with `['filename', 'raw_split', 'rel_img_path', 'rel_lbl_path', 'modality', 'label_quality', 'primary_class', 'num_boxes', 'old_base_id', 'group_id', 'fold', 'split']`.
2. **File and Code Location:** `phase 1/data/dataset_metadata_cleaned.csv`, `data_processed/dataset_metadata_cleaned.csv`, `phase 1/src/dataset.py` (line 31), and `phase 1/run_phase1_pipeline.py` (lines 52–71).
3. **Why It Is Problematic:** `phase 1/src/dataset.py` and `phase 1/run_phase1_pipeline.py` referenced `is_empty_label`, `base_id`, `original_split`, and `stratified_group_split`, causing an immediate `KeyError: 'is_empty_label'` when running the pipeline against the overwritten CSV.
4. **Recommended Correction:** Standardize a single authoritative 17-column schema across all scripts and loaders: `filename`, `rel_img_path`, `rel_lbl_path`, `base_id`, `original_split`, `modality`, `is_empty_label`, `num_boxes`, `num_unique_classes`, `primary_class`, `all_classes`, `label_quality`, `group_id`, `dup_group_id`, `cv_fold`, `stratified_group_split`, and `exclusion_reason`.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/scripts/create_metadata.py`, `phase 1/scripts/stratified_split.py`, `phase 1/src/dataset.py`, and `phase 1/run_phase1_pipeline.py`.
6. **How the Correction Was Tested:** Step 2 of `phase 1/run_phase1_pipeline.py` programmatically asserts the presence of all required columns and verifies identical schemas in `phase 1/data/dataset_metadata_cleaned.csv` and `data_processed/dataset_metadata_cleaned.csv`.

---

### Issue 3: Unaligned Geometric Augmentation Between Deep and Classical CV Branches
1. **Identified Problem:** During training (`is_train=True`), `BoneFractureDataset.__getitem__` applied `RandomHorizontalFlip(p=0.5)` and `RandomRotation(degrees=15)` only to the PIL image entering the Deep Vision branch (`self.deep_transforms(pil_img)`), while reading the unaugmented image from disk (`cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)`) for the Classical CV branch.
2. **File and Code Location:** `phase 1/src/dataset.py` (lines 35–41 and 171–179) and `src/dataset.py`.
3. **Why It Is Problematic:** The 64-dimensional classical feature vector encodes spatially localized descriptors ($4 \times 4$ grid edge densities in `[0..31]`, quadrant edge densities in `[56..59]`, and directional Hough/Sobel orientation statistics in `[36..40, 44..51]`). When the deep branch image is horizontally flipped or rotated by $\pm 15^\circ$ while the classical features are extracted from the unflipped/unrotated image, the two branches represent geometrically contradictory views of the radiograph.
4. **Recommended Correction:** Implement `apply_shared_geometric_transform(pil_img, is_train)` prior to branching so that both the Deep Vision branch and the Classical CV branch receive representations derived from the exact same geometrically transformed radiograph.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/src/dataset.py` (lines 195–222 and 325–345) and synchronized to `src/dataset.py`.
6. **How the Correction Was Tested:** Step 6 of `phase 1/run_phase1_pipeline.py` tests `apply_shared_geometric_transform` with `do_flip=False` and `do_flip=True`, asserting exact pixel mirror symmetry (`arr_no_flip == arr_flipped[:, ::-1, :]`) and verifying that top-left (`[56]`) and top-right (`[57]`) quadrant high-threshold edge densities swap symmetrically.

---

### Issue 4: Silent Arbitrary Class Assignment on Tied Multi-Class Annotations
1. **Identified Problem:** For radiographs containing multiple bounding boxes of different classes with equal counts (107 records, e.g., 1 box of `Segmental` and 1 box of `Comminuted`), `Counter(labels).most_common(1)[0][0]` assigned whichever class happened to appear on the first line of the YOLO `.txt` file as `primary_class`.
2. **File and Code Location:** `phase 1/scripts/create_metadata.py` (line 59) and `scripts/build_stratified_folds.py` (line 34).
3. **Why It Is Problematic:** Line ordering in YOLO annotation files is arbitrary. Assigning a single class label to an exact multi-class tie injects label noise into single-label training and evaluation.
4. **Recommended Correction:** Categorize every record into `label_quality` $\in \{\text{single}, \text{majority}, \text{tie}, \text{unannotated}\}$. Assign `primary_class = 'Ambiguous_Tie'` to the 107 tied records, exclude them from single-label training/validation/test splits (`stratified_group_split = 'excluded'`), preserve all box classes in `all_classes`, and log all 110 excluded records (107 ties + 3 empty files) in `phase 1/data/excluded_records_audit_log.csv` and `phase 1/data/annotation_investigation_report.csv`.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/scripts/create_metadata.py`.
6. **How the Correction Was Tested:** Verified that all 107 tied records and 3 empty records are logged in `excluded_records_audit_log.csv` and asserted in Step 4 of `phase 1/run_phase1_pipeline.py` that 0 `tie` or `unannotated` records enter `train`, `valid`, or `test`.

---

### Issue 5: Incomplete Source-Image & Perceptual Duplicate Grouping in Splitting
1. **Identified Problem:** `phase 1/scripts/create_metadata.py` and `phase 1/scripts/stratified_split.py` grouped only by filename prefix `base_id` without checking whether identical or augmented radiographs existed under different `base_id` prefixes. Conversely, `scripts/build_stratified_folds.py` grouped only by perceptual hash `dup_group_id`, allowing images sharing the same `base_id` to be split across train, validation, and test.
2. **File and Code Location:** `phase 1/scripts/create_metadata.py` (lines 85–96), `phase 1/scripts/stratified_split.py` (lines 12–63), and `scripts/build_stratified_folds.py` (lines 95–132).
3. **Why It Is Problematic:** In HBFMID, 158 of 245 `base_id` prefixes contain multiple distinct source radiographs (because Roboflow flattened class subfolders numbered `1.jpg..225.jpg`), while 4 perceptual duplicate clusters merge distinct `base_id`s. To satisfy both zero `base_id` overlap and zero perceptual duplicate-group overlap across splits, the splitting group identifier must be the connected component uniting both relations.
4. **Recommended Correction:** Compute dihedral perceptual hashes (`pHash` + `dHash` across original, horizontal flip, vertical flip, and $180^\circ$ rotation) to form perceptual duplicate clusters (`dup_group_id` at threshold $\le 10$), and construct composite `group_id` via Union-Find uniting `base_id` and `dup_group_id`. Perform 5-fold `StratifiedGroupKFold` on `group_id`.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/scripts/create_metadata.py` and `phase 1/scripts/stratified_split.py`.
6. **How the Correction Was Tested:** Automated assertions in `phase 1/scripts/stratified_split.py` and `phase 1/run_phase1_pipeline.py` verify 0 `base_id` overlap, 0 `dup_group_id` overlap, and 0 `group_id` overlap across `train`, `valid`, and `test`.

---

### Issue 6: Numerical & Statistical Flaws in the 64-Dimensional Classical Feature Vector
1. **Identified Problem:**
   - `norm_line_count = min(line_count, 500.0) / 500.0` saturated at 500 lines.
   - `mean_ang = np.mean(ang_arr) / 180.0` and `std_ang = np.std(ang_arr) / 180.0` computed linear mean and standard deviation on axial angles $\theta \in [0, 180^\circ)$, which wraps around at $0^\circ / 180^\circ$.
   - `img_contrast` and `img_brightness` (indices 54–55) encoded raw pixel intensity/exposure shortcuts (confounding MRI vs. X-ray) rather than structural edge geometry.
   - Normalized Shannon orientation entropy was omitted despite being specified in the methodology.
   - `high_to_low_edge_ratio` used `+ 1e-7` in the denominator, risking extreme values if `edges_low` had 0 pixels, and zero-gradient images could trigger NaN in weighted `np.histogram(..., density=True)`.
2. **File and Code Location:** `phase 1/src/dataset.py` (lines 80–165).
3. **Why It Is Problematic:** Linear averaging of axial angles produces wrong geometric summaries (e.g., lines at $2^\circ$ and $178^\circ$ average to $90^\circ$ instead of $0^\circ$), raw brightness/contrast acts as a modality shortcut, and `density=True` with zero weights yields `NaN`.
4. **Recommended Correction:**
   - Replace `min(line_count, 500)/500` with `np.log1p(line_count) / np.log1p(1000.0)`.
   - Replace linear `mean_ang` and `std_ang` with axial circular statistics using doubled angles $2\theta$: circular mean `((0.5 * arctan2(mean(sin(2*theta)), mean(cos(2*theta)))) % pi) / pi` (`[36]`) and circular dispersion `1 - R` (`[37]`).
   - Replace `img_contrast` and `img_brightness` with normalized Shannon orientation entropy (`[54]`) and `log1p(total_line_length / diag)` (`[55]`).
   - Safeguard zero-gradient images by checking `total_mag > 1e-7` and `img_gray.max() == img_gray.min()`, and sanitize output with `np.nan_to_num`.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/src/dataset.py` (`extract_classical_edge_features_np`).
6. **How the Correction Was Tested:** Verified on all 1,429 included radiographs in `verify_cv_hypothesis.py` and tested on synthetic zero-gradient (`np.zeros`) and constant (`np.full(..., 128)`) images in Step 6 of `phase 1/run_phase1_pipeline.py` (0 NaNs, 0 Infs, exact length 64).

---

### Issue 7: Lack of Train-Only `StandardScaler` on the 64-Dimensional Classical Feature Vector
1. **Identified Problem:** Raw 64-dimensional classical features (which span different scales such as small grid densities $\sim 0.01$ vs. log lengths and ratios) were fed directly into the dataset output without standardization.
2. **File and Code Location:** `phase 1/src/dataset.py` (lines 165, 179).
3. **Why It Is Problematic:** Unscaled features with disparate magnitudes degrade gradient-based optimization in the MLP fusion branch and bias distance/regularization terms. Fitting a scaler on the whole dataset would leak validation/test distribution statistics.
4. **Recommended Correction:** Fit `sklearn.preprocessing.StandardScaler` strictly on the unaugmented training split (`stratified_group_split == 'train'`, $N=857$), save it to `phase 1/data/edge_feature_scaler.joblib`, and apply `scaler.transform` across `train`, `valid`, and `test`.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/src/dataset.py` (`fit_or_load_edge_scaler`) and `phase 1/scripts/verify_cv_hypothesis.py`.
6. **How the Correction Was Tested:** Step 6 of `phase 1/run_phase1_pipeline.py` loads `edge_feature_scaler.joblib` and asserts `scaler.n_samples_seen_ == 857` (the exact training split size), and verifies zero NaNs/Infs on scaled train, valid, and test batches.

---

### Issue 8: Flawed Dataset Pixel Mean/Std Computation in `eda.py`
1. **Identified Problem:** `eda.py` computed image normalization statistics across all 1,539 images (including validation and test splits) and calculated `overall_std = pixel_stds.mean(axis=0)` (the average of within-image standard deviations).
2. **File and Code Location:** `phase 1/scripts/eda.py` (lines 57–59, 178–183).
3. **Why It Is Problematic:** Including validation and test images leaks evaluation statistics into preprocessing, and averaging per-image standard deviations ignores between-image intensity variance, underestimating the true dataset standard deviation (`0.2331` vs. true train std `0.2594`).
4. **Recommended Correction:** Compute exact pixel mean and true dataset standard deviation $\sqrt{\mathbb{E}[X^2] - (\mathbb{E}[X])^2}$ strictly over the training split (`stratified_group_split == 'train'`, $N=857$), both overall and stratified by modality (X-ray vs. MRI), and update `BoneFractureDataset.TRAIN_PIXEL_MEAN` and `TRAIN_PIXEL_STD`.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/scripts/eda.py` and `phase 1/src/dataset.py`.
6. **How the Correction Was Tested:** Verified by running `phase 1/scripts/eda.py` and saving `phase 1/data/train_pixel_stats.json` (`mean=[0.2555, 0.2555, 0.2555]`, `std=[0.2594, 0.2594, 0.2596]`).

---

### Issue 9: Single-Pair Anecdotal Comparison Instead of Dataset-Wide Classical CV Hypothesis Testing
1. **Identified Problem:** The previous implementation evaluated Canny edge pixels and Hough line counts on only a single fracture image (`101_jpg`) and a single healthy image (`109_jpg`), and generalized that fracture radiographs have $10\times$ more Hough lines.
2. **File and Code Location:** `phase 1/run_phase1_pipeline.py` (lines 74–87) and `phase 1/PHASE_1_SUMMARY.md` (Section 7).
3. **Why It Is Problematic:** A sample size of $n=2$ has zero statistical validity, and Hough lines detect normal cortical bone boundaries as well as fracture lines.
4. **Recommended Correction:** Create `phase 1/scripts/verify_cv_hypothesis.py` to evaluate all 1,429 included radiographs across all 10 classes and separately within the X-ray modality ($N=64$ Healthy vs. $N=1,163$ Fracture X-rays), reporting Mean, SD, Median, IQR, Mann-Whitney U tests with rank-biserial effect sizes ($r_{rb}$), and 10-class Kruskal-Wallis H-tests with $\varepsilon^2$ effect sizes.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/scripts/verify_cv_hypothesis.py`.
6. **How the Correction Was Tested:** Executed across all 1,429 images, producing `phase 1/data/cv_hypothesis_binary_comparison.csv`, `phase 1/data/cv_hypothesis_classwise_stats.csv`, `phase 1/data/cv_kruskal_wallis_summary.json`, and figures `cv_hypothesis_distributions.png` and `classwise_edge_metrics.png`.

---

### Issue 10: Duplicate / Outdated Scripts & Relative Path Fragility
1. **Identified Problem:** Duplicate copies of `eda.py`, `create_metadata.py`, `stratified_split.py`, and `dataset.py` existed in both the root workspace (`scripts/`, `src/`) and `phase 1/` (`phase 1/scripts/`, `phase 1/src/`), alongside 7 temporary scratch scripts in `scripts/`. Moreover, `phase 1/scripts/*.py` used hardcoded relative paths (`data_processed/...`) that failed when invoked from different working directories.
2. **File and Code Location:** `scripts/`, `src/`, `phase 1/scripts/`, `phase 1/src/`.
3. **Why It Is Problematic:** Maintaining unsynchronized duplicate files leads to version drift, and hardcoded relative paths break execution when running scripts from `phase 1/` vs. the project root.
4. **Recommended Correction:** Use `pathlib.Path(__file__).resolve()` in every script to resolve directories deterministically regardless of working directory; archive the 7 temporary scripts to `phase 1/archive/`; and synchronize the 4 canonical scripts (`create_metadata.py`, `stratified_split.py`, `eda.py`, `verify_cv_hypothesis.py`) and `src/dataset.py`.
5. **Whether It Was Corrected:** **Yes (Corrected)**.
6. **How the Correction Was Tested:** Executed all scripts and `phase 1/run_phase1_pipeline.py` and verified identical file contents and clean path resolution.

---

### Issue 11: Windows Backslash Separators & Unconditional "SUCCESS" Banner in Master Pipeline
1. **Identified Problem:** `phase 1/scripts/create_metadata.py` used `os.path.join(s, 'images', img_name)` which wrote Windows backslashes (`train\images\...`) into `rel_img_path` and `rel_lbl_path` in the old CSV. Furthermore, `phase 1/run_phase1_pipeline.py` printed hardcoded strings (`"40.8% patient data leakage"`, `"SUCCESS: ... 100% Ready!"`) without asserting split non-overlap, tensor shapes, or NaN/Inf absence, and only loaded a batch from `train_loader` (ignoring `val_loader` and `test_loader`).
2. **File and Code Location:** `phase 1/scripts/create_metadata.py` (lines 64–65) and `phase 1/run_phase1_pipeline.py` (lines 61–108).
3. **Why It Is Problematic:** Backslashes in CSV paths fail on Linux/Colab, and printing `"SUCCESS"` without checking all three DataLoaders or verifying split integrity can mask critical errors.
4. **Recommended Correction:** Normalize all relative paths to forward slashes (`/`), and rewrite `phase 1/run_phase1_pipeline.py` to execute 12 strict validation checks (including loading and validating batches from all 3 splits: `train`, `valid`, and `test`) before printing the summary.
5. **Whether It Was Corrected:** **Yes (Corrected)** in `phase 1/scripts/create_metadata.py` and `phase 1/run_phase1_pipeline.py`.
6. **How the Correction Was Tested:** Executed `python "phase 1/run_phase1_pipeline.py"`, confirming all 12 automated checks pass with 0 backslash paths and verified `(16, 3, 224, 224)`, `(16, 64)`, and `(16,)` tensor shapes across `train`, `valid`, and `test`.
