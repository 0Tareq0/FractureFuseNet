# Dataset Decision & Experimental Scope Report
**Project 12:** FractureFuseNet  
**Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID / Mendeley `xwfs6xbk47`)  
**Status:** Authoritative Research Scope & Experimental Design  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This report establishes the final dataset policy, modality protocol, and experimental evaluation scope for **FractureFuseNet**. Based on empirical audits across all 1,539 image-annotation pairs, 245 source groups, and both imaging modalities (X-ray vs. MRI), we resolve the clinical and statistical contradictions present in the raw Mendeley/Roboflow distribution.

### Core Strategic Decisions:
1. **Primary Experimental Benchmark:** **X-ray Fracture Classification across 9 Well-Supported Classes** on a leak-free holdout test partition (244 images).
2. **Comprehensive Secondary Benchmark:** **Full 10-Class Evaluation under 5-Fold Grouped Cross-Validation** (1,227 X-ray / 1,429 Multimodal images), evaluating 100% of the rare class `Segmental` ($n=8$) out-of-fold.
3. **Exploratory Modality Benchmark:** **MRI-Specific Evaluation on the 5 Present Classes** ($n=202$ images) to measure domain transfer and quantify modality-specific shortcut vulnerability.
4. **Preservation of Raw Data:** Zero images are permanently deleted or discarded. Excluded records ($n=110$: 107 multi-class ties + 3 unannotated files) are isolated in immutable audit manifests.
5. **No Data Fabrication:** In strict adherence to scientific integrity, minority classes (`Segmental`, `Linear`) are neither synthetically oversampled nor duplicated across splits.

---

## 2. Modality Distribution & Severe Confounding Analysis

The HBFMID dataset comprises two distinct radiological imaging modalities: Plain Radiography (**X-ray**) and Magnetic Resonance Imaging (**MRI**).

### Table 2.1: Modality Census Across the Dataset
| Metric | X-ray | MRI | Combined Dataset |
| :--- | :---: | :---: | :---: |
| **Total Raw Records** | 1,337 (86.87%) | 202 (13.13%) | 1,539 (100.0%) |
| **Clean Single/Majority Records** | 1,227 (85.86%) | 202 (14.14%) | 1,429 (92.85%) |
| **Excluded Ambiguous Ties** | 107 (100.0%) | 0 (0.0%) | 107 (6.95%) |
| **Excluded Empty Files** | 3 (100.0%) | 0 (0.0%) | 3 (0.20%) |
| **Unique Base IDs (Clean)** | 152 | 86 | 238 |
| **Composite Source Groups (Clean)** | 149 | 86 | 235 |
| **Classes Represented** | **10 / 10** | **5 / 10** | **10 / 10** |

### Table 2.2: Class Breakdown by Modality (Clean Records, $N=1,429$)
| Class Name | X-ray Images | X-ray Groups | MRI Images | MRI Groups | Total Images | Total Groups | MRI Proportion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Comminuted** | 56 | 8 | 102 | 46 | 158 | 54 | **64.56%** |
| **Greenstick** | 77 | 10 | 6 | 2 | 83 | 12 | 7.23% |
| **Healthy** | 64 | 10 | 0 | 0 | 64 | 10 | **0.00%** |
| **Linear** | 19 | 4 | 3 | 1 | 22 | 5 | 13.64% |
| **Oblique** | 52 | 8 | 0 | 0 | 52 | 8 | **0.00%** |
| **Oblique Displaced** | 282 | 41 | 25 | 9 | 307 | 50 | 8.14% |
| **Segmental** | 8 | 2 | 0 | 0 | 8 | 2 | **0.00%** |
| **Spiral** | 74 | 12 | 0 | 0 | 74 | 12 | **0.00%** |
| **Transverse** | 93 | 15 | 0 | 0 | 93 | 15 | **0.00%** |
| **Transverse Displaced** | 502 | 68 | 66 | 28 | 568 | 96 | 11.62% |
| **Total** | **1,227** | **149** | **202** | **86** | **1,429** | **235** | **14.14%** |

### Critical Finding: The Modality Confounder
1. **Total Absence of 5 Classes in MRI:** MRI contains **zero** instances of `Healthy`, `Oblique`, `Segmental`, `Spiral`, and `Transverse`.
2. **Massive Asymmetry in `Comminuted`:** Although MRI constitutes only 14.14% of clean images, **64.56% of all Comminuted fractures are MRI scans**. In fact, 50.5% of all MRI images (102/202) are Comminuted.
3. **Shortcut Learning Hazard:** A machine learning model trained on unstratified pooled modalities can easily learn a non-anatomical heuristic: *"Low bone contrast / dark cortical boundary => predict Comminuted or Transverse Displaced; never predict Healthy, Oblique, Segmental, Spiral, or Transverse."*
4. **Resolution:** Modality must be decoupled. The primary benchmark must focus on the complete X-ray modality where all classes reside, with MRI isolated for secondary exploratory benchmarking.

---

## 3. Anatomical View & Region Investigation

An exhaustive regex and metadata audit of all 1,539 filenames, directory structures, and annotation files was performed to determine if anatomical projection (e.g., Anteroposterior [AP], Posteroanterior [PA], Lateral [LAT], Oblique) or anatomical region (e.g., wrist, forearm, femur, tibia) information is accessible.

### Audit Findings:
1. **Filename Structure:** All filenames follow an opaque numeric convention (e.g., `101_jpg.rf.<hash>.jpg` or `13_mri_0_5388_jpeg.rf.<hash>.jpg`).
2. **Keyword Search Results:** Zero occurrences of projection terms (`ap`, `pa`, `lat`, `lateral`, `obl`, `axial`, `coronal`, `sagittal`) or anatomical region terms (`radius`, `ulna`, `femur`, `tibia`, `wrist`, `knee`, `elbow`, `skull`, `spine`) were detected in filenames or label files.
3. **Annotation Format:** Annotations are standard YOLO bounding boxes: `class_id x_center y_center width height`. No auxiliary metadata attributes exist.
4. **Mendeley Documentation Discrepancy:** The Mendeley repository description states that raw acquisitions were sourced across various body parts (elbow, finger, forearm, humerus, shoulder, femur, shinbone, knee, hipbone, wrist, spinal cord), but **these categorical tags were completely omitted from the exported dataset**.

> [!IMPORTANT]
> **Definitive Finding:** Anatomical view and body part distributions are **completely unannotated** in the HBFMID release. FractureFuseNet cannot stratify or evaluate by anatomical projection or bone region. This limitation is formally acknowledged.

---

## 4. Rare Class Investigation: The Segmental & Linear Bottleneck

### The `Segmental` Fracture Case ($n=8$ clean images, 2 source groups)
* **Definition:** A fracture where a bone is broken into three or more pieces, with a completely isolated intermediate segment between two separate fracture lines.
* **Empirical Count:** In the clean single-label dataset, exactly 8 images exist, all X-rays.
* **Origin:** All 8 images derive from **exactly 2 source groups**:
  * Source group `70_jpg`: 5 augmented views
  * Source group `225_jpg`: 3 augmented views
* **Excluded Multi-Class Occurrences:** 13 additional raw images contain a Segmental bounding box, but every one of them is tied with another class (e.g., 1 Segmental + 1 Oblique Displaced box in `116_jpg`; 1 Segmental + 1 Comminuted box in `134_jpg`). These are excluded to prevent arbitrary single-label assignment.
* **The Pigeonhole Principle Constraint:**
  $$\text{Given } n=2 \text{ independent source groups and } k=3 \text{ disjoint splits } (\text{Train}, \text{Valid}, \text{Test}),$$
  $$\text{At least one split must receive } 0 \text{ groups: } \lceil n/k \rceil = \lfloor 2/3 \rfloor = 0.$$
  To prevent data leakage, images sharing `70_jpg` or `225_jpg` cannot be divided across splits.
  * Allocation: `70_jpg` (5 images) $\to$ **Train**; `225_jpg` (3 images) $\to$ **Valid**; **Test** receives **0 images**.
* **Implication for Holdout Evaluation:** On a fixed holdout test partition, $N_{\text{test}} = 0$ for `Segmental`. Evaluating a 10-class model on this test set results in an undefined precision/recall/F1 score for Segmental, or artificially penalizes the Macro-F1 if `zero_division=0` is used.

### The `Linear` Fracture Case ($n=22$ clean images, 5 source groups)
* **Empirical Count:** 22 clean images (19 X-ray across 4 groups, 3 MRI in 1 group).
* **Split Allocation:** Because $n=5 \ge 3$, Linear is successfully distributed across all three splits:
  * **Train:** 12 images (3 groups, all X-ray)
  * **Valid:** 3 images (1 group, MRI)
  * **Test:** 7 images (1 group, all X-ray)
* In contrast to the original Roboflow test set (which contained 0 Linear images), our leak-free split provides 7 independent test images for Linear.

---

## 5. Sample Sufficiency & Evaluation Feasibility Matrix

We categorize each class according to its statistical viability for supervised classification and independent holdout testing:

| Class Name | Clean Images | Source Groups | Holdout Test Support | Viability Tier | Primary Benchmark Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Transverse Displaced** | 568 | 96 | 114 (103 X-ray, 11 MRI) | **Tier 1 (Abundant)** | Full Holdout & CV Evaluation |
| **Oblique Displaced** | 307 | 50 | 61 (55 X-ray, 6 MRI) | **Tier 1 (Abundant)** | Full Holdout & CV Evaluation |
| **Transverse** | 93 | 15 | 18 (all X-ray) | **Tier 2 (Sufficient)** | Full Holdout & CV Evaluation |
| **Greenstick** | 83 | 12 | 17 (14 X-ray, 3 MRI) | **Tier 2 (Sufficient)** | Full Holdout & CV Evaluation |
| **Spiral** | 74 | 12 | 15 (all X-ray) | **Tier 2 (Sufficient)** | Full Holdout & CV Evaluation |
| **Healthy** | 64 | 10 | 11 (all X-ray) | **Tier 2 (Sufficient)** | Full Holdout & CV Evaluation |
| **Comminuted** | 158 | 54 | 32 (9 X-ray, 23 MRI) | **Tier 2 (Sufficient\*)** | Full Holdout & CV Evaluation (\*X-ray has 8 groups) |
| **Oblique** | 52 | 8 | 12 (all X-ray) | **Tier 2 (Sufficient)** | Full Holdout & CV Evaluation |
| **Linear** | 22 | 5 | 7 (all X-ray) | **Tier 3 (Marginal)** | Evaluated with Explicit Sample Size Caveat ($N=7$) |
| **Segmental** | 8 | 2 | **0 (Zero Support)** | **Tier 4 (Insufficient)** | **Excluded from Holdout Test Metric; Evaluated via 5-Fold CV** |

---

## 6. Benchmarking Dilemma: 9-Class vs. 10-Class Benchmark

### Why the 9-Class Primary Holdout Benchmark is Mathematically Superior
In standard computer vision and medical benchmark protocols (e.g., CheXpert, MURA, VinDr-CXR):
1. **Non-Zero Metric Denominator:** A valid test classification metric requires non-zero ground-truth support ($N_{\text{support}} > 0$). When a class has $N_{\text{support}} = 0$, Recall is mathematically undefined ($\frac{0}{0}$). Setting it arbitrarily to $0.0$ penalizes the Macro-F1 score by $\frac{1}{K} = 10\%$, distorting the true capability of the model on the remaining 9 classes.
2. **Zero-Leakage Guarantee:** The only way to force Segmental into the test set would be to violate source group independence or sacrifice the validation split entirely.
3. **Precedent in Medical ML:** When a rare subtype lacks sufficient clinical volume to populate a test set, published literature reports the primary holdout benchmark across well-supported classes, while reporting the full spectrum via cross-validation.

### Dual Benchmark Solution (The Defensible Strategy)
We adopt a **Dual-Level Evaluation Protocol** that satisfies both clinical reality and dataset completeness:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FRACTUREFUSENET EVALUATION PROTOCOL                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  [ Primary Benchmark: 9-Class X-Ray Holdout Evaluation ]                               │
│  * Modality: X-ray Only (N = 1,227 total; 743 train, 240 valid, 244 test)             │
│  * Classes: 9 Evaluated Classes (Comminuted, Greenstick, Healthy, Linear, Oblique,     │
│             Oblique Displaced, Spiral, Transverse, Transverse Displaced)               │
│  * Metric: Macro-F1, Weighted-F1, Multi-Class AUROC, Classwise Confusion Matrix        │
│  * Scientific Justification: Every class has >= 1 independent test group (N >= 7)      │
│                              and zero data leakage.                                    │
│                                                                                        │
│  [ Comprehensive Benchmark: 10-Class 5-Fold Grouped Cross-Validation ]                │
│  * Modality: Full Clean Dataset (N = 1,429) & X-ray Subset (N = 1,227)                 │
│  * Classes: All 10 Classes (Including Segmental, n = 8 images across 2 groups)         │
│  * Rotation: Fold 1 evaluates group 225_jpg (3 imgs); Fold 4 evaluates 70_jpg (5 imgs)│
│  * Out-of-Fold Coverage: 100% of all images are evaluated out-of-fold                  │
│  * Documentation: Segmental results explicitly marked with (n = 8, 2 groups)           │
│                                                                                        │
│  [ Exploratory Benchmark: MRI Modality Transfer & Subset Evaluation ]                  │
│  * Modality: MRI Only (N = 202 images across 86 groups)                                │
│  * Classes: 5 Present Classes (Comminuted, Greenstick, Linear, Oblique Displaced,      │
│             Transverse Displaced)                                                      │
│  * Analysis: Measure feature stability, edge/line degradation, and modality shortcut   │
│              vulnerability when models trained on X-ray encounter MRI.                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Treatment of Excluded & Insufficiently Represented Classes

### Policy 1: Excluded Records ($n=110$, $7.15\%$)
1. **Ambiguous Multi-Class Ties ($n=107$ images):**
   * These images contain multiple bounding boxes of conflicting classes with identical box counts (e.g., 1 Comminuted + 1 Transverse box).
   * **Rule:** Excluded from single-label classification training and evaluation. Assigning them based on whichever box occurs first in the text file is scientifically arbitrary and injects label noise.
2. **Unannotated Images ($n=3$ images):**
   * Annotation text files are empty ($0$ bytes, $0$ bounding boxes).
   * **Rule:** Excluded from supervised training.
3. **Audit Trail:** All 110 records are permanently preserved with exact paths, SHA-256 hashes, and rejection reasons in `phase 1/data/manifests/excluded_manifest.csv` and `phase 1/data/excluded_records_audit_log.csv`.

### Policy 2: Insufficiently Represented Class (`Segmental`, $n=8$)
1. **Preservation:** The 8 clean Segmental images are **not** discarded from the dataset.
2. **Training Inclusion:** For 10-class models, group `70_jpg` (5 images) is utilized in training, and group `225_jpg` (3 images) is utilized in validation.
3. **Holdout Test Isolation:** Because $N_{\text{test}} = 0$ in the zero-leakage split, Segmental is omitted from the holdout Macro-F1 denominator, avoiding arbitrary `zero_division` distortions.
4. **Cross-Validation Ingathering:** Segmental performance is fully reported under 5-Fold Grouped Cross-Validation, with explicit documentation of its 2-group limitation.

---

## 8. Final Modality Policy

1. **Modality Separation:** Radiographs (X-ray) and Magnetic Resonance Images (MRI) possess fundamentally different physical contrast mechanisms, resolution properties, and gradient profiles. Pooling them without modality indicators introduces severe confounding.
2. **Primary Modality (X-ray):** Constitutes 85.86% of clean images (1,227 images, 149 source groups) and contains all 10 clinical fracture classes. All primary architecture comparisons (Deep-only baseline vs. Classical-only baseline vs. Dual-Branch FractureFuseNet) will be established on X-ray.
3. **Secondary Modality (MRI):** Constitutes 14.14% of clean images (202 images, 86 source groups) and spans only 5 classes. MRI will be utilized exclusively for secondary exploratory experiments (cross-modality generalization, domain shift evaluation, and modality-specific feature analysis).

---

## 9. Final Class Policy

1. **Dataset Scope:** The dataset maintains the full 10-class taxonomy established by HBFMID.
2. **Primary Benchmark Scope:** The primary holdout evaluation benchmark is defined as **9-Class Fracture Classification** (omitting `Segmental` solely from the holdout test metric due to mathematical support absence).
3. **Secondary Benchmark Scope:** The secondary benchmark is defined as **10-Class 5-Fold Grouped Cross-Validation**, capturing all 10 classes.
4. **Ablation Control:** Both benchmarks will compare:
   * (a) Deep Radiographic Backbone (ResNet-50 / EfficientNet-B0)
   * (b) Classical Edge/Line Classifier (Standardized 64-dim vector with Random Forest / MLP)
   * (c) Proposed FractureFuseNet (Late Fusion Dual-Branch)
   * (d) Random-Edge Control (Late fusion with permuted edge vectors to prove genuine physical edge utility)

---

## 10. Updated Research Objective

> **Formal Research Objective:**  
> "To design, implement, and rigorously benchmark **FractureFuseNet**—a hybrid dual-branch computer vision architecture that fuses deep radiographic convolutional representations with classical edge- and Hough-transform-based fracture-line descriptors—for bone fracture classification. The primary objective is to determine whether classical structural edge features provide orthogonal geometric cues that enhance fine-grained fracture classification on plain X-rays across 9 well-supported classes under leak-free, group-aware validation, and to assess model generalizability and shortcut vulnerability across modalities (X-ray vs. MRI) and rare fracture subtypes."

---

## 11. Recommended Research Title

Based on the actual technical implementation (hybrid dual-branch fusion of deep CNN representations and classical Hough/edge descriptors, evaluated primarily on plain radiographs with secondary exploratory MRI analysis), we recommend the following title:

### Recommended Title (Primary):
> **"FractureFuseNet: Hybrid Late Fusion of Deep Radiographic Representations and Classical Edge-Based Fracture-Line Descriptors for Bone Fracture Classification"**

* **Why this title is superior:**
  1. *"Hybrid Late Fusion":* Accurately describes the fusion mechanism (combining representation learning from CNNs with feature engineering from classical computer vision).
  2. *"Deep Radiographic Representations and Classical Edge-Based Fracture-Line Descriptors":* Directly specifies the two branches (ResNet/EfficientNet + Canny/Hough/Sobel).
  3. *"Bone Fracture Classification":* Conforms precisely to the actual classification task on HBFMID, avoiding misleading claims of full multi-modal cross-attention between X-ray and MRI while maintaining clinical relevance.

### Alternative Options:
* **Option B (Architecture-focused):** *"FractureFuseNet: A Dual-Branch Framework Fusing Convolutional Neural Features and Classical Hough Line Detectors for Radiographic Fracture Classification"*
* **Option C (Preserving Original Wording with Clarification):** *"FractureFuseNet: Late Fusion of Deep Radiographic Features and Classical Edge/Hough Fracture-Line Detection on Multi-Modal Skeletal Imaging"*
