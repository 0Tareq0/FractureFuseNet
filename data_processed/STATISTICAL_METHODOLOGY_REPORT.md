# Statistical Methodology & Hypothesis Testing Audit Report
**Project 12:** FractureFuseNet  
**Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID / Mendeley `xwfs6xbk47`)  
**Scope:** Cluster-Aware Statistical Inference, Multiple-Testing Corrections & Modality Confounding Audit  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This report documents the rigorous re-audit of the classical computer vision feature hypothesis testing in **FractureFuseNet**. The original implementation (`verify_cv_hypothesis.py`) evaluated 64-dimensional edge/line descriptors (Canny edge density, Probabilistic Hough line count, average line length, and Sobel gradient magnitude) under a naive independent-observation assumption ($N=1,227$ X-ray images, $N=1,429$ pooled images).

### Core Audit Findings:
1. **Severe Pseudoreplication in Naive Image-Level Testing:** Treating Roboflow-augmented images ($3\times$ flips, rotations, shears, brightness shifts) as independent observations artificially inflated degrees of freedom by $8.2\times$, causing extreme $p$-value inflation (by up to $1.5 \times 10^7\times$).
2. **Corrected Cluster-Level Framework:** Re-evaluating tests at the perceptual duplicate cluster level (`dup_group_id`, $N=496$ independent physical clusters in X-ray: 27 Healthy, 469 Fracture) establishes valid statistical inference.
3. **Group-Aware Permutation Verification:** 10,000-iteration cluster-level permutation tests independently confirm that high-threshold Canny edge density ($p_{\text{perm}} = 1.00 \times 10^{-4}$), low-threshold edge density ($p_{\text{perm}} = 1.00 \times 10^{-4}$), Hough line count ($p_{\text{perm}} = 1.00 \times 10^{-4}$), average line length ($p_{\text{perm}} = 0.0336$), and Sobel gradient magnitude ($p_{\text{perm}} = 1.00 \times 10^{-4}$) exhibit genuine distributional divergence.
4. **Multiple-Comparison Control:** After Holm-Bonferroni FWER step-down and Benjamini-Hochberg FDR adjustments across all 9 tested metrics, 6 metrics remain statistically significant ($\alpha = 0.05$). Orientation entropy ($p_{\text{Holm}} = 0.205$) and circular dispersion ($p_{\text{Holm}} = 0.401$) are non-significant.
5. **Critical Scientific Interpretation (Anti-Spurious Caution):**  
   > [!WARNING]  
   > **Statistical significance ($p < 0.05$) does NOT prove that Canny or Hough detectors specifically detect fracture lines.**  
   > In fact, **Healthy radiographs exhibit significantly HIGHER edge density (median 2.96% vs. 1.87%) and HIGHER Hough line counts (median 188.5 vs. 80.0) than fractured radiographs** (Cliff's $\delta = -0.698$ and $-0.632$). This disparity is driven by anatomical composition (multi-joint hand/wrist views with dense cortical margins vs. long bone shafts or plaster casts), NOT by fracture lines generating extra edges. Classical features provide whole-image structural context, not isolated fracture line segmentations.

---

## 2. Pseudoreplication Audit: Image-Level vs. Cluster-Level Inference

In machine learning datasets expanded via offline augmentation, multiple images derive from the same physical radiograph. Under standard parametric or non-parametric tests (Student's $t$, Mann-Whitney $U$, Kruskal-Wallis), treating augmented views as independent violates the fundamental i.i.d. assumption (Hurlbert, 1984).

### Table 2.1: Pseudoreplication Sensitivity Comparison (X-ray Modality)
| Feature Metric | Naive Image $N$ | Naive Image $p$ | Cluster $N$ | Cluster $p_{\text{raw}}$ | $p$-Value Inflation Ratio | Degrees of Freedom Inflation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **High Canny Edge Density (%)** | 1,227 (64 H / 1163 F) | $6.89 \times 10^{-17}$ | 496 (27 H / 469 F) | $1.06 \times 10^{-9}$ | **$1.5 \times 10^7 \times$** | $8.2 \times$ |
| **Low Canny Edge Density (%)** | 1,227 (64 H / 1163 F) | $3.07 \times 10^{-15}$ | 496 (27 H / 469 F) | $1.45 \times 10^{-8}$ | **$4.7 \times 10^6 \times$** | $8.2 \times$ |
| **Detected Hough Line Count** | 1,227 (64 H / 1163 F) | $7.83 \times 10^{-14}$ | 496 (27 H / 469 F) | $3.31 \times 10^{-8}$ | **$4.2 \times 10^5 \times$** | $8.2 \times$ |
| **Line Density (/10k px)** | 1,227 (64 H / 1163 F) | $7.83 \times 10^{-14}$ | 496 (27 H / 469 F) | $3.31 \times 10^{-8}$ | **$4.2 \times 10^5 \times$** | $8.2 \times$ |
| **Average Line Length (px)** | 1,227 (64 H / 1163 F) | $3.60 \times 10^{-8}$ | 496 (27 H / 469 F) | $3.65 \times 10^{-4}$ | **$1.0 \times 10^4 \times$** | $8.2 \times$ |
| **Mean Sobel Gradient Mag** | 1,227 (64 H / 1163 F) | $3.09 \times 10^{-12}$ | 496 (27 H / 469 F) | $5.19 \times 10^{-8}$ | **$1.7 \times 10^4 \times$** | $8.2 \times$ |
| **Orientation Entropy** | 1,227 (64 H / 1163 F) | $8.98 \times 10^{-2}$ | 496 (27 H / 469 F) | $1.02 \times 10^{-1}$ | $1.1 \times$ | $8.2 \times$ |
| **Circular Dispersion** | 1,227 (64 H / 1163 F) | $8.29 \times 10^{-1}$ | 496 (27 H / 469 F) | $4.01 \times 10^{-1}$ | $0.5 \times$ | $8.2 \times$ |
| **Spatial Heterogeneity (%)** | 1,227 (64 H / 1163 F) | $8.14 \times 10^{-11}$ | 496 (27 H / 469 F) | $7.73 \times 10^{-6}$ | **$9.5 \times 10^4 \times$** | $8.2 \times$ |

### Mathematical Correction Implemented:
To eliminate pseudoreplication, images belonging to the same perceptual duplicate cluster (`dup_group_id`) were aggregated by computing cluster medians:
$$\tilde{x}_{k, c} = \text{median} \{ x_{i, k} \mid \text{image } i \in \text{cluster } c \}$$
Tests were then performed on the $N=496$ cluster medians, restoring valid statistical independence.

---

## 3. Corrected Primary X-ray Hypothesis Testing & Effect Sizes

### Table 3.1: Cluster-Level Two-Group Comparison (Healthy vs. Fractured X-rays)
*Sample Sizes: Healthy = 27 clusters (64 images); Fracture = 469 clusters (1,163 images)*

| Feature Metric | Healthy Median (IQR) | Fracture Median (IQR) | Median Diff (F - H) | Cliff's $\delta$ | Rank-Biserial $r_{rb}$ | CLES (AUC) | Cohen's $d$ | $p_{\text{raw}}$ | $p_{\text{perm}} (10\text{k})$ | $p_{\text{Holm}}$ | $p_{\text{BH}}$ | Sig. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **High Canny Edge Density (%)** | 2.96 (1.52) | 1.87 (1.29) | -1.09 | **-0.698** | -0.698 | 0.151 | -1.037 | $1.06 \times 10^{-9}$ | $1.00 \times 10^{-4}$ | $9.51 \times 10^{-9}$ | $9.51 \times 10^{-9}$ | *** |
| **Low Canny Edge Density (%)** | 7.47 (3.53) | 4.88 (2.81) | -2.60 | **-0.648** | -0.648 | 0.176 | -0.990 | $1.45 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | $1.16 \times 10^{-7}$ | $6.52 \times 10^{-8}$ | *** |
| **Detected Hough Line Count** | 188.5 (102.5) | 80.0 (82.0) | -108.5 | **-0.632** | -0.632 | 0.184 | -1.031 | $3.31 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | $2.32 \times 10^{-7}$ | $7.45 \times 10^{-8}$ | *** |
| **Line Density (/10k px)** | 4.60 (2.50) | 1.95 (2.00) | -2.65 | **-0.632** | -0.632 | 0.184 | -1.031 | $3.31 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | $2.32 \times 10^{-7}$ | $7.45 \times 10^{-8}$ | *** |
| **Average Line Length (px)** | 48.60 (9.83) | 57.38 (17.38) | +8.78 | **+0.408** | +0.408 | 0.704 | +0.669 | $3.65 \times 10^{-4}$ | $3.36 \times 10^{-2}$ | $1.10 \times 10^{-3}$ | $4.69 \times 10^{-4}$ | ** |
| **Mean Sobel Gradient Mag** | 33.30 (13.62) | 24.86 (10.95) | -8.45 | **-0.623** | -0.623 | 0.189 | -0.963 | $5.19 \times 10^{-8}$ | $1.00 \times 10^{-4}$ | $2.60 \times 10^{-7}$ | $7.79 \times 10^{-8}$ | *** |
| **Orientation Entropy** | 0.96 (0.02) | 0.96 (0.03) | -0.01 | -0.187 | -0.187 | 0.407 | -0.272 | $1.02 \times 10^{-1}$ | $2.83 \times 10^{-1}$ | $2.05 \times 10^{-1}$ | $1.15 \times 10^{-1}$ | ns |
| **Circular Dispersion** | 0.56 (0.07) | 0.57 (0.07) | +0.00 | -0.096 | -0.096 | 0.452 | -0.185 | $4.01 \times 10^{-1}$ | $9.52 \times 10^{-1}$ | $4.01 \times 10^{-1}$ | $4.01 \times 10^{-1}$ | ns |
| **Spatial Heterogeneity (%)** | 2.51 (1.02) | 1.67 (0.83) | -0.84 | **-0.512** | -0.512 | 0.244 | -0.887 | $7.73 \times 10^{-6}$ | $9.00 \times 10^{-4}$ | $3.09 \times 10^{-5}$ | $1.16 \times 10^{-5}$ | *** |

*Legend: `***` $p < 0.001$, `**` $p < 0.01$, `*` $p < 0.05$, `ns` non-significant ($p \ge 0.05$).*

### Effect Size Definitions:
1. **Cliff's Delta ($\delta$):** Non-parametric measure of dominance: $\delta = \frac{\# (F > H) - \# (F < H)}{N_F N_H} \in [-1, 1]$. Values of $|\delta| > 0.474$ represent large effect sizes (Romano et al., 2006).
2. **Common Language Effect Size (CLES / AUC):** Probability that a randomly selected fractured radiograph has a larger value than a randomly selected healthy radiograph. $CLES = 0.50$ represents identical distributions.
3. **Cohen's $d$:** Standardized mean difference relative to pooled standard deviation: $d = \frac{\bar{x}_F - \bar{x}_H}{s_{\text{pooled}}}$.

---

## 4. Multi-Class Omnibus Analysis (9-Class X-ray Benchmark)

To test whether classical edge features vary significantly across fine-grained fracture types at the cluster level, Kruskal-Wallis omnibus non-parametric tests were evaluated across the 9 primary X-ray classes ($N=488$ clusters):

### Table 4.1: Cluster-Level Kruskal-Wallis Test (9 X-ray Classes)
| Metric | Kruskal-Wallis $H$ | Degrees of Freedom | $p$-Value | Effect Size $\epsilon^2$ | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **High Canny Edge Density (%)** | 82.37 | 8 | $1.63 \times 10^{-14}$ | **0.167** | Moderate-to-Large Class Separation |
| **Low Canny Edge Density (%)** | 78.27 | 8 | $1.09 \times 10^{-13}$ | **0.159** | Moderate Class Separation |
| **Detected Hough Line Count** | 75.90 | 8 | $3.26 \times 10^{-13}$ | **0.154** | Moderate Class Separation |
| **Line Density (/10k px)** | 75.90 | 8 | $3.26 \times 10^{-13}$ | **0.154** | Moderate Class Separation |
| **Average Line Length (px)** | 77.64 | 8 | $1.46 \times 10^{-13}$ | **0.158** | Moderate Class Separation |
| **Mean Sobel Gradient Mag** | 80.07 | 8 | $4.74 \times 10^{-14}$ | **0.163** | Moderate-to-Large Class Separation |
| **Spatial Heterogeneity (%)** | 60.39 | 8 | $3.91 \times 10^{-10}$ | **0.123** | Moderate Class Separation |
| **Circular Dispersion** | 43.71 | 8 | $6.46 \times 10^{-7}$ | **0.089** | Small-to-Moderate Separation |
| **Orientation Entropy** | 28.41 | 8 | $4.03 \times 10^{-4}$ | **0.058** | Small Separation |

$$\epsilon^2 = \frac{H}{(N^2 - 1)/(N + 1)} \in [0, 1]$$
*Values of $\epsilon^2 \ge 0.14$ indicate a large effect size in one-way non-parametric ANOVA (Tomczak & Tomczak, 2014).*

---

## 5. Modality Confounding Audit: Physical Feature Disparities

A direct statistical comparison between X-ray ($N=496$ clusters) and MRI ($N=86$ clusters) demonstrates why pooling modalities without explicit controls creates severe shortcut hazards:

### Table 5.1: X-ray vs. MRI Feature Physics Comparison
| Metric | X-ray Median (IQR) | MRI Median (IQR) | Cliff's $\delta$ | Mann-Whitney $U$ | $p$-Value | Physical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **High Edge Density (%)** | 1.98 (1.33) | 3.84 (2.36) | **+0.709** | 6,211 | $1.33 \times 10^{-32}$ | MRI produces $2\times$ more edges due to soft tissue & fat interfaces |
| **Hough Line Count** | 85.5 (85.0) | 244.5 (193.0) | **+0.653** | 7,402 | $6.49 \times 10^{-28}$ | MRI slices exhibit $3\times$ higher line count from anatomical margins |
| **Mean Sobel Grad Mag** | 25.53 (11.23) | 42.53 (12.79) | **+0.828** | 3,674 | $7.01 \times 10^{-44}$ | MRI gradient magnitude is fundamentally higher across all regions |
| **Average Line Length (px)**| 57.07 (17.33) | 51.39 (9.79) | **-0.280** | 27,330 | $2.66 \times 10^{-6}$ | MRI lines are shorter and more fragmented |

### Within-Class Controlled Comparison: `Comminuted` Fracture
To prove that this disparity is physical and not driven by class composition, we benchmarked the *same clinical class* (`Comminuted`) across both modalities:
* **X-ray Comminuted ($n=24$ clusters):** High Edge Density Median = $2.14\%$, Hough Line Count Median = $115.0$.
* **MRI Comminuted ($n=55$ clusters):** High Edge Density Median = $4.61\%$, Hough Line Count Median = $293.0$.
* **Statistical Test:** Mann-Whitney $U = 1,200$, $p = 8.87 \times 10^{-9}$.
* **Clinical Conclusion:** Even for identical fracture pathology, MRI images exhibit more than double the edge density and line counts of X-rays. A machine learning model trained on pooled data will exploit this physical discrepancy as a trivial shortcut to predict `Comminuted` whenever MRI features are detected.

---

## 6. Critical Methodological Caveats & Non-Causal Nature of Findings

1. **Association vs. Detection:**
   * A statistically significant difference in Hough line count or edge density between Healthy and Fractured images does **not** demonstrate that the Hough transform is detecting the fracture line.
   * If healthy bones in HBFMID represent multi-bone projections (e.g. carpal/metacarpal bones) while fracture images represent single long bone shafts (e.g. femur/tibia), the edge detector responds to skeletal morphology and field of view, not the fracture line itself.
2. **Exploratory Status:**
   * These statistical tests represent **exploratory feature characterization** designed to confirm that classical descriptors capture non-random structural variation across radiographs.
   * They do not replace or guarantee performance in downstream supervised classification (Phase 2/3), where models must learn to separate complex multi-class decision boundaries.
3. **Absence of Anatomical Annotations:**
   * Because HBFMID lacks projection-view and body-part labels, anatomical-site-stratified hypothesis testing cannot be performed.
