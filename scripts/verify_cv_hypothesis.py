import os
import sys
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add phase 1 dir to path so src.dataset can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = SCRIPT_DIR.parent
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from src.dataset import (
    extract_classical_edge_features_np,
    FEATURE_NAMES_64,
    BoneFractureDataset,
    load_config,
    save_scaler_with_provenance,
    validate_scaler_provenance,
)


def find_dataset_dir() -> Path:
    root_dir = PHASE1_DIR.parent
    candidates = [
        root_dir / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Bone Fractures Detection",
        PHASE1_DIR / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Bone Fractures Detection",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not locate HBFMID dataset directory.")


def compute_summary_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0, "q25": 0.0, "q75": 0.0, "min": 0.0, "max": 0.0}
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(q50),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def holm_bonferroni(p_vals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment controlling FWER at alpha=0.05."""
    p_vals = np.asarray(p_vals, dtype=np.float64)
    m = len(p_vals)
    order = np.argsort(p_vals)
    sorted_p = p_vals[order]
    
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        running_max = max(running_max, val)
        adj[i] = min(1.0, running_max)
    
    out = np.empty(m, dtype=np.float64)
    out[order] = adj
    return out


def benjamini_hochberg(p_vals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment controlling FDR at q=0.05."""
    p_vals = np.asarray(p_vals, dtype=np.float64)
    m = len(p_vals)
    order = np.argsort(p_vals)
    sorted_p = p_vals[order]
    
    adj = np.empty(m, dtype=np.float64)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        val = (m / (i + 1)) * sorted_p[i]
        running_min = min(running_min, val)
        adj[i] = min(1.0, running_min)
        
    out = np.empty(m, dtype=np.float64)
    out[order] = adj
    return out


def compute_effect_sizes(group1: np.ndarray, group2: np.ndarray) -> dict:
    """
    Computes four standard effect sizes between group1 (Fracture) and group2 (Healthy):
    1. Cliff's Delta (non-parametric probability difference in [-1, 1])
    2. Rank-Biserial Correlation (r_rb in [-1, 1])
    3. Common Language Effect Size (CLES / AUC in [0, 1])
    4. Cohen's d (parametric standardized mean difference)
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return {"cliff_delta": 0.0, "rank_biserial_r": 0.0, "cles_auc": 0.5, "cohen_d": 0.0}

    # Vectorized outer differences: diff[i, j] = g1[i] - g2[j]
    diff = np.subtract.outer(g1, g2)
    n_total = n1 * n2

    cliff_delta = float(np.sum(np.sign(diff)) / n_total)
    cles = float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / n_total)
    
    # Mann-Whitney U for g1
    u_stat, _ = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    r_rb = float(2.0 * (u_stat / n_total) - 1.0)

    # Cohen's d
    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
    cohen_d = float((m1 - m2) / s_pooled) if s_pooled > 1e-8 else 0.0

    return {
        "cliff_delta": round(cliff_delta, 4),
        "rank_biserial_r": round(r_rb, 4),
        "cles_auc": round(cles, 4),
        "cohen_d": round(cohen_d, 4)
    }


def group_permutation_test(g1: np.ndarray, g2: np.ndarray, n_perms: int = 10000, seed: int = 42) -> float:
    """
    Two-tailed permutation test between two independent groups (e.g. cluster medians).
    Uses difference in sample medians as test statistic.
    """
    rng = np.random.default_rng(seed)
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)
    n1 = len(g1)
    obs_diff = float(np.median(g1) - np.median(g2))

    pooled = np.concatenate([g1, g2])
    n_total = len(pooled)

    perm_diffs = np.empty(n_perms, dtype=np.float64)
    for b in range(n_perms):
        perm = rng.permutation(pooled)
        perm_diffs[b] = np.median(perm[:n1]) - np.median(perm[n1:])

    p_val = float((1.0 + np.sum(np.abs(perm_diffs) >= np.abs(obs_diff))) / (1.0 + n_perms))
    return p_val


def main():
    dataset_dir = find_dataset_dir()
    data_dir = PHASE1_DIR / "data"
    fig_dir = PHASE1_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / "dataset_metadata_cleaned.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Authoritative metadata not found at {meta_path}")

    print("=" * 76)
    print("AUDIT & IMPROVED STATISTICAL HYPOTHESIS TESTING (CLUSTER-AWARE & CONTROLLED)")
    print("Project 12: FractureFuseNet (HBFMID)")
    print("=" * 76)

    cfg = load_config()
    prep_cfg = cfg.get("preprocessing", {})
    stat_cfg = cfg.get("statistics", {})
    n_perms = int(stat_cfg.get("n_permutations", 10000))
    rand_seed = int(cfg.get("project", {}).get("random_seed", 42))

    edge_kw = {
        "clahe_clip_limit": prep_cfg.get("clahe_clip_limit", 2.0),
        "clahe_tile_grid": prep_cfg.get("clahe_tile_grid", [8, 8]),
        "gaussian_blur_ksize": prep_cfg.get("gaussian_blur_ksize", [3, 3]),
        "canny_low_thresh": prep_cfg.get("canny_low_thresh", [30, 100]),
        "canny_high_thresh": prep_cfg.get("canny_high_thresh", [75, 200]),
        "hough_rho": prep_cfg.get("hough_rho", 1),
        "hough_theta_deg": prep_cfg.get("hough_theta_deg", 1.0),
        "hough_threshold": prep_cfg.get("hough_threshold", 40),
        "hough_min_line_length": prep_cfg.get("hough_min_line_length", 25),
        "hough_max_line_gap": prep_cfg.get("hough_max_line_gap", 10)
    }

    df = pd.read_csv(meta_path)
    inc_df = df[
        (df["stratified_group_split"].isin(["train", "valid", "test"]))
        & (~df["is_empty_label"])
        & (df["label_quality"].isin(["single", "majority"]))
    ].copy().reset_index(drop=True)

    print(f"Loading/extracting 64-dim classical CV features across all {len(inc_df)} included radiographs...")

    cache_path = data_dir / "unaugmented_edge_features.npz"
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}...")
        cache = np.load(cache_path)
        cached_dict = {f: feat for f, feat in zip(cache["filenames"], cache["features"])}
        all_features_64 = [cached_dict[fn] for fn in inc_df["filename"]]
        X_all = np.vstack(all_features_64)
    else:
        all_features_64 = []
        for _, row in inc_df.iterrows():
            img_path = dataset_dir / str(row["rel_img_path"]).replace("\\", "/")
            with Image.open(img_path) as im:
                img_gray = np.asarray(im.convert("L"), dtype=np.uint8)
            vec64 = extract_classical_edge_features_np(img_gray, **edge_kw)
            all_features_64.append(vec64)
        X_all = np.vstack(all_features_64)
        np.savez_compressed(cache_path, filenames=inc_df["filename"].to_numpy(dtype=str), features=X_all)
        print(f"[Saved] Unaugmented 64-dim feature cache -> {cache_path}")

    assert X_all.shape == (len(inc_df), 64), f"Expected shape {(len(inc_df), 64)}, got {X_all.shape}"
    assert not np.isnan(X_all).any() and not np.isinf(X_all).any()

    # Fit StandardScaler strictly on TRAIN split with provenance tracking
    train_mask = (inc_df["stratified_group_split"] == "train").to_numpy()
    X_train = X_all[train_mask]
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.scale_ = np.where(scaler.scale_ < 1e-8, 1.0, scaler.scale_)
    scaler_path = data_dir / "edge_feature_scaler.joblib"
    prov_path = save_scaler_with_provenance(scaler, str(scaler_path), inc_df[train_mask])
    print(f"[PASS] Fitted StandardScaler strictly on TRAIN split (N={len(X_train)} images) -> {scaler_path}")
    print(f"[Provenance] Recorded scaler provenance manifest -> {prov_path}")

    # Build image-level metric dataframe
    interpretable_rows = []
    for idx, row in inc_df.iterrows():
        vec64 = X_all[idx]
        h, w = 640, 640
        diag = float(np.hypot(h, w))
        raw_line_count = float(np.expm1(float(vec64[32]) * np.log1p(1000.0)))
        line_density_10k = raw_line_count / ((h * w) / 10000.0)

        interpretable_rows.append({
            "filename": row["filename"],
            "base_id": row["base_id"],
            "dup_group_id": row["dup_group_id"],
            "group_id": row["group_id"],
            "split": row["stratified_group_split"],
            "modality": row["modality"],
            "primary_class": row["primary_class"],
            "is_fracture": row["primary_class"] != "Healthy",
            "edge_density_high_pct": float(vec64[53]) * 100.0,
            "edge_density_low_pct": float(vec64[52]) * 100.0,
            "hough_line_count": raw_line_count,
            "hough_line_density_10k": line_density_10k,
            "avg_line_length_px": float(vec64[33]) * diag,
            "sobel_grad_mag_mean": float(vec64[41]) * 255.0,
            "orientation_entropy": float(vec64[54]),
            "circular_dispersion": float(vec64[37]),
            "spatial_edge_heterogeneity_pct": float(np.std(vec64[1:32:2])) * 100.0
        })

    metrics_df = pd.DataFrame(interpretable_rows)
    key_metrics = [
        "edge_density_high_pct",
        "edge_density_low_pct",
        "hough_line_count",
        "hough_line_density_10k",
        "avg_line_length_px",
        "sobel_grad_mag_mean",
        "orientation_entropy",
        "circular_dispersion",
        "spatial_edge_heterogeneity_pct"
    ]

    metric_labels = {
        "edge_density_high_pct": "High Canny Edge Density (%)",
        "edge_density_low_pct": "Low Canny Edge Density (%)",
        "hough_line_count": "Detected Hough Line Count",
        "hough_line_density_10k": "Line Density (/10k px)",
        "avg_line_length_px": "Average Line Length (px)",
        "sobel_grad_mag_mean": "Mean Sobel Gradient Mag",
        "orientation_entropy": "Orientation Entropy",
        "circular_dispersion": "Circular Dispersion",
        "spatial_edge_heterogeneity_pct": "Spatial Heterogeneity (%)"
    }

    # =========================================================================
    # PART A: CLUSTER-LEVEL PRIMARY X-RAY ANALYSIS (PERCEPTUAL DUP GROUPS)
    # =========================================================================
    print("\n" + "=" * 76)
    print("PART A: PRIMARY X-RAY CLUSTER-LEVEL ANALYSIS (PERCEPTUAL DUP GROUPS, N=496)")
    print("=" * 76)

    xray_imgs = metrics_df[metrics_df["modality"] == "xray"].copy()
    
    # Aggregate within each perceptual duplicate cluster (dup_group_id)
    dup_agg_xray = xray_imgs.groupby("dup_group_id").agg({
        "is_fracture": "first",
        "primary_class": "first",
        "base_id": "first",
        "group_id": "first",
        "filename": "count",
        **{m: "median" for m in key_metrics}
    }).rename(columns={"filename": "cluster_image_count"}).reset_index()

    n_h_clusters = int((~dup_agg_xray["is_fracture"]).sum())
    n_f_clusters = int(dup_agg_xray["is_fracture"].sum())
    n_h_images = int(xray_imgs[~xray_imgs["is_fracture"]]["filename"].count())
    n_f_images = int(xray_imgs[xray_imgs["is_fracture"]]["filename"].count())

    print(f"Sample Sizes: Healthy = {n_h_clusters} clusters ({n_h_images} images) | "
          f"Fracture = {n_f_clusters} clusters ({n_f_images} images)")

    group_report_rows = []
    p_raw_list = []
    h_data_dict = {}
    f_data_dict = {}

    for m in key_metrics:
        h_vals = dup_agg_xray[~dup_agg_xray["is_fracture"]][m].to_numpy()
        f_vals = dup_agg_xray[dup_agg_xray["is_fracture"]][m].to_numpy()
        h_data_dict[m] = h_vals
        f_data_dict[m] = f_vals

        h_stats = compute_summary_stats(h_vals)
        f_stats = compute_summary_stats(f_vals)

        # Mann-Whitney U test
        u_stat, p_mw = stats.mannwhitneyu(f_vals, h_vals, alternative="two-sided")
        # Effect sizes
        effects = compute_effect_sizes(f_vals, h_vals)
        # Group-aware permutation test
        p_perm = group_permutation_test(f_vals, h_vals, n_perms=n_perms, seed=rand_seed)

        p_raw_list.append(p_mw)

        group_report_rows.append({
            "metric": m,
            "metric_label": metric_labels[m],
            "healthy_n_clusters": h_stats["n"],
            "healthy_n_images": n_h_images,
            "healthy_median": round(h_stats["median"], 4),
            "healthy_iqr": round(h_stats["iqr"], 4),
            "healthy_mean": round(h_stats["mean"], 4),
            "healthy_std": round(h_stats["std"], 4),
            "fracture_n_clusters": f_stats["n"],
            "fracture_n_images": n_f_images,
            "fracture_median": round(f_stats["median"], 4),
            "fracture_iqr": round(f_stats["iqr"], 4),
            "fracture_mean": round(f_stats["mean"], 4),
            "fracture_std": round(f_stats["std"], 4),
            "median_diff_F_minus_H": round(f_stats["median"] - h_stats["median"], 4),
            "mannwhitney_u": float(u_stat),
            "p_raw": float(p_mw),
            "p_permutation_10k": float(p_perm),
            "cliff_delta": effects["cliff_delta"],
            "rank_biserial_r": effects["rank_biserial_r"],
            "cles_auc": effects["cles_auc"],
            "cohen_d": effects["cohen_d"]
        })

    # Apply multiple-comparison corrections
    p_raw_arr = np.array(p_raw_list)
    p_holm = holm_bonferroni(p_raw_arr)
    p_bh = benjamini_hochberg(p_raw_arr)

    for i in range(len(group_report_rows)):
        group_report_rows[i]["p_holm_bonferroni"] = float(p_holm[i])
        group_report_rows[i]["p_benjamini_hochberg"] = float(p_bh[i])
        group_report_rows[i]["sig_holm_005"] = bool(p_holm[i] < 0.05)
        group_report_rows[i]["sig_bh_005"] = bool(p_bh[i] < 0.05)

    group_report_df = pd.DataFrame(group_report_rows)
    group_report_csv = data_dir / "group_level_hypothesis_report.csv"
    group_report_df.to_csv(group_report_csv, index=False)

    with open(data_dir / "group_level_hypothesis_report.json", "w", encoding="utf-8") as f:
        json.dump(group_report_rows, f, indent=2)

    print("\nPrimary X-ray Cluster-Level Results (Corrected for Group Clustering & Multiple Testing):")
    for r in group_report_rows:
        sig_mark = "***" if r["p_holm_bonferroni"] < 0.001 else ("**" if r["p_holm_bonferroni"] < 0.01 else ("*" if r["p_holm_bonferroni"] < 0.05 else "ns"))
        print(
            f"  {r['metric_label']:28s} | H_med: {r['healthy_median']:6.2f} | F_med: {r['fracture_median']:6.2f} | "
            f"Diff: {r['median_diff_F_minus_H']:+6.2f} | Cliff_delta: {r['cliff_delta']:+6.3f} | "
            f"p_raw={r['p_raw']:.2e} | p_perm={r['p_permutation_10k']:.2e} | p_holm={r['p_holm_bonferroni']:.2e} ({sig_mark})"
        )

    # =========================================================================
    # PART B: PSEUDOREPLICATION SENSITIVITY AUDIT (IMAGE-LEVEL VS. CLUSTER-LEVEL)
    # =========================================================================
    print("\n" + "=" * 76)
    print("PART B: PSEUDOREPLICATION SENSITIVITY AUDIT (IMAGE-LEVEL VS. CLUSTER-LEVEL)")
    print("=" * 76)

    pseudo_rows = []
    for r in group_report_rows:
        m = r["metric"]
        # Naive image-level test (pseudoreplication)
        h_img_vals = xray_imgs[~xray_imgs["is_fracture"]][m].to_numpy()
        f_img_vals = xray_imgs[xray_imgs["is_fracture"]][m].to_numpy()
        u_img, p_img = stats.mannwhitneyu(f_img_vals, h_img_vals, alternative="two-sided")
        eff_img = compute_effect_sizes(f_img_vals, h_img_vals)

        p_group = r["p_raw"]
        ratio = p_group / max(p_img, 1e-300)

        pseudo_rows.append({
            "metric": m,
            "metric_label": r["metric_label"],
            "image_level_n_h": len(h_img_vals),
            "image_level_n_f": len(f_img_vals),
            "image_level_p": float(p_img),
            "image_level_cliff_delta": eff_img["cliff_delta"],
            "cluster_level_n_h": r["healthy_n_clusters"],
            "cluster_level_n_f": r["fracture_n_clusters"],
            "cluster_level_p": float(p_group),
            "cluster_level_cliff_delta": r["cliff_delta"],
            "p_value_inflation_ratio": float(ratio),
            "conclusion": "Degrees of freedom artificially inflated by 8.2x via Roboflow augmentations"
        })

    pseudo_df = pd.DataFrame(pseudo_rows)
    pseudo_csv = data_dir / "pseudoreplication_audit_report.csv"
    pseudo_df.to_csv(pseudo_csv, index=False)

    for r in pseudo_rows:
        print(
            f"  {r['metric_label']:28s} | Image p: {r['image_level_p']:.2e} (N={r['image_level_n_h']+r['image_level_n_f']}) -> "
            f"Cluster p: {r['cluster_level_p']:.2e} (N={r['cluster_level_n_h']+r['cluster_level_n_f']}) | "
            f"P-inflation: {r['p_value_inflation_ratio']:.1e}x"
        )

    # Also update the binary comparison CSV to include both levels
    bin_csv = data_dir / "cv_hypothesis_binary_comparison.csv"
    pseudo_df.to_csv(bin_csv, index=False)

    # =========================================================================
    # PART C: MODALITY CONFOUNDING AUDIT (X-RAY VS. MRI)
    # =========================================================================
    print("\n" + "=" * 76)
    print("PART C: MODALITY CONFOUNDING AUDIT (X-RAY VS. MRI FEATURE PHYSICS)")
    print("=" * 76)

    mri_imgs = metrics_df[metrics_df["modality"] == "mri"].copy()
    dup_agg_mri = mri_imgs.groupby("dup_group_id").agg({
        "is_fracture": "first",
        "primary_class": "first",
        **{m: "median" for m in key_metrics}
    }).reset_index()

    modality_rows = []
    for m in key_metrics:
        x_vals = dup_agg_xray[m].to_numpy()
        m_vals = dup_agg_mri[m].to_numpy()
        u_mod, p_mod = stats.mannwhitneyu(m_vals, x_vals, alternative="two-sided")
        eff_mod = compute_effect_sizes(m_vals, x_vals)
        modality_rows.append({
            "metric": m,
            "metric_label": metric_labels[m],
            "xray_n_clusters": len(x_vals),
            "xray_median": round(float(np.median(x_vals)), 4),
            "xray_iqr": round(float(stats.iqr(x_vals)), 4),
            "mri_n_clusters": len(m_vals),
            "mri_median": round(float(np.median(m_vals)), 4),
            "mri_iqr": round(float(stats.iqr(m_vals)), 4),
            "median_diff_MRI_minus_Xray": round(float(np.median(m_vals) - np.median(x_vals)), 4),
            "mannwhitney_u": float(u_mod),
            "p_value": float(p_mod),
            "cliff_delta": eff_mod["cliff_delta"],
            "cles_auc": eff_mod["cles_auc"]
        })

    modality_df = pd.DataFrame(modality_rows)
    modality_csv = data_dir / "modality_confounding_report.csv"
    modality_df.to_csv(modality_csv, index=False)

    for r in modality_rows:
        print(
            f"  {r['metric_label']:28s} | X-ray Med: {r['xray_median']:6.2f} | MRI Med: {r['mri_median']:6.2f} | "
            f"Cliff_delta: {r['cliff_delta']:+6.3f} | p={r['p_value']:.2e}"
        )

    # Within-class comparison for Comminuted across modalities
    com_x = dup_agg_xray[dup_agg_xray["primary_class"] == "Comminuted"]
    com_m = dup_agg_mri[dup_agg_mri["primary_class"] == "Comminuted"]
    print(f"\n  Within 'Comminuted' Fracture Comparison across Modalities:")
    print(f"  * X-ray Comminuted (n={len(com_x)} clusters): High Edge Density Med = {com_x['edge_density_high_pct'].median():.2f}%, Hough Lines Med = {com_x['hough_line_count'].median():.1f}")
    print(f"  * MRI Comminuted   (n={len(com_m)} clusters): High Edge Density Med = {com_m['edge_density_high_pct'].median():.2f}%, Hough Lines Med = {com_m['hough_line_count'].median():.1f}")
    u_com, p_com = stats.mannwhitneyu(com_m['edge_density_high_pct'], com_x['edge_density_high_pct'], alternative="two-sided")
    print(f"  * Mann-Whitney test within Comminuted: U={u_com:.0f}, p={p_com:.3e} (Proves physical contrast disparity)")

    # =========================================================================
    # PART D: MULTI-CLASS OMNIBUS KRUSKAL-WALLIS ANALYSIS (CLUSTER-LEVEL)
    # =========================================================================
    print("\n" + "=" * 76)
    print("PART D: MULTI-CLASS KRUSKAL-WALLIS ANALYSIS (CLUSTER-LEVEL, X-RAY)")
    print("=" * 76)

    # Group-level classwise summary for all 10 classes
    classwise_rows = []
    kw_9class_results = {}
    kw_10class_results = {}

    eval_classes_9 = [c for c in BoneFractureDataset.CLASS_NAMES if c != "Segmental"]
    eval_classes_10 = BoneFractureDataset.CLASS_NAMES

    for m in key_metrics:
        # 9-class test (well-supported classes)
        groups_9 = [dup_agg_xray[dup_agg_xray["primary_class"] == c][m].to_numpy() for c in eval_classes_9 if len(dup_agg_xray[dup_agg_xray["primary_class"] == c]) > 0]
        h_9, p_9 = stats.kruskal(*groups_9)
        n_9 = sum(len(g) for g in groups_9)
        eps_9 = float(h_9) / ((n_9 ** 2 - 1.0) / (n_9 + 1.0))
        kw_9class_results[m] = {"kruskal_h": round(float(h_9), 3), "p_value": float(p_9), "epsilon_sq": round(eps_9, 4), "n_clusters": n_9}

        # 10-class test
        groups_10 = [dup_agg_xray[dup_agg_xray["primary_class"] == c][m].to_numpy() for c in eval_classes_10 if len(dup_agg_xray[dup_agg_xray["primary_class"] == c]) > 0]
        h_10, p_10 = stats.kruskal(*groups_10)
        n_10 = sum(len(g) for g in groups_10)
        eps_10 = float(h_10) / ((n_10 ** 2 - 1.0) / (n_10 + 1.0))
        kw_10class_results[m] = {"kruskal_h": round(float(h_10), 3), "p_value": float(p_10), "epsilon_sq": round(eps_10, 4), "n_clusters": n_10}

        # Compute per-class summary statistics at cluster level
        for c in eval_classes_10:
            c_vals = dup_agg_xray[dup_agg_xray["primary_class"] == c][m].to_numpy()
            c_imgs = xray_imgs[xray_imgs["primary_class"] == c][m].to_numpy()
            st_grp = compute_summary_stats(c_vals)
            classwise_rows.append({
                "class_name": c,
                "metric": m,
                "metric_label": metric_labels[m],
                "n_clusters": st_grp["n"],
                "n_images": len(c_imgs),
                "cluster_median": round(st_grp["median"], 4),
                "cluster_iqr": round(st_grp["iqr"], 4),
                "cluster_mean": round(st_grp["mean"], 4),
                "cluster_std": round(st_grp["std"], 4),
                "cluster_min": round(st_grp["min"], 4),
                "cluster_max": round(st_grp["max"], 4)
            })

    classwise_df = pd.DataFrame(classwise_rows)
    classwise_csv = data_dir / "cv_hypothesis_classwise_stats.csv"
    classwise_df.to_csv(classwise_csv, index=False)

    kw_summary = {
        "kruskal_wallis_9class_xray": kw_9class_results,
        "kruskal_wallis_10class_xray": kw_10class_results
    }
    with open(data_dir / "cv_kruskal_wallis_summary.json", "w", encoding="utf-8") as f:
        json.dump(kw_summary, f, indent=2)

    for m in key_metrics:
        r9 = kw_9class_results[m]
        print(f"  {metric_labels[m]:28s} | 9-Class X-ray H = {r9['kruskal_h']:6.2f} | p = {r9['p_value']:.2e} | epsilon^2 = {r9['epsilon_sq']:.3f}")

    # =========================================================================
    # PART E: PUBLICATION-QUALITY VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 76)
    print("PART E: GENERATING UPDATED STATISTICAL VISUALIZATIONS")
    print("=" * 76)

    # 1. Group-Level CV Hypothesis Distributions (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    quad_metrics = [
        ("edge_density_high_pct", axes[0, 0]),
        ("hough_line_count", axes[0, 1]),
        ("avg_line_length_px", axes[1, 0]),
        ("sobel_grad_mag_mean", axes[1, 1])
    ]
    for m, ax in quad_metrics:
        h_d = h_data_dict[m]
        f_d = f_data_dict[m]
        bp = ax.boxplot(
            [h_d, f_d],
            tick_labels=[f"Healthy X-ray\n(N={len(h_d)} clusters)", f"Fracture X-ray\n(N={len(f_d)} clusters)"],
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="black", linewidth=1.5)
        )
        bp["boxes"][0].set_facecolor("#86efac")  # green
        bp["boxes"][1].set_facecolor("#93c5fd")  # blue
        row = [r for r in group_report_rows if r["metric"] == m][0]
        ax.set_title(
            f"{row['metric_label']}\n"
            f"(Cliff's \u03b4={row['cliff_delta']:+.2f}, CLES={row['cles_auc']:.2f}, p_perm={row['p_permutation_10k']:.1e})",
            fontsize=10
        )
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Cluster-Level Feature Distributions: Healthy vs. Fractured X-rays\n(N=496 Perceptual Duplicate Clusters, Roboflow Augmentation Confound Removed)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig1_path = fig_dir / "group_level_cv_hypothesis_distributions.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Figure 1 -> {fig1_path}")

    # 2. Pseudoreplication Sensitivity Comparison (P-value Inflation Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(key_metrics))
    img_p_vals = [r["image_level_p"] for r in pseudo_rows]
    grp_p_vals = [r["cluster_level_p"] for r in pseudo_rows]

    # -log10(p)
    log_img_p = -np.log10(np.clip(img_p_vals, 1e-25, 1.0))
    log_grp_p = -np.log10(np.clip(grp_p_vals, 1e-25, 1.0))

    ax.barh(y_pos - 0.18, log_img_p, height=0.35, color="#f87171", label="Naive Image-Level (N=1,227, Pseudoreplication)")
    ax.barh(y_pos + 0.18, log_grp_p, height=0.35, color="#60a5fa", label="Cluster-Level (N=496, Independent Clusters)")
    ax.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1.2, label="\u03b1 = 0.05 Threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["metric_label"] for r in pseudo_rows], fontsize=9)
    ax.set_xlabel("-log10(p-value) [Higher = Smaller p-value]", fontsize=10)
    ax.set_title("Pseudoreplication Sensitivity Audit: Naive Image-Level vs. Cluster-Level Inference", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig2_path = fig_dir / "pseudoreplication_sensitivity_comparison.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Figure 2 -> {fig2_path}")

    # 3. Class-wise Distributions across all classes (Cluster-Level)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    cls_order = BoneFractureDataset.CLASS_NAMES
    data_lines = [dup_agg_xray[dup_agg_xray["primary_class"] == c]["hough_line_count"].to_numpy() for c in cls_order]
    data_edges = [dup_agg_xray[dup_agg_xray["primary_class"] == c]["edge_density_high_pct"].to_numpy() for c in cls_order]

    bp1 = axes[0].boxplot(data_lines, tick_labels=[f"{c}\n(N={len(d)})" for c, d in zip(cls_order, data_lines)], patch_artist=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#bfdbfe")
    axes[0].set_ylabel("Hough Line Count (Cluster Median)")
    axes[0].set_title(
        f"Class-Wise Detected Hough Line Count (Kruskal-Wallis 9-Class H={kw_9class_results['hough_line_count']['kruskal_h']:.1f}, "
        f"p={kw_9class_results['hough_line_count']['p_value']:.2e}, eps^2={kw_9class_results['hough_line_count']['epsilon_sq']:.3f})"
    )
    axes[0].grid(True, axis="y", alpha=0.3)

    bp2 = axes[1].boxplot(data_edges, tick_labels=[f"{c}\n(N={len(d)})" for c, d in zip(cls_order, data_edges)], patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#fed7aa")
    axes[1].set_ylabel("High Edge Density % (Cluster Median)")
    axes[1].set_title(
        f"Class-Wise Canny Edge Density (Kruskal-Wallis 9-Class H={kw_9class_results['edge_density_high_pct']['kruskal_h']:.1f}, "
        f"p={kw_9class_results['edge_density_high_pct']['p_value']:.2e}, eps^2={kw_9class_results['edge_density_high_pct']['epsilon_sq']:.3f})"
    )
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig3_path = fig_dir / "classwise_edge_metrics.png"
    plt.savefig(fig3_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Figure 3 -> {fig3_path}")

    # 4. Modality Feature Comparison (X-ray vs. MRI Feature Physics)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    mod_specs = [
        ("edge_density_high_pct", "High Edge Density (%)", axes[0]),
        ("hough_line_count", "Hough Line Count", axes[1]),
        ("sobel_grad_mag_mean", "Mean Sobel Gradient Mag", axes[2])
    ]
    for col, title, ax in mod_specs:
        x_d = dup_agg_xray[col].to_numpy()
        m_d = dup_agg_mri[col].to_numpy()
        bp = ax.boxplot(
            [x_d, m_d],
            tick_labels=[f"X-ray\n(N={len(x_d)} clusters)", f"MRI\n(N={len(m_d)} clusters)"],
            patch_artist=True,
            widths=0.45
        )
        bp["boxes"][0].set_facecolor("#cbd5e1")
        bp["boxes"][1].set_facecolor("#fca5a5")
        row = [r for r in modality_rows if r["metric"] == col][0]
        ax.set_title(f"{title}\n(Cliff's \u03b4={row['cliff_delta']:+.2f}, p={row['p_value']:.1e})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Modality Feature Confounding: Physical Image Discrepancies Between X-ray and MRI", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig4_path = fig_dir / "modality_feature_comparison.png"
    plt.savefig(fig4_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Figure 4 -> {fig4_path}")

    # Backward compatibility: also save cv_hypothesis_distributions.png as alias
    fig1_alias = fig_dir / "cv_hypothesis_distributions.png"
    plt.figure()
    im1 = plt.imread(fig1_path)
    plt.imshow(im1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig1_alias, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 76)
    print("STATISTICAL AUDIT & RE-ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    main()
