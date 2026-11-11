import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]
SPLITS = ['train', 'valid', 'test']


def find_dataset_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    phase1_dir = script_dir.parent
    root_dir = phase1_dir.parent
    candidates = [
        root_dir / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Bone Fractures Detection",
        phase1_dir / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)" / "Bone Fractures Detection",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not locate HBFMID dataset directory.")


def main():
    dataset_dir = find_dataset_dir()
    phase1_dir = Path(__file__).resolve().parent.parent
    data_dir = phase1_dir / "data"
    meta_path = data_dir / "dataset_metadata_cleaned.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Authoritative metadata not found at {meta_path}. Run create_metadata.py and stratified_split.py first.")

    df = pd.read_csv(meta_path)

    md5_map = defaultdict(list)
    corrupt_images = []
    image_sizes = Counter()
    image_modes = Counter()
    invalid_coords = []
    bbox_class_counts = {s: Counter() for s in SPLITS}
    box_widths, box_heights, box_areas = [], [], []

    # Accumulators for exact TRAIN-ONLY pixel mean and true dataset variance
    # Using exact sum and sum of squares across all pixels in train split
    train_pixel_sum = np.zeros(3, dtype=np.float64)
    train_pixel_sq_sum = np.zeros(3, dtype=np.float64)
    train_pixel_count = 0

    train_xray_sum = np.zeros(3, dtype=np.float64)
    train_xray_sq_sum = np.zeros(3, dtype=np.float64)
    train_xray_count = 0

    train_mri_sum = np.zeros(3, dtype=np.float64)
    train_mri_sq_sum = np.zeros(3, dtype=np.float64)
    train_mri_count = 0

    for _, row in df.iterrows():
        s = row["original_split"]
        img_path = dataset_dir / row["rel_img_path"]
        lbl_path = dataset_dir / row["rel_lbl_path"]

        try:
            with Image.open(img_path) as im:
                image_sizes[im.size] += 1
                image_modes[im.mode] += 1
                h = hashlib.md5(im.tobytes()).hexdigest()
                md5_map[h].append((s, row["filename"]))

                if row["stratified_group_split"] == "train":
                    arr = np.asarray(im, dtype=np.float64) / 255.0
                    n_pix = arr.shape[0] * arr.shape[1]
                    ch_sum = arr.sum(axis=(0, 1))
                    ch_sq_sum = np.square(arr).sum(axis=(0, 1))

                    train_pixel_sum += ch_sum
                    train_pixel_sq_sum += ch_sq_sum
                    train_pixel_count += n_pix

                    if row["modality"] == "xray":
                        train_xray_sum += ch_sum
                        train_xray_sq_sum += ch_sq_sum
                        train_xray_count += n_pix
                    else:
                        train_mri_sum += ch_sum
                        train_mri_sq_sum += ch_sq_sum
                        train_mri_count += n_pix
        except Exception as e:
            corrupt_images.append((str(img_path), str(e)))

        if lbl_path.exists() and not row["is_empty_label"]:
            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]
            for idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    invalid_coords.append((str(lbl_path), idx, f"Expected 5 tokens, got {len(parts)}"))
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = [float(x) for x in parts[1:]]
                if not (0 <= cls_id < len(CLASS_NAMES)):
                    invalid_coords.append((str(lbl_path), idx, f"Invalid class ID {cls_id}"))
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    invalid_coords.append((str(lbl_path), idx, f"Out of bounds {[cx, cy, bw, bh]}"))
                bbox_class_counts[s][CLASS_NAMES[cls_id]] += 1
                box_widths.append(bw)
                box_heights.append(bh)
                box_areas.append(bw * bh)

    # Calculate true train-only mean and standard deviation: Var(X) = E[X^2] - (E[X])^2
    train_mean = train_pixel_sum / train_pixel_count
    train_std = np.sqrt(np.maximum(0.0, (train_pixel_sq_sum / train_pixel_count) - np.square(train_mean)))

    xray_mean = train_xray_sum / max(1, train_xray_count)
    xray_std = np.sqrt(np.maximum(0.0, (train_xray_sq_sum / max(1, train_xray_count)) - np.square(xray_mean)))

    mri_mean = train_mri_sum / max(1, train_mri_count)
    mri_std = np.sqrt(np.maximum(0.0, (train_mri_sq_sum / max(1, train_mri_count)) - np.square(mri_mean)))

    pixel_stats = {
        "train_overall": {
            "mean": [round(float(x), 4) for x in train_mean],
            "std": [round(float(x), 4) for x in train_std],
            "num_images": int((df["stratified_group_split"] == "train").sum())
        },
        "train_xray": {
            "mean": [round(float(x), 4) for x in xray_mean],
            "std": [round(float(x), 4) for x in xray_std],
            "num_images": int(((df["stratified_group_split"] == "train") & (df["modality"] == "xray")).sum())
        },
        "train_mri": {
            "mean": [round(float(x), 4) for x in mri_mean],
            "std": [round(float(x), 4) for x in mri_std],
            "num_images": int(((df["stratified_group_split"] == "train") & (df["modality"] == "mri")).sum())
        }
    }
    with open(data_dir / "train_pixel_stats.json", "w", encoding="utf-8") as f:
        json.dump(pixel_stats, f, indent=2)

    print("=" * 72)
    print("1. FILE INTEGRITY & CORRUPTION CHECKS")
    print("=" * 72)
    print(f"Total audited images      : {len(df)}")
    print(f"Corrupt/unreadable images : {len(corrupt_images)}")
    print(f"Image resolutions         : {dict(image_sizes)}")
    print(f"Image color modes         : {dict(image_modes)}")
    print(f"Invalid YOLO annotations  : {len(invalid_coords)}")

    print("\n" + "=" * 72)
    print("2. EXACT-BYTE & PERCEPTUAL DUPLICATE ANALYSIS")
    print("=" * 72)
    exact_dupes = {h: locs for h, locs in md5_map.items() if len(locs) > 1}
    print(f"Exact MD5 byte duplicates : {len(exact_dupes)}")
    print(f"Unique filename base_ids  : {df['base_id'].nunique()}")
    print(f"Perceptual dup_group_ids  : {df['dup_group_id'].nunique()}")
    print(f"Composite group_ids       : {df['group_id'].nunique()}")

    # Check how many base_ids span multiple perceptual clusters
    base_to_dup = df.groupby("base_id")["dup_group_id"].nunique()
    dup_to_base = df.groupby("dup_group_id")["base_id"].nunique()
    print(f"base_ids containing >1 distinct perceptual cluster : {(base_to_dup > 1).sum()} / {len(base_to_dup)}")
    print(f"Perceptual clusters merging >1 base_id             : {(dup_to_base > 1).sum()} / {len(dup_to_base)}")

    # Leakage in original split
    orig_base_leaks = (df.groupby("base_id")["original_split"].nunique() > 1).sum()
    orig_dup_leaks = (df.groupby("dup_group_id")["original_split"].nunique() > 1).sum()
    print(f"Original Roboflow split leaked base_ids            : {orig_base_leaks} / {df['base_id'].nunique()} ({orig_base_leaks / df['base_id'].nunique() * 100:.2f}%)")
    print(f"Original Roboflow split leaked dup_group_ids       : {orig_dup_leaks} / {df['dup_group_id'].nunique()} ({orig_dup_leaks / df['dup_group_id'].nunique() * 100:.2f}%)")

    print("\n" + "=" * 72)
    print("3. ANNOTATION QUALITY & MODALITY CONFOUND CROSSTAB")
    print("=" * 72)
    print("Label Quality Distribution:")
    print(df["label_quality"].value_counts().to_string())
    print("\nPrimary Class x Modality Crosstab (All 1,539 records):")
    print(pd.crosstab(df["primary_class"], df["modality"], margins=True).to_string())

    print("\n" + "=" * 72)
    print("4. CORRECTED SPLIT COVERAGE TABLE (Included Records Only)")
    print("=" * 72)
    inc_df = df[df["stratified_group_split"].isin(SPLITS)]
    print(pd.crosstab(inc_df["primary_class"], inc_df["stratified_group_split"], margins=True).to_string())

    print("\n" + "=" * 72)
    print("5. TRAIN-ONLY PIXEL NORMALIZATION & BOUNDING BOX STATISTICS")
    print("=" * 72)
    print(f"Train-Only Overall Mean (R,G,B) : {pixel_stats['train_overall']['mean']}")
    print(f"Train-Only Overall Std  (R,G,B) : {pixel_stats['train_overall']['std']} (True dataset std, N={pixel_stats['train_overall']['num_images']} train images)")
    print(f"Train-Only X-ray Mean / Std     : {pixel_stats['train_xray']['mean']} / {pixel_stats['train_xray']['std']} (N={pixel_stats['train_xray']['num_images']})")
    print(f"Train-Only MRI Mean / Std       : {pixel_stats['train_mri']['mean']} / {pixel_stats['train_mri']['std']} (N={pixel_stats['train_mri']['num_images']})")

    box_widths = np.array(box_widths)
    box_heights = np.array(box_heights)
    box_areas = np.array(box_areas)
    print(f"BBox Width  - Mean: {box_widths.mean():.4f}, Std: {box_widths.std():.4f}, Min: {box_widths.min():.4f}, Max: {box_widths.max():.4f}")
    print(f"BBox Height - Mean: {box_heights.mean():.4f}, Std: {box_heights.std():.4f}, Min: {box_heights.min():.4f}, Max: {box_heights.max():.4f}")
    print(f"BBox Area   - Median: {np.median(box_areas):.4f}, IQR: {np.percentile(box_areas, 75) - np.percentile(box_areas, 25):.4f}, Mean: {box_areas.mean():.4f}")


if __name__ == "__main__":
    main()
