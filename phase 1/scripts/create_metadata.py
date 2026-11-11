import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]
SPLITS = ['train', 'valid', 'test']

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.dataset import load_config
    cfg = load_config()
    PERCEPTUAL_HASH_THRESHOLD = cfg.get("grouping", {}).get("phash_threshold", 10)
except Exception:
    PERCEPTUAL_HASH_THRESHOLD = 10  # Conservative combined (pHash + dHash) dihedral Hamming distance



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


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def main():
    dataset_dir = find_dataset_dir()
    phase1_dir = Path(__file__).resolve().parent.parent
    root_dir = phase1_dir.parent
    data_dir = phase1_dir / "data"
    fig_dir = phase1_dir / "figures"
    root_data_dir = root_dir / "data_processed"

    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    root_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PART 2 & 3: AUTHORITATIVE METADATA CREATION & ANNOTATION CLEANING")
    print("=" * 72)
    print(f"Dataset directory: {dataset_dir}")

    records = []
    excluded_audit_records = []
    multiclass_investigation_records = []
    invalid_annotations = []

    for s in SPLITS:
        img_dir = dataset_dir / s / "images"
        lbl_dir = dataset_dir / s / "labels"
        img_files = sorted(os.listdir(img_dir))

        for img_name in img_files:
            img_path = img_dir / img_name
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = lbl_dir / lbl_name

            rel_img_path = f"{s}/images/{img_name}"
            rel_lbl_path = f"{s}/labels/{lbl_name}"
            base_id = img_name.split(".rf.")[0] if ".rf." in img_name else os.path.splitext(img_name)[0]
            modality = "mri" if "_mri_" in img_name.lower() or "mri" in img_name.lower() else "xray"

            boxes = []
            labels = []
            raw_cls_ids = []
            missing_lbl_file = not lbl_path.exists()
            is_empty = False

            if not missing_lbl_file:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(lines) == 0:
                    is_empty = True
                else:
                    for line_idx, line in enumerate(lines):
                        parts = line.split()
                        if len(parts) != 5:
                            invalid_annotations.append((rel_lbl_path, line_idx, f"Malformed token count: {len(parts)}"))
                            continue
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        cx, cy, bw, bh = coords
                        if not (0 <= cls_id < len(CLASS_NAMES)):
                            invalid_annotations.append((rel_lbl_path, line_idx, f"Invalid class ID: {cls_id}"))
                            continue
                        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                            invalid_annotations.append((rel_lbl_path, line_idx, f"Invalid box coords: {coords}"))
                            continue
                        raw_cls_ids.append(cls_id)
                        labels.append(CLASS_NAMES[cls_id])
                        boxes.append(coords)
            else:
                is_empty = True

            # Determine annotation quality and primary_class without arbitrary tie-breaking
            if is_empty or len(labels) == 0:
                label_quality = "unannotated"
                primary_class = "Unannotated"
                num_unique_classes = 0
                all_classes_str = "None"
                class_counts_str = "None"
                exclusion_reason = "missing_annotation_file" if missing_lbl_file else "empty_annotation_file"
                excluded_audit_records.append({
                    "filename": img_name,
                    "rel_img_path": rel_img_path,
                    "rel_lbl_path": rel_lbl_path,
                    "base_id": base_id,
                    "original_split": s,
                    "modality": modality,
                    "num_boxes": 0,
                    "num_unique_classes": 0,
                    "label_quality": label_quality,
                    "all_classes": "None",
                    "class_box_counts": "None",
                    "exclusion_reason": exclusion_reason
                })
            else:
                cls_counter = Counter(labels)
                most_common = cls_counter.most_common()
                num_unique_classes = len(most_common)
                all_classes_str = ",".join(sorted(cls_counter.keys()))
                class_counts_str = ";".join([f"{k}:{v}" for k, v in most_common])

                if num_unique_classes == 1:
                    label_quality = "single"
                    primary_class = most_common[0][0]
                    exclusion_reason = "none"
                elif most_common[0][1] > most_common[1][1]:
                    label_quality = "majority"
                    primary_class = most_common[0][0]
                    exclusion_reason = "none"
                    multiclass_investigation_records.append({
                        "filename": img_name,
                        "base_id": base_id,
                        "original_split": s,
                        "modality": modality,
                        "num_boxes": len(boxes),
                        "num_unique_classes": num_unique_classes,
                        "label_quality": label_quality,
                        "assigned_primary_class": primary_class,
                        "class_box_counts": class_counts_str,
                        "policy_decision": "Included (strict majority box count)"
                    })
                else:
                    # Exact tie between top classes: do NOT arbitrarily pick first box!
                    label_quality = "tie"
                    primary_class = "Ambiguous_Tie"
                    exclusion_reason = "ambiguous_multiclass_tie"
                    excluded_audit_records.append({
                        "filename": img_name,
                        "rel_img_path": rel_img_path,
                        "rel_lbl_path": rel_lbl_path,
                        "base_id": base_id,
                        "original_split": s,
                        "modality": modality,
                        "num_boxes": len(boxes),
                        "num_unique_classes": num_unique_classes,
                        "label_quality": label_quality,
                        "all_classes": all_classes_str,
                        "class_box_counts": class_counts_str,
                        "exclusion_reason": exclusion_reason
                    })
                    multiclass_investigation_records.append({
                        "filename": img_name,
                        "base_id": base_id,
                        "original_split": s,
                        "modality": modality,
                        "num_boxes": len(boxes),
                        "num_unique_classes": num_unique_classes,
                        "label_quality": label_quality,
                        "assigned_primary_class": "Ambiguous_Tie (Excluded)",
                        "class_box_counts": class_counts_str,
                        "policy_decision": "Excluded from single-label task (exact tie in box counts)"
                    })

            records.append({
                "filename": img_name,
                "rel_img_path": rel_img_path,
                "rel_lbl_path": rel_lbl_path,
                "base_id": base_id,
                "original_split": s,
                "modality": modality,
                "is_empty_label": is_empty,
                "num_boxes": len(boxes),
                "num_unique_classes": num_unique_classes,
                "primary_class": primary_class,
                "all_classes": all_classes_str,
                "label_quality": label_quality,
                "exclusion_reason": exclusion_reason,
                "_full_img_path": str(img_path)
            })

    assert len(invalid_annotations) == 0, f"Found invalid YOLO annotations: {invalid_annotations}"
    print(f"[Check] All {len(records)} annotation files parsed with 0 invalid class IDs or out-of-bounds boxes.")

    # Compute perceptual hashes (pHash + dHash across 4 dihedral orientations)
    N = len(records)
    print(f"[Check] Computing dihedral perceptual hashes (pHash + dHash) across {N} images...")
    p_orig, p_h, p_v, p_hv = [], [], [], []
    d_orig, d_h, d_v, d_hv = [], [], [], []

    for r in records:
        with Image.open(r["_full_img_path"]) as im:
            l = im.convert("L")
            lh = l.transpose(Image.FLIP_LEFT_RIGHT)
            lv = l.transpose(Image.FLIP_TOP_BOTTOM)
            lhv = l.transpose(Image.ROTATE_180)
            p_orig.append(imagehash.phash(l).hash.flatten())
            p_h.append(imagehash.phash(lh).hash.flatten())
            p_v.append(imagehash.phash(lv).hash.flatten())
            p_hv.append(imagehash.phash(lhv).hash.flatten())
            d_orig.append(imagehash.dhash(l).hash.flatten())
            d_h.append(imagehash.dhash(lh).hash.flatten())
            d_v.append(imagehash.dhash(lv).hash.flatten())
            d_hv.append(imagehash.dhash(lhv).hash.flatten())

    p_orig = np.array(p_orig); p_h = np.array(p_h); p_v = np.array(p_v); p_hv = np.array(p_hv)
    d_orig = np.array(d_orig); d_h = np.array(d_h); d_v = np.array(d_v); d_hv = np.array(d_hv)

    def get_dist(p1, d1, p2, d2):
        return (p1[:, None, :] != p2[None, :, :]).sum(axis=-1, dtype=np.uint8) + (d1[:, None, :] != d2[None, :, :]).sum(axis=-1, dtype=np.uint8)

    min_d4 = np.minimum(
        np.minimum(get_dist(p_orig, d_orig, p_orig, d_orig), get_dist(p_orig, d_orig, p_h, d_h)),
        np.minimum(get_dist(p_orig, d_orig, p_v, d_v), get_dist(p_orig, d_orig, p_hv, d_hv))
    )
    min_d4 = np.minimum(min_d4, min_d4.T)
    np.fill_diagonal(min_d4, 0)

    # Save distance histogram figure
    triu_dists = min_d4[np.triu_indices(N, k=1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(triu_dists, bins=64, range=(0, 64), color="#2563eb", edgecolor="black", alpha=0.75)
    axes[0].axvline(PERCEPTUAL_HASH_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Threshold = {PERCEPTUAL_HASH_THRESHOLD}")
    axes[0].set_title("All Pairwise Dihedral (pHash + dHash) Distances")
    axes[0].set_xlabel("Minimum Hamming Distance")
    axes[0].set_ylabel("Image Pair Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(triu_dists[triu_dists <= 30], bins=30, range=(0, 30), color="#d97706", edgecolor="black", alpha=0.8)
    axes[1].axvline(PERCEPTUAL_HASH_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Threshold = {PERCEPTUAL_HASH_THRESHOLD}")
    axes[1].set_title("Zoomed Near-Duplicate Region (Distance <= 30)")
    axes[1].set_xlabel("Minimum Hamming Distance")
    axes[1].set_ylabel("Image Pair Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    hist_fig_path = fig_dir / "duplicate_distance_histogram.png"
    plt.savefig(hist_fig_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Perceptual distance histogram -> {hist_fig_path}")

    # Union-Find for perceptual duplicate clusters (dup_group_id) and composite groups (group_id = base_id U dup_group_id)
    uf_dup = UnionFind(N)
    uf_comp = UnionFind(N)
    for i in range(N):
        for j in range(i + 1, N):
            if min_d4[i, j] <= PERCEPTUAL_HASH_THRESHOLD:
                uf_dup.union(i, j)
                uf_comp.union(i, j)
            if records[i]["base_id"] == records[j]["base_id"]:
                uf_comp.union(i, j)

    dup_root_map = {}
    comp_root_map = {}
    for i in range(N):
        dr = uf_dup.find(i)
        if dr not in dup_root_map:
            dup_root_map[dr] = f"dup_{len(dup_root_map):04d}"
        cr = uf_comp.find(i)
        if cr not in comp_root_map:
            comp_root_map[cr] = f"grp_{len(comp_root_map):04d}"
        records[i]["dup_group_id"] = dup_root_map[dr]
        records[i]["group_id"] = comp_root_map[cr]
        del records[i]["_full_img_path"]

    df = pd.DataFrame(records)
    # Initialize split columns (to be populated by stratified_split.py)
    df["cv_fold"] = -1
    df["stratified_group_split"] = "excluded"

    # Reorder columns to match authoritative schema
    ordered_cols = [
        "filename",
        "rel_img_path",
        "rel_lbl_path",
        "base_id",
        "original_split",
        "modality",
        "is_empty_label",
        "num_boxes",
        "num_unique_classes",
        "primary_class",
        "all_classes",
        "label_quality",
        "group_id",
        "dup_group_id",
        "cv_fold",
        "stratified_group_split",
        "exclusion_reason"
    ]
    df = df[ordered_cols]

    # Save metadata and audit logs
    meta_out = data_dir / "dataset_metadata_cleaned.csv"
    root_meta_out = root_data_dir / "dataset_metadata_cleaned.csv"
    df.to_csv(meta_out, index=False)
    df.to_csv(root_meta_out, index=False)

    excl_df = pd.DataFrame(excluded_audit_records)
    excl_out = data_dir / "excluded_records_audit_log.csv"
    excl_df.to_csv(excl_out, index=False)

    mc_df = pd.DataFrame(multiclass_investigation_records)
    mc_out = data_dir / "annotation_investigation_report.csv"
    mc_df.to_csv(mc_out, index=False)

    print("\n--- Annotation Quality Breakdown (Actual Counts) ---")
    for qual, cnt in df["label_quality"].value_counts().items():
        print(f"  {qual:15s}: {cnt:4d} images ({cnt / len(df) * 100:5.2f}%)")
    print(f"  Total records  : {len(df):4d}")
    print(f"  Included (single + majority): {(df['label_quality'].isin(['single', 'majority'])).sum():4d}")
    print(f"  Excluded (tie + unannotated): {(~df['label_quality'].isin(['single', 'majority'])).sum():4d}")
    print(f"  Unique base_ids: {df['base_id'].nunique()}")
    print(f"  Unique perceptual duplicate groups (dup_group_id): {df['dup_group_id'].nunique()}")
    print(f"  Unique composite groups (group_id): {df['group_id'].nunique()}")
    print(f"[Saved] Authoritative metadata -> {meta_out}")
    print(f"[Saved] Excluded records audit log ({len(excl_df)} records) -> {excl_out}")
    print(f"[Saved] Multi-class investigation report ({len(mc_df)} records) -> {mc_out}")


if __name__ == "__main__":
    main()
