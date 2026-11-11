import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import hashlib
import json
import warnings
from datetime import datetime, timezone
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import InconsistentVersionWarning
import yaml


FEATURE_NAMES_64 = (
    [f"grid4x4_r{r}_c{c}_{lvl}" for r in range(4) for c in range(4) for lvl in ("low", "high")]  # [0..31]
    + [
        "hough_log1p_line_count",      # [32]
        "hough_mean_len_norm",         # [33]
        "hough_std_len_norm",          # [34]
        "hough_max_len_norm",          # [35]
        "hough_circ_mean_angle",       # [36]
        "hough_circ_dispersion",       # [37]
        "hough_horiz_ratio",           # [38]
        "hough_vert_ratio",            # [39]
        "hough_oblique_ratio",         # [40]
        "sobel_mag_mean",              # [41]
        "sobel_mag_std",               # [42]
        "sobel_mag_max",               # [43]
    ]
    + [f"sobel_ori_hist_bin{b}" for b in range(8)]  # [44..51]
    + [
        "global_edge_density_low",     # [52]
        "global_edge_density_high",    # [53]
        "sobel_orientation_entropy",   # [54]
        "hough_log1p_total_len_norm",  # [55]
        "quad_tl_edge_density_high",   # [56]
        "quad_tr_edge_density_high",   # [57]
        "quad_bl_edge_density_high",   # [58]
        "quad_br_edge_density_high",   # [59]
        "sobel_mag_p90",               # [60]
        "sobel_mag_p95",               # [61]
        "sobel_mag_p99",               # [62]
        "high_to_low_edge_ratio",      # [63]
    ]
)
assert len(FEATURE_NAMES_64) == 64, f"Expected 64 feature names, got {len(FEATURE_NAMES_64)}"


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Loads YAML configuration from provided path or standard discovery paths."""
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    candidates = [
        Path.cwd() / "configs" / "phase1_config.yaml",
        Path.cwd() / "phase 1" / "configs" / "phase1_config.yaml",
        Path(__file__).resolve().parent.parent / "configs" / "phase1_config.yaml",
        Path(__file__).resolve().parent.parent.parent / "configs" / "phase1_config.yaml",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def extract_classical_edge_features_np(
    img_gray: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
    gaussian_blur_ksize: Tuple[int, int] = (3, 3),
    canny_low_thresh: Tuple[int, int] = (30, 100),
    canny_high_thresh: Tuple[int, int] = (75, 200),
    hough_rho: int = 1,
    hough_theta_deg: float = 1.0,
    hough_threshold: int = 40,
    hough_min_line_length: int = 25,
    hough_max_line_gap: int = 10,
) -> np.ndarray:
    """
    Extracts a numerically stable, domain-informed 64-dimensional classical CV feature vector
    from a single-channel 8-bit grayscale radiograph. Fully configurable via parameters.
    """
    if img_gray is None or img_gray.size == 0:
        return np.zeros(64, dtype=np.float32)

    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    img_gray = np.asarray(img_gray, dtype=np.uint8)

    h, w = img_gray.shape
    diag = float(np.hypot(h, w)) + 1e-8

    # Check for completely flat / zero-gradient image before CLAHE
    if int(img_gray.max()) == int(img_gray.min()):
        return np.zeros(64, dtype=np.float32)

    # 1. Medical contrast enhancement (CLAHE) + Gaussian smoothing
    tile_grid = tuple(clahe_tile_grid) if isinstance(clahe_tile_grid, (list, tuple)) else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=tile_grid)
    enhanced = clahe.apply(img_gray)
    blur_k = tuple(gaussian_blur_ksize) if isinstance(gaussian_blur_ksize, (list, tuple)) else (3, 3)
    smoothed = cv2.GaussianBlur(enhanced, blur_k, 0)

    # 2. Multi-threshold Canny Edge Detection
    c_low0, c_low1 = int(canny_low_thresh[0]), int(canny_low_thresh[1])
    c_high0, c_high1 = int(canny_high_thresh[0]), int(canny_high_thresh[1])
    edges_low = cv2.Canny(smoothed, c_low0, c_low1)
    edges_high = cv2.Canny(smoothed, c_high0, c_high1)

    # 3. Spatial 4x4 Grid Edge Densities [0..31]
    grid_h, grid_w = max(1, h // 4), max(1, w // 4)
    spatial_densities = []
    for r in range(4):
        r0, r1 = r * grid_h, (r + 1) * grid_h if r < 3 else h
        for c in range(4):
            c0, c1 = c * grid_w, (c + 1) * grid_w if c < 3 else w
            patch_low = edges_low[r0:r1, c0:c1]
            patch_high = edges_high[r0:r1, c0:c1]
            spatial_densities.append(float(np.mean(patch_low > 0)))
            spatial_densities.append(float(np.mean(patch_high > 0)))

    # 4. Probabilistic Hough Lines & Circular Axial Statistics [32..40]
    h_theta_rad = float(np.radians(hough_theta_deg))
    lines = cv2.HoughLinesP(
        edges_high,
        rho=int(hough_rho),
        theta=h_theta_rad,
        threshold=int(hough_threshold),
        minLineLength=int(hough_min_line_length),
        maxLineGap=int(hough_max_line_gap)
    )

    lengths = []
    angles_deg = []
    if lines is not None and len(lines) > 0:
        for line in lines:
            vals = line.ravel()
            if len(vals) >= 4:
                x1, y1, x2, y2 = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length > 0.0:
                    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
                    lengths.append(length)
                    angles_deg.append(angle)

    line_count = float(len(lengths))
    log_line_count = float(np.log1p(line_count) / np.log1p(1000.0))

    if len(lengths) > 0:
        len_arr = np.asarray(lengths, dtype=np.float64)
        ang_arr = np.asarray(angles_deg, dtype=np.float64)
        mean_len = float(np.mean(len_arr)) / diag
        std_len = float(np.std(len_arr)) / diag
        max_len = float(np.max(len_arr)) / diag
        log_total_len = float(np.log1p(np.sum(len_arr) / diag))

        # Axial circular statistics (period = 180 deg -> double angle 2*theta)
        two_theta = np.radians(2.0 * ang_arr)
        c_bar = float(np.mean(np.cos(two_theta)))
        s_bar = float(np.mean(np.sin(two_theta)))
        r_len = float(np.hypot(c_bar, s_bar))
        circ_mean_ang = float(((0.5 * np.arctan2(s_bar, c_bar)) % np.pi) / np.pi)
        circ_dispersion = float(np.clip(1.0 - r_len, 0.0, 1.0))

        horiz_ratio = float(np.mean((ang_arr < 30.0) | (ang_arr >= 150.0)))
        vert_ratio = float(np.mean((ang_arr >= 60.0) & (ang_arr < 120.0)))
        oblique_ratio = float(np.clip(1.0 - horiz_ratio - vert_ratio, 0.0, 1.0))
    else:
        mean_len = std_len = max_len = log_total_len = 0.0
        circ_mean_ang = circ_dispersion = 0.0
        horiz_ratio = vert_ratio = oblique_ratio = 0.0

    # 5. Sobel Gradient Magnitude & Orientation Statistics [41..51]
    sobel_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)
    grad_angle = (np.degrees(np.arctan2(sobel_y, sobel_x)) + 360.0) % 360.0

    mag_mean = float(np.mean(grad_mag)) / 255.0
    mag_std = float(np.std(grad_mag)) / 255.0
    mag_max = float(np.max(grad_mag)) / 255.0

    # Upper percentiles of gradient magnitude [60..62]
    p90 = float(np.percentile(grad_mag, 90)) / 255.0
    p95 = float(np.percentile(grad_mag, 95)) / 255.0
    p99 = float(np.percentile(grad_mag, 99)) / 255.0

    # 8-bin magnitude-weighted orientation histogram [44..51]
    bins = np.linspace(0.0, 360.0, 9)
    hist, _ = np.histogram(grad_angle, bins=bins, weights=grad_mag)
    hist_sum = float(np.sum(hist))
    if hist_sum > 1e-8:
        hist_norm = hist / hist_sum
        p_safe = np.where(hist_norm > 1e-12, hist_norm, 1.0)
        entropy = float(-np.sum(hist_norm * np.log2(p_safe)) / 3.0)  # Normalized by log2(8)=3
    else:
        hist_norm = np.full(8, 0.125, dtype=np.float64)
        entropy = 1.0

    # 6. Global Edge Densities & Quadrant Descriptors [52..59]
    global_edge_low = float(np.mean(edges_low > 0))
    global_edge_high = float(np.mean(edges_high > 0))

    mid_h, mid_w = h // 2, w // 2
    quad_tl = float(np.mean(edges_high[:mid_h, :mid_w] > 0))
    quad_tr = float(np.mean(edges_high[:mid_h, mid_w:] > 0))
    quad_bl = float(np.mean(edges_high[mid_h:, :mid_w] > 0))
    quad_br = float(np.mean(edges_high[mid_h:, mid_w:] > 0))

    # High-to-low edge ratio [63]
    edge_ratio = float((np.sum(edges_high > 0)) / (np.sum(edges_low > 0) + 1.0))

    features = (
        spatial_densities  # 32
        + [
            log_line_count,
            mean_len,
            std_len,
            max_len,
            circ_mean_ang,
            circ_dispersion,
            horiz_ratio,
            vert_ratio,
            oblique_ratio,
            mag_mean,
            mag_std,
            mag_max,
        ]  # 12 -> 44
        + hist_norm.tolist()  # 8 -> 52
        + [
            global_edge_low,
            global_edge_high,
            entropy,
            log_total_len,
            quad_tl,
            quad_tr,
            quad_bl,
            quad_br,
            p90,
            p95,
            p99,
            edge_ratio,
        ]  # 12 -> 64
    )

    out = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def apply_shared_geometric_transform(
    pil_img: Image.Image,
    is_train: bool = True,
    do_flip: Optional[bool] = None,
    angle: Optional[float] = None
) -> Tuple[Image.Image, dict]:
    """
    Applies an identical, synchronized geometric transformation to a PIL image.
    Both deep and classical branches receive the exact same transformed image.
    """
    params = {"flipped": False, "rotation_angle": 0.0}
    if not is_train and do_flip is None and angle is None:
        return pil_img, params

    if do_flip is None:
        do_flip = bool(torch.rand(1).item() > 0.5)
    if angle is None:
        angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())

    transformed = pil_img
    if do_flip:
        transformed = TF.hflip(transformed)
        params["flipped"] = True

    if abs(angle) > 1e-3:
        transformed = TF.rotate(
            transformed,
            angle=angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )
        params["rotation_angle"] = angle

    return transformed, params


def compute_dataset_hash(train_df: pd.DataFrame) -> str:
    """Computes a SHA-256 hash of sorted training filenames to verify split provenance."""
    sorted_filenames = sorted(train_df["filename"].astype(str).tolist())
    content = "\n".join(sorted_filenames).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def validate_scaler_provenance(
    scaler_path: str,
    train_df: pd.DataFrame
) -> Tuple[bool, str, Optional[StandardScaler]]:
    """
    Validates whether an existing StandardScaler on disk matches the current training split:
    - Verifies file exists.
    - Suppresses scikit-learn InconsistentVersionWarning while loading.
    - Asserts n_features_in_ == 64.
    - Asserts n_samples_seen_ matches len(train_df).
    - Checks provenance JSON hash if available.
    """
    if not os.path.exists(scaler_path):
        return False, f"Scaler file not found at {scaler_path}", None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            scaler = joblib.load(scaler_path)
    except Exception as e:
        return False, f"Failed to unpickle scaler: {e}", None

    if not hasattr(scaler, "mean_") or len(scaler.mean_) != 64:
        return False, f"Scaler feature dimension mismatch: expected 64, got {getattr(scaler, 'mean_', None)}", None

    expected_n = len(train_df)
    actual_n = int(getattr(scaler, "n_samples_seen_", 0))
    if actual_n != expected_n:
        return False, f"Scaler sample count mismatch: expected {expected_n} train samples, scaler has {actual_n}", None

    prov_path = Path(scaler_path).parent / f"{Path(scaler_path).stem}_provenance.json"
    if prov_path.exists():
        try:
            with open(prov_path, "r", encoding="utf-8") as f:
                prov = json.load(f)
            expected_hash = compute_dataset_hash(train_df)
            if prov.get("train_filenames_hash") != expected_hash:
                return False, f"Training split filenames hash mismatch in {prov_path.name}", None
        except Exception:
            pass

    return True, "Scaler verified and compatible with current training split.", scaler


def save_scaler_with_provenance(
    scaler: StandardScaler,
    scaler_path: str,
    train_df: pd.DataFrame
) -> str:
    """Saves fitted StandardScaler along with an immutable provenance JSON manifest."""
    os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    prov_path = Path(scaler_path).parent / f"{Path(scaler_path).stem}_provenance.json"
    prov_data = {
        "scaler_file": Path(scaler_path).name,
        "n_samples_fitted": int(scaler.n_samples_seen_),
        "n_features": int(scaler.n_features_in_) if hasattr(scaler, "n_features_in_") else 64,
        "sklearn_version": sklearn.__version__,
        "train_filenames_hash": compute_dataset_hash(train_df),
        "fit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mean_norm": float(np.linalg.norm(scaler.mean_)),
        "scale_norm": float(np.linalg.norm(scaler.scale_))
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov_data, f, indent=2)

    return str(prov_path)


def fit_or_load_edge_scaler(
    metadata_df: pd.DataFrame,
    base_data_dir: str,
    split_col: str = "stratified_group_split",
    scaler_path: Optional[str] = None,
    force_refit: bool = False,
    edge_params: Optional[dict] = None
) -> StandardScaler:
    """
    Fits a StandardScaler strictly on the unaugmented TRAINING split images, or loads an
    already-fitted scaler from disk with strict provenance checking. Never fits on validation or test data.
    """
    train_df = metadata_df[
        (metadata_df[split_col] == "train")
        & (~metadata_df["is_empty_label"])
        & (metadata_df["label_quality"].isin(["single", "majority"]))
    ].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(f"No valid training rows found for column '{split_col}' to fit StandardScaler.")

    if scaler_path is not None and not force_refit:
        is_valid, msg, scaler = validate_scaler_provenance(scaler_path, train_df)
        if is_valid and scaler is not None:
            return scaler
        else:
            print(f"[Provenance Check] Refitting scaler: {msg}")

    base_path = Path(base_data_dir)
    train_feats = []
    kw = edge_params or {}
    for _, row in train_df.iterrows():
        rel_p = str(row["rel_img_path"]).replace("\\", "/")
        full_p = base_path / rel_p
        with Image.open(full_p) as im:
            gray_np = np.asarray(im.convert("L"), dtype=np.uint8)
        vec = extract_classical_edge_features_np(gray_np, **kw)
        train_feats.append(vec)

    X_train = np.vstack(train_feats)
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.scale_ = np.where(scaler.scale_ < 1e-8, 1.0, scaler.scale_)

    if scaler_path is not None:
        save_scaler_with_provenance(scaler, scaler_path, train_df)

    return scaler


class BoneFractureDataset(Dataset):
    """
    Multi-Modal Dataset for FractureFuseNet:
      - Applies shared geometric augmentation first so both branches see identical geometry.
      - Branch A (Deep Vision): Resize to 224x224 -> ToTensor -> Train-calibrated normalization -> (3, 224, 224)
      - Branch B (Classical CV): CLAHE -> Canny -> Hough -> Sobel -> 64-dim vector -> Train-fitted StandardScaler -> (64,)
      - Target Label: Integer class index in [0..9]
    """
    CLASS_NAMES = [
        'Comminuted', 'Greenstick', 'Healthy', 'Linear',
        'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
        'Transverse Displaced', 'Transverse'
    ]
    CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Calibrated strictly on the 857 training-split radiographs (see train_pixel_stats.json)
    TRAIN_PIXEL_MEAN = [0.2555, 0.2555, 0.2555]
    TRAIN_PIXEL_STD = [0.2594, 0.2594, 0.2596]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        base_data_dir: str,
        split: str = 'train',
        split_col: str = 'stratified_group_split',
        is_train: bool = True,
        img_size: int = 224,
        edge_scaler: Optional[StandardScaler] = None,
        norm_mode: str = 'train_dataset',
        precomputed_unaugmented_features: Optional[Dict[str, np.ndarray]] = None,
        edge_params: Optional[dict] = None
    ):
        self.base_data_dir = Path(base_data_dir)
        self.split = split
        self.split_col = split_col
        self.is_train = is_train
        self.img_size = img_size
        self.edge_scaler = edge_scaler
        self.precomputed_unaugmented = precomputed_unaugmented_features
        self.edge_params = edge_params or {}

        # Filter to valid included records in the requested split
        mask = (
            (metadata_df[split_col] == split)
            & (~metadata_df['is_empty_label'])
        )
        if 'label_quality' in metadata_df.columns:
            mask = mask & (metadata_df['label_quality'].isin(['single', 'majority']))

        self.df = metadata_df[mask].reset_index(drop=True)

        if norm_mode == 'imagenet':
            mean, std = self.IMAGENET_MEAN, self.IMAGENET_STD
        else:
            mean, std = self.TRAIN_PIXEL_MEAN, self.TRAIN_PIXEL_STD

        self.deep_post_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self) -> int:
        return len(self.df)

    def extract_classical_edge_features(self, img_gray: np.ndarray) -> torch.Tensor:
        raw_vec = extract_classical_edge_features_np(img_gray, **self.edge_params)
        if self.edge_scaler is not None:
            scaled_vec = self.edge_scaler.transform(raw_vec.reshape(1, -1)).reshape(-1)
            scaled_vec = np.nan_to_num(scaled_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return torch.from_numpy(scaled_vec)
        return torch.from_numpy(raw_vec)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        rel_img_path = str(row['rel_img_path']).replace('\\', '/')
        full_img_path = self.base_data_dir / rel_img_path

        if not full_img_path.exists():
            raise FileNotFoundError(f"Image file not found: {full_img_path}")

        with Image.open(full_img_path) as raw_pil:
            pil_rgb = raw_pil.convert('RGB')

        # 1. Shared Geometric Augmentation
        geom_pil_rgb, geom_params = apply_shared_geometric_transform(pil_rgb, is_train=self.is_train)

        # 2. Branch A: Deep Radiographic Tensor (3, img_size, img_size)
        deep_tensor = self.deep_post_transforms(geom_pil_rgb)

        # 3. Branch B: Classical 64-dim Edge & Line Descriptor
        if (not self.is_train) and (self.precomputed_unaugmented is not None) and (row['filename'] in self.precomputed_unaugmented):
            raw_vec = self.precomputed_unaugmented[row['filename']]
            if self.edge_scaler is not None:
                scaled_vec = self.edge_scaler.transform(raw_vec.reshape(1, -1)).reshape(-1).astype(np.float32)
                edge_tensor = torch.from_numpy(np.nan_to_num(scaled_vec, nan=0.0, posinf=0.0, neginf=0.0))
            else:
                edge_tensor = torch.from_numpy(raw_vec.astype(np.float32))
        else:
            geom_gray_np = np.asarray(geom_pil_rgb.convert('L'), dtype=np.uint8)
            edge_tensor = self.extract_classical_edge_features(geom_gray_np)

        cls_name = row['primary_class']
        if cls_name not in self.CLASS_TO_IDX:
            raise ValueError(f"Invalid class label '{cls_name}' in file {row['filename']}")
        label_idx = self.CLASS_TO_IDX[cls_name]

        return {
            'image': deep_tensor,
            'edge_features': edge_tensor,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'filename': str(row['filename']),
            'base_id': str(row['base_id']),
            'group_id': str(row['group_id']) if 'group_id' in row else str(row['base_id']),
            'modality': str(row['modality']) if 'modality' in row else 'xray',
            'geom_params': geom_params
        }


def compute_train_class_weights(metadata_df: pd.DataFrame, split_col: str = 'stratified_group_split') -> torch.Tensor:
    """Computes inverse-frequency class weights strictly from the training split."""
    train_df = metadata_df[
        (metadata_df[split_col] == 'train')
        & (~metadata_df['is_empty_label'])
        & (metadata_df['label_quality'].isin(['single', 'majority']))
    ]
    counts = np.zeros(len(BoneFractureDataset.CLASS_NAMES), dtype=np.float64)
    for cls_name, cnt in train_df['primary_class'].value_counts().items():
        if cls_name in BoneFractureDataset.CLASS_TO_IDX:
            counts[BoneFractureDataset.CLASS_TO_IDX[cls_name]] = float(cnt)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional train-derived class weights for severe class imbalance."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_dataloaders(
    metadata_path: str,
    base_data_dir: str,
    split_col: str = 'stratified_group_split',
    batch_size: int = 16,
    num_workers: int = 0,
    norm_mode: str = 'train_dataset',
    scaler_path: Optional[str] = None,
    use_weighted_sampler: bool = False,
    edge_params: Optional[dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    df = pd.read_csv(metadata_path)

    if scaler_path is None:
        meta_dir = Path(metadata_path).resolve().parent
        scaler_path = str(meta_dir / "edge_feature_scaler.joblib")

    # Fit or load StandardScaler strictly using training split data
    edge_scaler = fit_or_load_edge_scaler(
        metadata_df=df,
        base_data_dir=base_data_dir,
        split_col=split_col,
        scaler_path=scaler_path,
        force_refit=False,
        edge_params=edge_params
    )

    # Load precomputed unaugmented feature cache if available
    cache_path = Path(scaler_path).resolve().parent / "unaugmented_edge_features.npz"
    precomputed_map = None
    if cache_path.exists():
        try:
            npz = np.load(cache_path, allow_pickle=False)
            filenames = npz["filenames"]
            features = npz["features"]
            precomputed_map = {str(fn): features[i] for i, fn in enumerate(filenames)}
        except Exception:
            precomputed_map = None

    train_ds = BoneFractureDataset(
        df, base_data_dir, split='train', split_col=split_col,
        is_train=True, edge_scaler=edge_scaler, norm_mode=norm_mode,
        precomputed_unaugmented_features=precomputed_map,
        edge_params=edge_params
    )
    val_ds = BoneFractureDataset(
        df, base_data_dir, split='valid', split_col=split_col,
        is_train=False, edge_scaler=edge_scaler, norm_mode=norm_mode,
        precomputed_unaugmented_features=precomputed_map,
        edge_params=edge_params
    )
    test_ds = BoneFractureDataset(
        df, base_data_dir, split='test', split_col=split_col,
        is_train=False, edge_scaler=edge_scaler, norm_mode=norm_mode,
        precomputed_unaugmented_features=precomputed_map,
        edge_params=edge_params
    )

    pin_mem = torch.cuda.is_available()

    if use_weighted_sampler and len(train_ds) > 0:
        cls_counts = train_ds.df['primary_class'].value_counts().to_dict()
        sample_weights = [1.0 / max(1, cls_counts[c]) for c in train_ds.df['primary_class']]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_mem)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader, train_ds.CLASS_NAMES
