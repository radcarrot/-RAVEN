"""
preprocess.py
-------------
Data ingestion and preprocessing pipeline for the SMD (Server Machine Dataset).

Pipeline:
  1. Load raw multivariate CSV data.
  2. Apply a sliding window to produce fixed-length sequence chunks.
  3. Apply instance-wise (per-window) Z-score normalization.

Why these design choices are defensible:
  - Sliding windows: transform a non-i.i.d. stream into i.i.d. samples
    the VAE can train on without recurrent state.
  - Instance-wise Z-score: IT metrics are non-stationary (CPU spikes, memory
    pressure). Global normalization would drown local anomaly signals. Per-window
    normalization makes the VAE learn *shape* (temporal pattern), not absolute scale.

Author: VAE-SMD Mini-Project
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Data Ingestion
# ---------------------------------------------------------------------------

def load_csv(
    filepath: str,
    label_col: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a multivariate time-series CSV file into NumPy arrays.

    Args:
        filepath  : Path to the CSV file (smd_train.csv or smd_test.csv).
        label_col : (Optional) Name of the binary anomaly label column.
                    Expected in the test set only (0 = normal, 1 = anomaly).
                    If None or column is absent, labels are not returned.

    Returns:
        data   : np.ndarray, shape (T, F) — T timesteps, F sensor features.
        labels : np.ndarray, shape (T,)   — anomaly labels, or None.
    """
    df = pd.read_csv(filepath)

    if label_col and label_col in df.columns:
        labels = df[label_col].values.astype(np.float32)
        data   = df.drop(columns=[label_col]).values.astype(np.float32)
    else:
        labels = None
        data   = df.values.astype(np.float32)

    print(f"[Preprocess] Loaded '{filepath}' → shape={data.shape}, "
          f"features={data.shape[1]}, labels={'yes' if labels is not None else 'no'}")
    return data, labels


# ---------------------------------------------------------------------------
# 2. Sliding Window
# ---------------------------------------------------------------------------

def sliding_window(
    data: np.ndarray,
    window_size: int = 64,
    stride: int = 1
) -> np.ndarray:
    """
    Convert a continuous time-series (T, F) into overlapping fixed-length windows.

    Each window is treated as one independent sample by the VAE. This is the
    standard approach for applying batch-mode deep learning to streaming data.

    Output shape is (N, F, W) — channels-first — to match PyTorch Conv1d
    which expects input of shape (batch, channels, length).

    Args:
        data        : np.ndarray, shape (T, F).
        window_size : Number of timesteps per window, W  (default=64).
        stride      : Step between consecutive window starts (default=1).
                      stride=1 → maximum data usage (dense overlap).
                      stride=window_size → no overlap (faster, less data).

    Returns:
        windows : np.ndarray, shape (N, F, W).
                  N = floor((T - window_size) / stride) + 1.
    """
    T, F = data.shape
    indices = range(0, T - window_size + 1, stride)

    # Pre-allocate the full output array for memory efficiency
    windows = np.empty((len(indices), F, window_size), dtype=np.float32)

    for i, start in enumerate(indices):
        # Slice (window_size, F) then transpose to (F, window_size) = channels-first
        windows[i] = data[start : start + window_size].T

    print(f"[Preprocess] Sliding window: T={T}, W={window_size}, stride={stride} "
          f"→ {len(indices)} windows, output shape={windows.shape}")
    return windows


# ---------------------------------------------------------------------------
# 3. Instance-wise Z-score Normalization
# ---------------------------------------------------------------------------

def normalize_windows(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply instance-wise (per-window, per-feature) Z-score normalization.

    Formula applied independently per window w and feature channel f:
        x_norm[w, f, :] = (x[w, f, :] - mean_f) / (std_f + eps)

    WHY instance-wise (not global):
        - IT metrics are non-stationary: a CPU metric's "normal" range at 2 AM
          differs from its "normal" at peak hours. Global stats would distort this.
        - Anomalies are relative deviations within a window, not absolute values.
        - The VAE learns to reconstruct the normalized *shape* of each window.
          If it fails to reconstruct, the window is anomalous regardless of scale.

    Args:
        windows : np.ndarray, shape (N, F, W).
        eps     : Small constant to prevent division by zero for flat signals.

    Returns:
        Normalized windows, same shape (N, F, W), dtype float32.
    """
    # mean/std shape: (N, F, 1) — keepdims=True enables broadcasting over W
    mean = windows.mean(axis=2, keepdims=True)
    std  = windows.std(axis=2,  keepdims=True)

    normalized = (windows - mean) / (std + eps)

    print(f"[Preprocess] Instance-wise Z-score normalization applied. "
          f"Output shape: {normalized.shape}")
    return normalized


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset
# ---------------------------------------------------------------------------

class SMDWindowDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset with lazy (on-demand) windowing.

    Instead of pre-allocating an (N, F, W) array for all windows — which causes
    OOM for 500k+ timestep datasets at stride=1 — this class stores only the
    raw (T, F) time-series and extracts + normalises each window in __getitem__.

    Memory footprint: O(T × F) instead of O(N × F × W).
    For T=500k, F=38, W=64, stride=1 this is ~73 MB vs ~4.9 GB.

    Each __getitem__ returns one normalised window tensor of shape (F, W),
    ready to be stacked into a batch by DataLoader for Conv1d consumption.

    Normalization modes:
      - Global (recommended): pass train_mean/train_std from the full training set.
        Preserves absolute level information — level-shift anomalies remain visible.
      - Per-window Z-score (legacy): omit train_mean/train_std. Each window is
        independently Z-scored per feature. Erases level-shift anomalies.
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        eps: float = 1e-8,
        train_mean: Optional[np.ndarray] = None,
        train_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            data        : np.ndarray, shape (T, F) — raw data.
            window_size : Number of timesteps per window W.
            stride      : Step between consecutive window starts.
            eps         : Small constant for numerically stable division.
            train_mean  : np.ndarray, shape (F,) — per-feature mean from training set.
                          If provided (with train_std), uses global normalization.
            train_std   : np.ndarray, shape (F,) — per-feature std from training set.
        """
        self.data        = torch.from_numpy(data).float()   # (T, F)
        self.window_size = window_size
        self.eps         = eps

        # Global normalization stats (shape (F, 1) for broadcasting over W)
        if train_mean is not None and train_std is not None:
            # Clip std to minimum 1e-3 to prevent near-constant features from
            # producing NaN/Inf values during normalization (std ≈ 0 → divide by ~eps → explosion).
            clipped_std = np.maximum(train_std, 1e-3)
            self.global_mean = torch.from_numpy(train_mean).float().unsqueeze(1)   # (F, 1)
            self.global_std  = torch.from_numpy(clipped_std).float().unsqueeze(1)  # (F, 1)
        else:
            self.global_mean = None
            self.global_std  = None

        T = data.shape[0]
        self.indices = list(range(0, T - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start  = self.indices[idx]
        window = self.data[start : start + self.window_size]   # (W, F)
        window = window.T                                       # (F, W) — channels-first

        if self.global_mean is not None:
            # Global normalization: use training set statistics
            return (window - self.global_mean) / (self.global_std + self.eps)
        else:
            # Legacy per-window Z-score normalization
            mean = window.mean(dim=1, keepdim=True)   # (F, 1)
            std  = window.std(dim=1,  keepdim=True)   # (F, 1)
            return (window - mean) / (std + self.eps)


# ---------------------------------------------------------------------------
# 5. Full Pipeline Helper
# ---------------------------------------------------------------------------

def build_dataloader(
    filepath: str,
    window_size: int  = 64,
    stride: int       = 1,
    batch_size: int   = 256,
    shuffle: bool     = True,
    label_col: Optional[str] = None,
    num_workers: int  = 0,
    global_scale: bool = False
) -> Tuple[DataLoader, Optional[np.ndarray]]:
    """
    End-to-end convenience function: CSV path → ready-to-use PyTorch DataLoader.

    Uses lazy windowing via SMDWindowDataset — memory footprint is O(T×F) regardless
    of stride, making it safe for datasets with 500,000+ timesteps.

    Steps internally:
        load_csv → [optional: StandardScaler] → SMDWindowDataset (lazy) → DataLoader

    Args:
        filepath     : Path to smd_train.csv or smd_test.csv.
        window_size  : Sliding window length W.
        stride       : Sliding window step.
        batch_size   : Mini-batch size.
        shuffle      : True for training (randomises order), False for inference
                       (preserves temporal order for plotting).
        label_col    : Column name for anomaly labels (test set only).
        num_workers  : Parallel data loading workers (0 = main process only,
                       safe on Windows; increase on Linux for speed).
        global_scale : If True, apply sklearn StandardScaler to normalise each
                       feature column across the full time-series before windowing.
                       This is a global (cross-timestep) scale step; instance-wise
                       Z-score normalisation per window is always applied on top
                       inside SMDWindowDataset.__getitem__.

    Returns:
        loader : PyTorch DataLoader, ready for training or inference.
        labels : Raw label array aligned with the original time-series T,
                 or None if not provided. NOTE: these are *per-timestep* labels,
                 not per-window; alignment is handled in the notebook.
    """
    data, labels = load_csv(filepath, label_col=label_col)

    if global_scale:
        scaler = StandardScaler()
        data   = scaler.fit_transform(data).astype(np.float32)
        print(f"[Preprocess] Global StandardScaler applied — "
              f"each feature column is now zero-mean / unit-variance across all timesteps.")

    dataset = SMDWindowDataset(data, window_size=window_size, stride=stride)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available()   # speeds up CPU→GPU tensor transfer
    )

    print(f"[Preprocess] DataLoader ready: {len(dataset)} windows, "
          f"batch_size={batch_size}, shuffle={shuffle}\n")
    return loader, labels


# ---------------------------------------------------------------------------
# 6. SMD-Native Loader  (OmniAnomaly / ServerMachineDataset format)
# ---------------------------------------------------------------------------

def load_smd_machine(
    data_dir: str,
    machine_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one machine's data from the OmniAnomaly ServerMachineDataset layout.

    Expected directory structure (clone of https://github.com/NetManAIOps/OmniAnomaly):
        <data_dir>/
            train/          machine-1-1.txt, machine-1-2.txt, ...
            test/           machine-1-1.txt, ...
            test_label/     machine-1-1.txt, ...

    File format: plain comma-separated floats, NO header row, 38 columns.
    Labels file: one integer (0 or 1) per row, NO header.

    Available machine names (28 total):
        machine-1-1  …  machine-1-8
        machine-2-1  …  machine-2-9
        machine-3-1  …  machine-3-11

    Args:
        data_dir     : Path to the ServerMachineDataset root directory.
        machine_name : Machine identifier string, e.g. 'machine-1-1'.

    Returns:
        train_data  : np.ndarray, shape (T_train, 38) — normal training samples.
        test_data   : np.ndarray, shape (T_test,  38) — test samples (mixed normal/anomalous).
        test_labels : np.ndarray, shape (T_test,)     — binary labels (0=normal, 1=anomaly).
    """
    from pathlib import Path
    root = Path(data_dir)

    train_path = root / 'train'      / f'{machine_name}.txt'
    test_path  = root / 'test'       / f'{machine_name}.txt'
    label_path = root / 'test_label' / f'{machine_name}.txt'

    for p in (train_path, test_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"[Preprocess] Expected file not found: {p}\n"
                f"  Make sure data_dir points to the ServerMachineDataset root\n"
                f"  (clone https://github.com/NetManAIOps/OmniAnomaly and use\n"
                f"   data_dir='path/to/OmniAnomaly/ServerMachineDataset')"
            )

    train_data  = np.genfromtxt(train_path, delimiter=',', dtype=np.float32)
    test_data   = np.genfromtxt(test_path,  delimiter=',', dtype=np.float32)
    test_labels = np.genfromtxt(label_path, delimiter=',', dtype=np.float32)

    print(f"[Preprocess] Loaded '{machine_name}':")
    print(f"  train : {train_data.shape}   (T={train_data.shape[0]}, F={train_data.shape[1]})")
    print(f"  test  : {test_data.shape}    (T={test_data.shape[0]},  F={test_data.shape[1]})")
    print(f"  labels: {test_labels.shape}  "
          f"  anomaly ratio={test_labels.mean()*100:.2f}%")

    return train_data, test_data, test_labels


# ---------------------------------------------------------------------------
# 7. Array-first DataLoader Builder  (for pre-loaded / SMD-native data)
# ---------------------------------------------------------------------------

def build_dataloader_from_array(
    data: np.ndarray,
    window_size: int = 64,
    stride: int      = 1,
    batch_size: int  = 256,
    shuffle: bool    = True,
    num_workers: int = 0,
    global_scale: bool = False,
    train_mean: Optional[np.ndarray] = None,
    train_std: Optional[np.ndarray] = None,
) -> DataLoader:
    """
    Build a DataLoader directly from a NumPy array, bypassing CSV loading.

    Use this when data is already in memory — e.g. after calling load_smd_machine()
    or when concatenating multiple machines.

    Args:
        data         : np.ndarray, shape (T, F) — raw time-series data.
        window_size  : Sliding window length W.
        stride       : Step between consecutive windows.
        batch_size   : Mini-batch size.
        shuffle      : True for training, False for inference (preserves order).
        num_workers  : DataLoader worker processes.
        global_scale : If True, apply sklearn StandardScaler per feature column
                       before constructing the lazy dataset.
        train_mean   : np.ndarray, shape (F,) — per-feature mean from training set.
                       Pass to SMDWindowDataset for global normalization.
        train_std    : np.ndarray, shape (F,) — per-feature std from training set.

    Returns:
        loader : PyTorch DataLoader ready for training or inference.
    """
    if global_scale:
        scaler = StandardScaler()
        data   = scaler.fit_transform(data).astype(np.float32)
        print(f"[Preprocess] Global StandardScaler applied.")

    dataset = SMDWindowDataset(
        data, window_size=window_size, stride=stride,
        train_mean=train_mean, train_std=train_std,
    )
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available()
    )

    norm_mode = "global" if train_mean is not None else "per-window Z-score"
    print(f"[Preprocess] DataLoader ready: {len(dataset)} windows, "
          f"batch_size={batch_size}, shuffle={shuffle}, norm={norm_mode}\n")
    return loader


def filter_contaminated_windows(
    data: np.ndarray,
    model,
    device,
    window_size: int = 64,
    batch_size: int = 256,
    contamination_ratio: float = 0.01,
) -> np.ndarray:
    """
    Remove potentially contaminated timesteps from training data.

    After an initial training pass, windows with the highest reconstruction
    error likely contain subtle anomalies that leaked into the training set.
    This function identifies and removes those windows' timesteps.

    Args:
        data                : np.ndarray, shape (T, F) — raw training data.
        model               : Trained VAE model (used in eval mode).
        device              : torch.device.
        window_size         : Sliding window length W.
        batch_size          : Batch size for error computation.
        contamination_ratio : Fraction of windows to remove (default 0.01 = top 1%).
    Returns:
        clean_data : np.ndarray, shape (T', F) where T' <= T.
    """
    from vae_model import compute_reconstruction_errors

    # Build a non-overlapping loader for efficiency
    loader = build_dataloader_from_array(
        data, window_size=window_size, stride=window_size,
        batch_size=batch_size, shuffle=False
    )

    errors = compute_reconstruction_errors(model, loader, device).numpy()

    # Find the threshold: top contamination_ratio% of errors
    threshold = np.quantile(errors, 1.0 - contamination_ratio)
    contaminated_mask = errors > threshold

    # Map contaminated windows back to timestep indices
    n_windows = len(errors)
    bad_timesteps = set()
    for i in range(n_windows):
        if contaminated_mask[i]:
            start = i * window_size
            end = min(start + window_size, len(data))
            bad_timesteps.update(range(start, end))

    # Keep only clean timesteps
    keep_mask = np.ones(len(data), dtype=bool)
    for t in bad_timesteps:
        if t < len(data):
            keep_mask[t] = False

    clean_data = data[keep_mask]
    removed_pct = 100 * (1 - len(clean_data) / len(data))
    print(f"[Preprocess] Contamination filter: removed {len(data) - len(clean_data)} "
          f"timesteps ({removed_pct:.1f}%), {len(clean_data)} remaining")

    return clean_data
