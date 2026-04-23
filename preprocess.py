"""
preprocess.py
-------------
Data ingestion and preprocessing pipeline for the SMD (Server Machine Dataset).

Pipeline:
  1. Load raw multivariate time-series data.
  2. Wrap in SMDWindowDataset (lazy sliding-window extraction + normalization).
  3. Build a PyTorch DataLoader for training or inference.

Author: VAE-SMD Mini-Project
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Data Ingestion (legacy CSV path)
# ---------------------------------------------------------------------------

def load_csv(
    filepath: str,
    label_col: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a multivariate time-series CSV file into NumPy arrays.

    Args:
        filepath  : Path to the CSV file.
        label_col : (Optional) Name of the binary anomaly label column.
                    If None or absent, labels are not returned.

    Returns:
        data   : np.ndarray, shape (T, F).
        labels : np.ndarray, shape (T,), or None.
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
# 2. PyTorch Dataset
# ---------------------------------------------------------------------------

class SMDWindowDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset with lazy (on-demand) windowing.

    Stores only the raw (T, F) time-series; extracts + normalises each window
    in __getitem__. Memory footprint: O(T×F) instead of O(N×F×W).
    For T=500k, F=38, W=64, stride=1 this is ~73 MB vs ~4.9 GB.

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
            train_std   : np.ndarray, shape (F,) — per-feature std from training set.
                          Dead features (std == 0) get std=1.0 so their raw deviation
                          maps directly to normalized space (bounded by clamp).
                          Near-constant features use their actual std (eps floor only).
        """
        self.data        = torch.from_numpy(data).float()   # (T, F)
        self.window_size = window_size
        self.eps         = eps

        # Global normalization stats (shape (F, 1) for broadcasting over W)
        if train_mean is not None and train_std is not None:
            # Zero-variance (dead) features: set std to 1.0 so raw values map
            # directly to normalized space. Features 26/28 on machine-1-1 are
            # binary (0 during normal, 1 during anomalies) — with std=1.0 a
            # spike of 1.0 normalizes to 1.0, preserving the anomaly signal.
            # Near-constant active features: use actual std with 1e-3 floor.
            clipped_std = np.where(train_std == 0, 1.0, np.maximum(train_std, 1e-3))
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
            normalized = (window - self.global_mean) / (self.global_std + self.eps)
            return normalized.clamp(-10.0, 10.0)
        else:
            mean = window.mean(dim=1, keepdim=True)   # (F, 1)
            std  = window.std(dim=1,  keepdim=True)   # (F, 1)
            return (window - mean) / (std + self.eps)


# ---------------------------------------------------------------------------
# 3. SMD-Native Loader  (OmniAnomaly / ServerMachineDataset format)
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

    Args:
        data_dir     : Path to the ServerMachineDataset root directory.
        machine_name : Machine identifier string, e.g. 'machine-1-1'.

    Returns:
        train_data  : np.ndarray, shape (T_train, 38).
        test_data   : np.ndarray, shape (T_test, 38).
        test_labels : np.ndarray, shape (T_test,).
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
# 4. Array-first DataLoader Builder
# ---------------------------------------------------------------------------

def build_dataloader_from_array(
    data: np.ndarray,
    window_size: int = 64,
    stride: int      = 1,
    batch_size: int  = 256,
    shuffle: bool    = True,
    num_workers: int = 0,
    train_mean: Optional[np.ndarray] = None,
    train_std: Optional[np.ndarray] = None,
) -> DataLoader:
    """
    Build a DataLoader directly from a NumPy array, bypassing CSV loading.

    Use this when data is already in memory — e.g. after load_smd_machine().

    Args:
        data         : np.ndarray, shape (T, F).
        window_size  : Sliding window length W.
        stride       : Step between consecutive windows.
        batch_size   : Mini-batch size.
        shuffle      : True for training, False for inference (preserves order).
        num_workers  : DataLoader worker processes (0 = required on Windows).
        train_mean   : np.ndarray, shape (F,) — per-feature mean for global normalization.
        train_std    : np.ndarray, shape (F,) — per-feature std for global normalization.
                       Dead features (std==0) get std=1.0; others use actual std.

    Returns:
        loader : PyTorch DataLoader ready for training or inference.
    """
    dataset = SMDWindowDataset(
        data, window_size=window_size, stride=stride,
        train_mean=train_mean, train_std=train_std,
    )
    loader = DataLoader(
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
    score_fn=None,
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
        score_fn            : callable(model, dataloader, device) → Tensor(N,).
                              Defaults to vae_model.compute_reconstruction_errors.
                              Pass the matching function for LSTM/Mamba models.
        window_size         : Sliding window length W.
        batch_size          : Batch size for error computation.
        contamination_ratio : Fraction of windows to remove (default 0.01 = top 1%).

    Returns:
        clean_data : np.ndarray, shape (T', F) where T' <= T.
    """
    if score_fn is None:
        from vae_model import compute_reconstruction_errors
        score_fn = compute_reconstruction_errors

    loader = build_dataloader_from_array(
        data, window_size=window_size, stride=window_size,
        batch_size=batch_size, shuffle=False
    )

    errors = score_fn(model, loader, device).numpy()

    threshold = np.quantile(errors, 1.0 - contamination_ratio)
    contaminated_mask = errors > threshold

    n_windows = len(errors)
    bad_timesteps = set()
    for i in range(n_windows):
        if contaminated_mask[i]:
            start = i * window_size
            end = min(start + window_size, len(data))
            bad_timesteps.update(range(start, end))

    keep_mask = np.ones(len(data), dtype=bool)
    for t in bad_timesteps:
        if t < len(data):
            keep_mask[t] = False

    clean_data = data[keep_mask]
    removed_pct = 100 * (1 - len(clean_data) / len(data))
    print(f"[Preprocess] Contamination filter: removed {len(data) - len(clean_data)} "
          f"timesteps ({removed_pct:.1f}%), {len(clean_data)} remaining")

    return clean_data
