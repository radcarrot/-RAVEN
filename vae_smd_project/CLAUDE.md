# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Session continuity:** Always read `primer.md` at the start of a session to restore context from the previous session. Update `primer.md` at the end of every session: move "This Session" into "Previous Sessions" and write a new "This Session" block.

## Project Overview

VAE-based unsupervised time-series anomaly detection for the Server Machine Dataset (SMD). The model trains on normal server metrics only; anomalous windows produce abnormally high reconstruction error at inference time.

No build system. The workflow is entirely notebook-driven — run `main_demo.ipynb` cell-by-cell in Jupyter.

**Dependencies (`requirements.txt` is provided):**
```bash
pip install -r requirements.txt
```
For CUDA support (RTX 3050 Ti / any NVIDIA GPU), install PyTorch separately first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Data setup:** Clone OmniAnomaly (`NetManAIOps/OmniAnomaly` on GitHub) and place its `ServerMachineDataset/` folder at `./data/ServerMachineDataset/`. Set `MACHINE_NAME` in Section 2 (28 machines: machine-1-1…machine-1-8, machine-2-1…machine-2-9, machine-3-1…machine-3-11).

---

## Architecture & Data Flow

Two workflows:

**1. Interactive (single model):** notebook-driven, run `main_demo.ipynb` cell-by-cell.
```
preprocess.py  →  vae_model.py   →  main_demo.ipynb  →  results_logger.py
(raw arrays)      (TCN-VAE defs)    (orchestration)      (save to results/)
                       ↑
               evaluation.py
            (threshold + metrics)
```

**2. Batch comparison (all 3 models × 28 machines):**
```
preprocess.py  →  vae_model.py     ─┐
                  lstm_model.py     ─┼→  train_compare.py  →  results_logger.py
                  mamba_model.py    ─┘    (training loop)      (save to results/)
                       ↑                                            ↓
               evaluation.py                              plot_results.py
            (threshold + metrics)                        (figures + tables)
```

Standalone utility modules (import as needed, not used by either workflow):
```
multiscale.py   — multi-scale ensemble (window sizes 32/64/128)
flows.py        — PlanarFlow normalizing flow prior
```

### `preprocess.py`

Two loader paths — the SMD-native path is the one used by the notebook:

- **`load_smd_machine(data_dir, machine_name)`** — reads headerless comma-separated `.txt` files from `train/`, `test/`, `test_label/` subdirs; returns `(train_data, test_data, test_labels)` as `float32` arrays of shape `(T, 38)`.
- **`build_dataloader_from_array(data, ..., train_mean=, train_std=)`** — wraps a NumPy array in `SMDWindowDataset` and returns a `DataLoader`. Pass `train_mean`/`train_std` (shape `(F,)`) for global normalization.
- **`load_csv()` / `build_dataloader()`** — legacy single-CSV path, kept for compatibility.
- **`filter_contaminated_windows(data, model, device, ...)`** — after initial training, removes top `contamination_ratio` (default 1%) highest-error windows from training data. Toggle `FILTER_CONTAMINATION=True` in notebook to use.

**`SMDWindowDataset`** does lazy windowing: stores only the raw `(T, F)` tensor and extracts + normalises each `(F, W)` window in `__getitem__`. Memory is O(T×F), not O(N×F×W). Two normalization modes: **global** (pass `train_mean`/`train_std` — recommended, preserves level-shift anomalies) or **per-window Z-score** (legacy, omit stats). `sliding_window()` and `normalize_windows()` are standalone utilities but are not used by the notebook — `SMDWindowDataset` replaces them.

### `vae_model.py` — v3: TCN + Feature Attention + Learned Variance

**Encoder pipeline:** `(B, F, W)` → `FeatureAttention` → 4× `TemporalBlock` (dilations 1,2,4,8) → `AdaptiveAvgPool1d(8)` → flatten → two FC heads → `(z_μ, z_log_var)`

**Decoder pipeline:** `z (B, latent_dim)` → FC → reshape `(B, 2H, 8)` → 3× `ConvTranspose1d` → `Upsample(size=W)` → split:
  - `FeatureAttention` → `x_μ (B, F, W)` (reconstruction mean)
  - `Conv1d(1×1)` → `x_log_var (B, F, W)` (learned per-element log-variance, clamped to [-4, 2])

**`VAE.forward` returns 4 tensors:** `(x_mu, x_log_var, z_mu, z_log_var)`.

**Key classes:**
- `FeatureAttention(n_features, window_size, d_model, n_heads)` — treats each sensor channel as a token; `proj_in` is a `Linear(window_size, d_model)`, so **`window_size` is baked into the module weights**.
- `TemporalBlock(in_ch, out_ch, dilation)` — dilated Conv1d + BN + GELU + residual (1×1 conv when channels differ). Receptive field of 4-block stack = 61 timesteps, covering the full default 64-step window.
- `ELBOLoss(warmup_epochs)` — Gaussian NLL reconstruction + β-annealed KL. Reconstruction loss: `0.5 * mean(x_log_var + (x - x_mu)² / exp(x_log_var))`.
- `compute_reconstruction_errors(model, dataloader, device)` → `(N,)` per-window MSE tensor. Ignores learned variance — may be preferable to NLL scoring since the variance head can absorb reconstruction error and hide anomalies (see primer.md "Proposed Changes").
- `compute_anomaly_scores(model, dataloader, device, n_samples=20, topk=3, kl_weight=0.1)` → `(N,)` MC-averaged score combining Gaussian NLL (top-k feature aggregation) + weighted KL divergence. **Notebook uses `kl_weight=0.0`** (KL collapsed to ~0 during training with `MAX_BETA=1.0`; including it adds noise). Default signature kept at 0.1 for API compatibility.

### `lstm_model.py` — BiLSTM-VAE

**Encoder pipeline:** `(B, F, W)` → `FeatureAttention(d_model=64)` → permute to `(B, W, F)` → 2-layer BiLSTM(hidden=128/dir) → concat final hidden states `[h_fwd, h_bwd]` → FC → `(z_μ, z_log_var)`

**Key design choices:**
- `attn_d_model=64` decouples FeatureAttention from `lstm_hidden` — ensures identical FeatureAttention weights (25,088 params) across all three models for controlled comparison.
- Decoder uses `tcn_equiv = lstm_hidden // 2 = 64` so the shared Decoder has identical parameter count (105,296) across models.
- Total params: **714,128** (encoder 608,832 — ~2.7× larger than TCN/Mamba due to BiLSTM recurrent weight matrices).

**Key class:**
- `LSTMVAE(in_channels, latent_dim, window_size, lstm_hidden=128, num_layers=2, n_heads=4, attn_d_model=64)` — `.forward()` returns `(x_mu, x_log_var, z_mu, z_log_var)`, same 4-tensor API as `VAE`.
- `compute_reconstruction_errors(model, dataloader, device)` → `(N,)` per-window MSE tensor. Same API as `vae_model.compute_reconstruction_errors`.

### `mamba_model.py` — Mamba-2 SSD VAE (Pure PyTorch)

**Encoder pipeline:** `(B, F, W)` → `FeatureAttention(d_model=64)` → `Linear(F, d_model)` → 4× `MambaBlock` → mean pool → FC → `(z_μ, z_log_var)`

**Selective scan implementation:** `selective_scan_seq()` — sequential O(L) scan with Zero-Order Hold discretization:
- `Ā = exp(Δ * A)`, `B̄ = Δ * B` (ZOH discretization per timestep)
- State `h(t) = Ā(t) * h(t-1) + B̄(t) * u(t)`, output `y(t) = C(t) * h(t) + D * u(t)`
- Pure-PyTorch loop over L timesteps — no CUDA extension required, but slower than `mamba-ssm` package.

**MambaBlock architecture:** LayerNorm → Linear expand(d_model → d_inner=2×d_model) → depthwise Conv1d(kernel=4) → selective SSM(d_state=16) → SiLU gate → Linear contract(d_inner → d_model) → residual connection.

**Key class:**
- `MambaVAE(in_channels, latent_dim, window_size, d_model=64, d_state=16, n_blocks=4, n_heads=4)` — `.forward()` returns `(x_mu, x_log_var, z_mu, z_log_var)`, same 4-tensor API.
- `compute_reconstruction_errors(model, dataloader, device)` → `(N,)` per-window MSE tensor.
- Total params: **268,240** (parameter-matched to TCN-VAE at ~264K).

### `evaluation.py`

- `pot_threshold(train_scores, q=0.80, level=1e-2)` — Generalized Pareto Distribution fit to error tail; returns threshold at target false alarm rate. (`level=1e-4` was too conservative; `1e-2` gives a more permissive threshold closer to the 99th pct.)
- `point_adjusted_f1(predictions, labels)` — OmniAnomaly protocol: if any prediction in a GT anomaly segment is positive, the whole segment counts as detected.

### `multiscale.py` (standalone)

- `train_multiscale_models(train_data, device, window_sizes=[32,64,128], ...)` — trains a separate VAE per window size.
- `multiscale_ensemble_scores(test_data, models, device, ...)` — Z-normalizes each model's scores, aligns to timestep axis, returns `np.nanmean` across scales.

### `flows.py` (standalone)

- `PlanarFlow(latent_dim, n_flows=8)` — stack of planar transforms (Rezende & Mohamed 2015). Integration guide is in the module docstring: insert between `reparameterize()` and `decoder()`, subtract `log_det.mean()` from KL loss, include `flow.parameters()` in the optimizer.

### `train_compare.py` — Batch training + evaluation

Trains TCN-VAE, LSTM-VAE, and Mamba-VAE on all 28 SMD machines with identical hyperparameters (except architecture-specific ones). Saves structured results via `RunLogger`.

**Key components:**
- `OMNI_BASELINES` — dict with published OmniAnomaly PA-F1 for all 28 machines.
- `train_model(model, loader, device, ...)` → `(model, history_dict)` where `history = {'total': [...], 'recon': [...], 'kl': [...], 'beta': [...]}`.
- `evaluate(model, train_data, test_data, test_labels, ...)` → `(metrics_dict, raw_arrays_dict)`. Returns both JSON-safe metrics and raw numpy arrays (prefixed with `_`) for RunLogger.
- Main loop integrates `RunLogger`: each model×machine produces a saved run directory under `results/`.
- Crash-safe: writes incremental `comparison.json` after each model completes.

**Usage:**
```bash
python train_compare.py                           # all 28 machines × 3 models
python train_compare.py --machines machine-1-1    # single machine
python train_compare.py --archs TCN LSTM          # subset of architectures
```

### `results_logger.py` — Structured experiment persistence

**`RunLogger`** accumulates outputs from a single training run and saves to a structured directory:
- `run_summary.json` — metadata, hyperparams, metrics, score statistics, thresholds
- `training_history.csv` — per-epoch total/recon/KL/beta losses
- `*.npy` files — train_scores, test_scores_raw, test_scores (smoothed), predictions, aligned_labels

**API (call from notebook or train_compare.py):**
```python
logger = RunLogger('machine-1-1', 'TCN-VAE')
logger.log_hyperparameters(WINDOW_SIZE=64, ...)
logger.log_training(history, train_time_s=elapsed)
logger.log_train_scores(train_scores, THRESHOLD, THRESHOLD_POT, THRESHOLD_PCT99)
logger.log_test_results(test_scores_raw, test_scores, predictions, labels, raw_f1, pa_f1_dict, roc_auc)
logger.save()  # → results/machine-1-1_TCN-VAE_20260328_143022/
```

**Loading:** `RunLogger.load(run_dir)` reconstructs all fields. `collect_all_runs('results/')` scans for all saved runs.

### `plot_results.py` — Visualization from saved runs

Reads `results/` directory via `collect_all_runs()` and generates publication-quality figures to `results/figures/`:
- `loss_curves_{machine}.png` — training loss curves (all archs overlaid)
- `kl_curves_{machine}.png` — KL divergence over epochs
- `score_dist_{machine}_{arch}.png` — train/test score distributions with threshold
- `anomaly_timeline_{machine}_{arch}.png` — test scores vs labels with threshold line
- `comparison_bar.png` — PA-F1 bar chart across machines × architectures
- `summary_table.md` — markdown comparison table

Uses Agg backend for headless environments. Run after `train_compare.py` completes.

### `main_demo.ipynb`

7 sections (imports → hyperparams/data → model → training → threshold → inference/metrics → visualization).

**Hyperparameters (Section 2):**

| Parameter | Default | Constraint |
|-----------|---------|------------|
| `WINDOW_SIZE` | 64 | Must match model's `FeatureAttention` — changing it invalidates saved checkpoints |
| `LATENT_DIM` | 32 | |
| `TCN_HIDDEN` | 64 | Must satisfy `TCN_HIDDEN % N_HEADS == 0` AND be even (decoder uses `tcn_hidden // 2`) |
| `N_HEADS` | 4 | |
| `EPOCHS` | 100 | |
| `LR` | 1e-3 | Adam |
| `WARMUP_EPOCHS` | 25 | KL annealing duration |
| `MAX_BETA` | 1.0 | KL weight ceiling; KL collapses regardless (0.1 and 1.0 both → KL≈0.002). TCN decoder is powerful enough to ignore the latent code. |
| `BATCH_SIZE` | 256 | |

**Training loop (Section 4):** gradient clipping `max_norm=1.0`, `ReduceLROnPlateau(patience=5, factor=0.5)`.

**Threshold calibration (Section 5):** reuses `train_data` in memory with `stride=WINDOW_SIZE` (non-overlapping) for speed. Uses `compute_reconstruction_errors()` (MSE) for scoring. Computes both **99th percentile (active)** and POT threshold (reference) via `evaluation.pot_threshold(q=0.80, level=1e-2)`. `THRESHOLD = THRESHOLD_PCT99`.

**Label alignment (Section 6):** predictions are per-window; each window's anomaly score is assigned to its **last timestep**. Aligned labels: `test_labels_raw[WINDOW_SIZE-1 : WINDOW_SIZE-1 + n_windows]`. The first `WINDOW_SIZE-1` timesteps have no prediction.

**Checkpoint format saved to `vae_smd_trained.pt`:**
```python
{
    'model_state_dict': ...,
    'hyperparameters': {'in_channels', 'latent_dim', 'window_size', 'tcn_hidden', 'n_heads'},
    'threshold': float,          # active threshold (99th pct)
    'threshold_pot': float,      # POT threshold
    'threshold_pct99': float,    # 99th percentile threshold (reference)
    'final_epoch_loss': float,
}
```
All hyperparameters needed to reconstruct the model are saved. Load with:
```python
ckpt = torch.load('vae_smd_trained.pt')
model = VAE(**ckpt['hyperparameters']).to(device)
model.load_state_dict(ckpt['model_state_dict'])
threshold = ckpt['threshold']
```

**Outputs written to disk:** `loss_curves.png`, `train_error_dist.png`, `anomaly_detection_result.png`, `vae_smd_trained.pt`, plus `results/{machine}_{arch}_{timestamp}/` via `RunLogger` (run_summary.json, training_history.csv, .npy score arrays).

---

## Non-Obvious Constraints

- **`TCN_HIDDEN % N_HEADS == 0`** — enforced by an `assert` in `FeatureAttention.__init__` because `d_model=tcn_hidden` is passed to `nn.MultiheadAttention`.
- **`TCN_HIDDEN` must be even** — the decoder's middle `ConvTranspose1d` uses `tcn_hidden // 2` channels; odd values will silently produce wrong dimensions.
- **`WINDOW_SIZE` baked into weights** — `FeatureAttention.proj_in` is `Linear(window_size, d_model)`. Changing `WINDOW_SIZE` requires a fresh model; you cannot load a saved checkpoint trained with a different `WINDOW_SIZE`.
- **`num_workers=0` on Windows** — both `build_dataloader` and `build_dataloader_from_array` default to `num_workers=0`, which is correct for Windows where PyTorch DataLoader multiprocessing requires a `if __name__ == '__main__'` guard that Jupyter doesn't provide.
- **Decoder has no final activation on the mean head** — reconstruction targets are real-valued globally-normalized data; do not add `Tanh`/`Sigmoid` to the mean output. The `log_var_head` is clamped to `[-20, 2]`. With MSE scoring the clamp doesn't affect inference — it only matters for NLL training gradients, and `-20` gives far stronger gradients for `x_mu`.
- **Decoder returns a tuple** — `(x_mu, x_log_var)`, not a single tensor. All code unpacking decoder/model output must handle 2 or 4 values respectively.
- **Two F1 scores are reported** — Section 6 outputs both raw binary F1 and point-adjusted F1 (OmniAnomaly protocol via `evaluation.point_adjusted_f1`). Point-adjusted F1 is always higher; use it for comparison with published benchmarks, raw F1 for strict evaluation.
- **MSE scoring is now the default** — Section 5 and Section 6 use `compute_reconstruction_errors()` (per-window MSE) instead of `compute_anomaly_scores()` (MC-averaged NLL). MSE ignores the learned variance head, preventing the model from hiding anomalies by predicting high variance. `compute_anomaly_scores()` is kept for comparison.
- **Score smoothing before thresholding** — `uniform_filter1d(size=SMOOTH_KERNEL)` is applied to raw MSE scores in Section 6 before thresholding. This means the threshold (calibrated on unsmoothed training scores) operates on smoothed test scores — this is intentional, as it reduces false positives without affecting threshold calibration on i.i.d. training data.
- **Global normalization is now the default** — `SMDWindowDataset` uses train-set `mean`/`std` (shape `(F,)`) when provided. The notebook computes these from `train_data` and passes them to all dataloaders. This preserves level-shift anomalies that per-window Z-score erased. Per-window Z-score mode is still available by omitting `train_mean`/`train_std`.
- **Global normalization NaN risk (FIXED)** — `SMDWindowDataset` now clips `train_std` to `max(std, 1e-3)` in `__init__` before computing the global normalization divisor. This prevents NaN/Inf from near-constant features where `train_std ≈ 0`. Previously observed in session 4 (F1=0, ROC-AUC failed). The fix is in `preprocess.py:SMDWindowDataset.__init__`.
- **All three models share the same 4-tensor forward API** — `VAE`, `LSTMVAE`, and `MambaVAE` all return `(x_mu, x_log_var, z_mu, z_log_var)`. All three share the same `Decoder` class (from `vae_model.py`) and `FeatureAttention(d_model=64)`. Code that handles one model's output works with all three.
- **LSTM's `attn_d_model` must stay 64** — `LSTMEncoder` has an `attn_d_model` parameter (default 64) that controls `FeatureAttention.d_model`, decoupled from `lstm_hidden`. Changing it breaks the controlled comparison across architectures.
- **`point_adjusted_f1()` returns a dict, not a float** — `evaluation.point_adjusted_f1(preds, labels)` returns `{'f1': float, 'precision': float, 'recall': float, 'adjusted_predictions': ndarray}`. Extract `['f1']` for the score. The `adjusted_predictions` array is not JSON-serializable.
- **Mamba selective scan is pure PyTorch** — `mamba_model.py:selective_scan_seq()` uses a Python loop over timesteps. It works on CPU and GPU without CUDA extensions but is significantly slower than the `mamba-ssm` package. Sufficient for research comparison; not production-speed.
