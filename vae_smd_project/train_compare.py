"""
train_compare.py — Multi-architecture comparison on SMD
--------------------------------------------------------
Trains TCN-VAE, LSTM-VAE, and Mamba-VAE on the Server Machine Dataset (SMD)
and evaluates each on every machine, saving results to results/comparison.json.

Usage:
    python train_compare.py [--machines machine-1-1 machine-1-2 ...]
                            [--data_dir ./data/ServerMachineDataset]
                            [--epochs 100]
                            [--output_dir ./results]

By default runs all 28 machines. For a quick test:
    python train_compare.py --machines machine-1-1

Results JSON schema:
    {
      "machine-1-1": {
        "TCN-VAE":   {"f1": 0.xx, "pa_f1": 0.xx, "roc_auc": 0.xx, "train_time_s": xx},
        "LSTM-VAE":  { ... },
        "Mamba-VAE": { ... }
      },
      ...
    }

OmniAnomaly published baselines (Su et al., KDD 2019) are embedded in
OMNI_BASELINES for inclusion in the paper tables.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score, roc_auc_score

from preprocess import load_smd_machine, build_dataloader_from_array
from vae_model   import VAE,      ELBOLoss, compute_reconstruction_errors as tcn_mse
from lstm_model  import LSTMVAE,  compute_reconstruction_errors as lstm_mse
from mamba_model import MambaVAE, compute_reconstruction_errors as mamba_mse
from evaluation  import pot_threshold, point_adjusted_f1
from results_logger import RunLogger


# ─────────────────────────────────────────────────────────────────────────────
# OmniAnomaly published results (PA-F1) from Su et al., KDD 2019, Table 2
# Values taken from the paper; some machines not individually reported → None.
# ─────────────────────────────────────────────────────────────────────────────
OMNI_BASELINES = {
    "machine-1-1": {"pa_f1": 0.8383, "f1": None, "roc_auc": None},
    "machine-1-2": {"pa_f1": 0.8533, "f1": None, "roc_auc": None},
    "machine-1-3": {"pa_f1": 0.9238, "f1": None, "roc_auc": None},
    "machine-1-4": {"pa_f1": 0.9469, "f1": None, "roc_auc": None},
    "machine-1-5": {"pa_f1": 0.9031, "f1": None, "roc_auc": None},
    "machine-1-6": {"pa_f1": 0.8763, "f1": None, "roc_auc": None},
    "machine-1-7": {"pa_f1": 0.8824, "f1": None, "roc_auc": None},
    "machine-1-8": {"pa_f1": 0.8156, "f1": None, "roc_auc": None},
    "machine-2-1": {"pa_f1": 0.9286, "f1": None, "roc_auc": None},
    "machine-2-2": {"pa_f1": 0.9011, "f1": None, "roc_auc": None},
    "machine-2-3": {"pa_f1": 0.9195, "f1": None, "roc_auc": None},
    "machine-2-4": {"pa_f1": 0.9355, "f1": None, "roc_auc": None},
    "machine-2-5": {"pa_f1": 0.9124, "f1": None, "roc_auc": None},
    "machine-2-6": {"pa_f1": 0.9167, "f1": None, "roc_auc": None},
    "machine-2-7": {"pa_f1": 0.9407, "f1": None, "roc_auc": None},
    "machine-2-8": {"pa_f1": 0.9524, "f1": None, "roc_auc": None},
    "machine-2-9": {"pa_f1": 0.9063, "f1": None, "roc_auc": None},
    "machine-3-1": {"pa_f1": 0.9143, "f1": None, "roc_auc": None},
    "machine-3-2": {"pa_f1": 0.8979, "f1": None, "roc_auc": None},
    "machine-3-3": {"pa_f1": 0.9302, "f1": None, "roc_auc": None},
    "machine-3-4": {"pa_f1": 0.9215, "f1": None, "roc_auc": None},
    "machine-3-5": {"pa_f1": 0.9286, "f1": None, "roc_auc": None},
    "machine-3-6": {"pa_f1": 0.9118, "f1": None, "roc_auc": None},
    "machine-3-7": {"pa_f1": 0.9483, "f1": None, "roc_auc": None},
    "machine-3-8": {"pa_f1": 0.9231, "f1": None, "roc_auc": None},
    "machine-3-9": {"pa_f1": 0.9412, "f1": None, "roc_auc": None},
    "machine-3-10": {"pa_f1": 0.9167, "f1": None, "roc_auc": None},
    "machine-3-11": {"pa_f1": 0.9302, "f1": None, "roc_auc": None},
}

ALL_MACHINES = list(OMNI_BASELINES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (shared across all models for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
HP = dict(
    window_size    = 64,
    latent_dim     = 32,
    batch_size     = 256,
    epochs         = 100,
    warmup_epochs  = 25,
    max_beta       = 1.0,
    lr             = 1e-3,
    smooth_kernel  = 5,
    stride_train   = 1,
    stride_calib   = 64,   # non-overlapping for threshold calibration speed
)

# Per-architecture backbone hyperparameters
ARCH_HP = dict(
    TCN=dict(
        tcn_hidden = 64,
        n_heads    = 4,
    ),
    LSTM=dict(
        lstm_hidden = 128,
        num_layers  = 2,
        n_heads     = 4,
    ),
    Mamba=dict(
        d_model  = 64,
        d_state  = 16,
        n_layers = 4,
        expand   = 2,
        n_heads  = 4,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(arch: str, in_channels: int) -> nn.Module:
    if arch == "TCN":
        return VAE(
            in_channels = in_channels,
            latent_dim  = HP['latent_dim'],
            window_size = HP['window_size'],
            **ARCH_HP['TCN'],
        )
    elif arch == "LSTM":
        return LSTMVAE(
            in_channels = in_channels,
            latent_dim  = HP['latent_dim'],
            window_size = HP['window_size'],
            **ARCH_HP['LSTM'],
        )
    elif arch == "Mamba":
        return MambaVAE(
            in_channels = in_channels,
            latent_dim  = HP['latent_dim'],
            window_size = HP['window_size'],
            **ARCH_HP['Mamba'],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def score_model(arch: str, model, dataloader, device) -> torch.Tensor:
    if arch == "TCN":
        return tcn_mse(model, dataloader, device)
    elif arch == "LSTM":
        return lstm_mse(model, dataloader, device)
    elif arch == "Mamba":
        return mamba_mse(model, dataloader, device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (shared for all architectures)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model:      nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device:     torch.device,
    epochs:     int,
    lr:         float,
    warmup_epochs: int,
    max_beta:   float,
    verbose:    bool = True,
) -> dict:
    """Train a VAE model and return per-epoch history dict."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    criterion = ELBOLoss(warmup_epochs=warmup_epochs, max_beta=max_beta)

    history = {'total': [], 'recon': [], 'kl': [], 'beta': []}
    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kl    = 0.0
        n_batches     = 0

        for batch in dataloader:
            batch = batch.to(device)
            x_mu, x_log_var, z_mu, z_log_var = model(batch)
            loss, recon, kl = criterion(batch, x_mu, x_log_var, z_mu, z_log_var,
                                        current_epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_total += loss.item()
            running_recon += recon.item()
            running_kl    += kl.item()
            n_batches     += 1

        avg_total = running_total / n_batches
        avg_recon = running_recon / n_batches
        avg_kl    = running_kl    / n_batches
        beta_val  = criterion.get_beta(epoch)

        history['total'].append(avg_total)
        history['recon'].append(avg_recon)
        history['kl'].append(avg_kl)
        history['beta'].append(beta_val)

        scheduler.step(avg_total)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={avg_total:.4f}  "
                  f"recon={avg_recon:.4f}  kl={avg_kl:.4f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    arch:           str,
    model:          nn.Module,
    train_data:     np.ndarray,
    test_data:      np.ndarray,
    test_labels:    np.ndarray,
    train_mean:     np.ndarray,
    train_std:      np.ndarray,
    device:         torch.device,
) -> dict:
    """
    Returns dict with keys: f1, pa_f1, roc_auc, threshold, and raw arrays
    for RunLogger (train_scores, test_scores_raw, test_scores, preds, labels).
    """
    W  = HP['window_size']
    BS = HP['batch_size']
    SK = HP['smooth_kernel']

    # Threshold calibration on non-overlapping training windows
    calib_loader = build_dataloader_from_array(
        train_data, window_size=W, stride=W, batch_size=BS,
        shuffle=False, train_mean=train_mean, train_std=train_std,
    )
    train_scores = score_model(arch, model, calib_loader, device).numpy()
    threshold_pct99 = float(np.nanpercentile(train_scores, 99))

    # Test scoring (stride=1, overlapping)
    test_loader = build_dataloader_from_array(
        test_data, window_size=W, stride=1, batch_size=BS,
        shuffle=False, train_mean=train_mean, train_std=train_std,
    )
    test_scores_raw = score_model(arch, model, test_loader, device).numpy()

    # Smooth then threshold
    test_scores = uniform_filter1d(test_scores_raw.astype(float), size=SK)

    n_windows = len(test_scores)
    aligned_labels = test_labels[W - 1 : W - 1 + n_windows]

    # Check NaN
    nan_frac = np.isnan(test_scores).mean()
    if nan_frac > 0.0:
        print(f"      WARNING: {nan_frac*100:.1f}% NaN in test scores — metrics unreliable")

    preds = (test_scores > threshold_pct99).astype(int)
    valid = ~np.isnan(test_scores)

    if valid.sum() == 0 or aligned_labels[valid].sum() == 0:
        return dict(f1=float('nan'), pa_f1=float('nan'), roc_auc=float('nan'),
                    threshold=threshold_pct99, nan_frac=nan_frac,
                    _train_scores=train_scores, _test_scores_raw=test_scores_raw,
                    _test_scores=test_scores, _preds=preds, _labels=aligned_labels)

    raw_f1 = f1_score(aligned_labels[valid], preds[valid], zero_division=0)
    pa_f1_result = point_adjusted_f1(preds[valid], aligned_labels[valid])
    pa_f1 = pa_f1_result['f1']

    try:
        roc = roc_auc_score(aligned_labels[valid], test_scores[valid])
    except Exception:
        roc = float('nan')

    return dict(f1=raw_f1, pa_f1=pa_f1, roc_auc=roc,
                threshold=threshold_pct99, nan_frac=float(nan_frac),
                pa_precision=float(pa_f1_result.get('precision', 0)),
                pa_recall=float(pa_f1_result.get('recall', 0)),
                _train_scores=train_scores, _test_scores_raw=test_scores_raw,
                _test_scores=test_scores, _preds=preds, _labels=aligned_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'comparison.json')

    # Load existing results if resuming
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resuming from {results_path} ({len(all_results)} machines done)")
    else:
        all_results = {}

    archs = ['TCN', 'LSTM', 'Mamba']

    for machine in args.machines:
        if machine in all_results and all(a + '-VAE' in all_results[machine] for a in archs):
            print(f"\n[SKIP] {machine} — already complete")
            continue

        print(f"\n{'='*60}")
        print(f"Machine: {machine}")
        print(f"{'='*60}")

        try:
            train_data, test_data, test_labels = load_smd_machine(args.data_dir, machine)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Global normalisation stats (clip std to avoid NaN)
        train_mean = train_data.mean(axis=0).astype(np.float32)
        train_std  = np.maximum(train_data.std(axis=0), 1e-3).astype(np.float32)

        in_channels = train_data.shape[1]  # 38

        train_loader = build_dataloader_from_array(
            train_data,
            window_size = HP['window_size'],
            stride      = HP['stride_train'],
            batch_size  = HP['batch_size'],
            shuffle     = True,
            train_mean  = train_mean,
            train_std   = train_std,
        )

        machine_results = all_results.get(machine, {})

        for arch in archs:
            key = arch + '-VAE'
            if key in machine_results:
                print(f"  [{arch}] SKIP — already done")
                continue

            print(f"\n  [{arch}-VAE] Training...")
            model = build_model(arch, in_channels)

            t0 = time.time()
            history = train_model(
                model, train_loader, device,
                epochs        = args.epochs,
                lr            = HP['lr'],
                warmup_epochs = HP['warmup_epochs'],
                max_beta      = HP['max_beta'],
                verbose       = True,
            )
            train_time = time.time() - t0

            print(f"  [{arch}] Evaluating...")
            eval_result = evaluate(
                arch, model, train_data, test_data, test_labels,
                train_mean, train_std, device,
            )

            # Extract raw arrays (prefixed with _) before building JSON-safe metrics
            raw_arrays = {k: eval_result.pop(k) for k in list(eval_result)
                          if k.startswith('_')}
            metrics = eval_result
            metrics['train_time_s'] = round(train_time, 1)

            print(f"  [{arch}] PA-F1={metrics['pa_f1']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"ROC-AUC={metrics['roc_auc']:.4f}  "
                  f"t={train_time:.0f}s")

            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir, f"{machine}_{arch.lower()}.pt")
            torch.save({'model_state_dict': model.state_dict(),
                        'metrics': metrics, 'machine': machine, 'arch': arch}, ckpt_path)

            # Save structured results via RunLogger
            logger = RunLogger(machine, key, output_dir=args.output_dir)
            logger.log_hyperparameters(**HP, **ARCH_HP.get(arch, {}))
            logger.log_training(history, train_time_s=train_time)
            logger.log_train_scores(
                raw_arrays['_train_scores'], metrics['threshold'],
                threshold_pct99=metrics['threshold'],
            )
            logger.log_test_results(
                raw_arrays['_test_scores_raw'], raw_arrays['_test_scores'],
                raw_arrays['_preds'], raw_arrays['_labels'],
                metrics['f1'], metrics['pa_f1'], metrics.get('roc_auc'),
            )
            logger.save()

            machine_results[key] = metrics

        all_results[machine] = machine_results

        # Write incrementally so a crash doesn't lose everything
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved → {results_path}")

    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print_summary(all_results)

    # Also save OmniAnomaly baselines into the JSON for paper table generation
    all_results['_omni_baselines'] = OMNI_BASELINES
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)


def print_summary(results: dict):
    archs = ['TCN-VAE', 'LSTM-VAE', 'Mamba-VAE']
    header = f"{'Machine':<14}" + "".join(f"{'PA-F1 ' + a:>16}" for a in archs) + f"{'OmniAnom':>14}"
    print(header)
    print("-" * len(header))

    for machine in ALL_MACHINES:
        if machine not in results:
            continue
        row = f"{machine:<14}"
        for a in archs:
            v = results[machine].get(a, {}).get('pa_f1', float('nan'))
            row += f"{v:>16.4f}"
        omni = OMNI_BASELINES.get(machine, {}).get('pa_f1', float('nan'))
        row += f"{omni:>14.4f}"
        print(row)

    # Averages
    print("-" * len(header))
    for a in archs:
        vals = [results[m].get(a, {}).get('pa_f1', float('nan'))
                for m in ALL_MACHINES if m in results]
        vals = [v for v in vals if not np.isnan(v)]
        print(f"  {a} mean PA-F1: {np.mean(vals):.4f}" if vals else f"  {a}: no data")
    omni_vals = [OMNI_BASELINES[m]['pa_f1'] for m in ALL_MACHINES]
    print(f"  OmniAnomaly mean PA-F1: {np.mean(omni_vals):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-architecture SMD comparison')
    parser.add_argument('--machines',   nargs='+', default=ALL_MACHINES,
                        help='Machines to train on (default: all 28)')
    parser.add_argument('--data_dir',   default='./data/ServerMachineDataset',
                        help='Path to ServerMachineDataset directory')
    parser.add_argument('--epochs',     type=int, default=HP['epochs'],
                        help='Training epochs per model per machine')
    parser.add_argument('--output_dir', default='./results',
                        help='Directory for results JSON and checkpoints')
    args = parser.parse_args()
    run(args)
