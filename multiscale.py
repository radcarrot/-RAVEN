"""
multiscale.py
-------------
Multi-scale ensemble for VAE anomaly detection.
Trains/loads models at multiple window sizes and combines their scores.

Anomalies in server telemetry occur at different temporal scales:
  - Short bursts (e.g. 32 timesteps): brief CPU spikes, network packet storms.
  - Medium patterns (e.g. 64 timesteps): sustained memory leaks, I/O saturation.
  - Slow drifts (e.g. 128 timesteps): gradual resource exhaustion, thermal throttling.

A single fixed window size may miss anomalies at other scales. This module trains
separate VAE models at multiple window sizes and combines their normalised anomaly
scores into a single per-timestep ensemble score.

Usage:
    from multiscale import train_multiscale_models, multiscale_ensemble_scores

    models = train_multiscale_models(train_data, device)
    scores = multiscale_ensemble_scores(test_data, models, device)
"""

import numpy as np
import torch
from vae_model import VAE, ELBOLoss, compute_anomaly_scores, reparameterize
from preprocess import build_dataloader_from_array


def multiscale_ensemble_scores(
    test_data: np.ndarray,
    models: dict,
    device: torch.device,
    window_sizes: list = None,
    n_samples: int = 20,
    topk: int = 3,
    kl_weight: float = 0.1,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Combine anomaly scores from multiple VAE models trained at different window sizes.

    Each model produces scores at its own window size. Scores are:
      1. Computed per model with compute_anomaly_scores().
      2. Z-score normalized (so different scales are comparable).
      3. Mapped to a common timestep axis (aligned to the last timestep of each window).
      4. Averaged across all models at each timestep.

    Args:
        test_data    : np.ndarray, shape (T, F) — raw test data.
        models       : dict mapping window_size (int) -> trained VAE model.
                       E.g., {32: model_32, 64: model_64, 128: model_128}.
        device       : torch.device.
        window_sizes : List of window sizes to use (default: keys of models dict).
        n_samples    : MC samples per window for scoring.
        topk         : Top-k features for aggregation.
        kl_weight    : KL weight in combined score.
        batch_size   : Batch size for inference.
    Returns:
        combined_scores : np.ndarray, shape (T,) — per-timestep ensemble score.
                          Timesteps without coverage from any model are set to 0.
    """
    if window_sizes is None:
        window_sizes = sorted(models.keys())

    T = test_data.shape[0]
    score_stack = []  # list of (T,) arrays, one per window size

    for ws in window_sizes:
        model = models[ws]

        # Build test loader for this window size
        loader = build_dataloader_from_array(
            test_data, window_size=ws, stride=1,
            batch_size=batch_size, shuffle=False
        )

        # Compute raw scores
        raw_scores = compute_anomaly_scores(
            model, loader, device,
            n_samples=n_samples, topk=topk, kl_weight=kl_weight
        ).numpy()

        # Z-score normalize so different window sizes are comparable
        mu = raw_scores.mean()
        sigma = raw_scores.std() + 1e-8
        normed = (raw_scores - mu) / sigma

        # Map to per-timestep: assign each window's score to its last timestep
        padded = np.full(T, np.nan, dtype=np.float32)
        n_windows = len(normed)
        padded[ws - 1 : ws - 1 + n_windows] = normed

        score_stack.append(padded)
        print(f'  Window size {ws}: {n_windows} windows scored')

    # Average across scales (ignoring NaN for timesteps with partial coverage)
    stacked = np.stack(score_stack, axis=0)  # (n_scales, T)
    with np.errstate(all='ignore'):
        combined = np.nanmean(stacked, axis=0)
    # Replace any remaining NaN (no coverage at all) with 0
    combined = np.nan_to_num(combined, nan=0.0)

    return combined


def train_multiscale_models(
    train_data: np.ndarray,
    device: torch.device,
    window_sizes: list = [32, 64, 128],
    latent_dim: int = 32,
    tcn_hidden: int = 64,
    n_heads: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    warmup_epochs: int = 25,
    max_beta: float = 0.1,
    batch_size: int = 256,
) -> dict:
    """
    Train separate VAE models for each window size.

    Args:
        train_data   : np.ndarray, shape (T, F) — normal training data.
        device       : torch.device.
        window_sizes : List of window sizes to train.
        latent_dim   : Latent space dimensionality.
        tcn_hidden   : TCN base channel width. Must satisfy tcn_hidden % n_heads == 0
                       and be even (decoder uses tcn_hidden // 2).
        n_heads      : Attention heads in FeatureAttention.
        epochs       : Number of training epochs per model.
        lr           : Learning rate for Adam optimizer.
        warmup_epochs: Epochs over which KL beta ramps from 0 to max_beta.
        max_beta     : Maximum KL weight in ELBO loss.
        batch_size   : Mini-batch size for training.
    Returns:
        models : dict mapping window_size -> trained VAE model (on device, eval mode).
    """
    in_channels = train_data.shape[1]
    models = {}

    for ws in window_sizes:
        print(f'\n{"="*60}')
        print(f'Training VAE with WINDOW_SIZE={ws}')
        print(f'{"="*60}')

        model = VAE(
            in_channels=in_channels, latent_dim=latent_dim,
            window_size=ws, tcn_hidden=tcn_hidden, n_heads=n_heads,
        ).to(device)

        criterion = ELBOLoss(warmup_epochs=warmup_epochs, max_beta=max_beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        loader = build_dataloader_from_array(
            train_data, window_size=ws, stride=1,
            batch_size=batch_size, shuffle=True
        )

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for batch in loader:
                batch = batch.to(device)
                x_mu, x_log_var, z_mu, z_log_var = model(batch)
                total_loss, _, _ = criterion(
                    batch, x_mu, x_log_var, z_mu, z_log_var, current_epoch=epoch
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += total_loss.item()

            avg_loss = running_loss / len(loader)
            scheduler.step(avg_loss)

            if epoch % 25 == 0 or epoch == 1:
                print(f'  Epoch [{epoch:3d}/{epochs}] Loss={avg_loss:.4f}')

        model.eval()
        models[ws] = model
        print(f'  Done. Model for ws={ws} ready.')

    return models
