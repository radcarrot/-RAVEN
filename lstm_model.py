"""
lstm_model.py — BiLSTM VAE for multivariate time-series anomaly detection
--------------------------------------------------------------------------
Variational Autoencoder with a Bidirectional LSTM encoder backbone.
Designed as a drop-in architectural comparison against the TCN-VAE in
vae_model.py and the Mamba-VAE in mamba_model.py.

Architecture:
    Encoder:
        (B, F, W) → FeatureAttention → permute (B, W, F) → BiLSTM (2 layers)
                  → concat final forward+backward hidden → FC → (z_mu, z_log_var)

    Decoder:
        Identical to TCN-VAE decoder — ConvTranspose1d stack + FeatureAttention.
        Keeping the decoder fixed isolates the encoder's temporal backbone as
        the sole variable in the TCN vs LSTM vs Mamba comparison.

Why BiLSTM:
    Bidirectional LSTM reads the window in both directions, capturing long-range
    dependencies that dilated convolutions can miss if the anomaly signature spans
    more than the TCN's 61-step receptive field. For W=64 this is rarely limiting,
    but BiLSTM's gating mechanism offers a qualitatively different inductive bias
    (learned forget/input gates vs fixed dilated convolution patterns).

Interface:
    LSTMVAE.forward(x) → (x_mu, x_log_var, z_mu, z_log_var)  [identical to VAE]
    compute_reconstruction_errors(model, dataloader, device)    [re-exported]

Author: VAE-SMD Research Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from vae_model import FeatureAttention, Decoder, reparameterize, compute_reconstruction_errors  # noqa: F401 — re-export


# ─────────────────────────────────────────────────────────────────────────────
# 1. BiLSTM Encoder
# ─────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for windowed multivariate time-series.

    Pipeline:
        (B, F, W)
            ↓
        FeatureAttention(F, W)       — cross-sensor correlations   (B, F, W)
            ↓
        permute → (B, W, F)          — time-major for LSTM input
            ↓
        BiLSTM(F → hidden, 2 layers) — sequential temporal encoding
            ↓
        last hidden: concat [fwd, bwd] → (B, 2*hidden)
            ↓
        FC → z_mu        (B, latent_dim)
        FC → z_log_var   (B, latent_dim)

    The FeatureAttention is shared with the TCN-VAE, ensuring the comparison
    is purely over the temporal backbone (LSTM vs TCN vs Mamba).
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim:  int,
        window_size: int,
        lstm_hidden: int = 128,
        num_layers:  int = 2,
        n_heads:     int = 4,
        dropout:     float = 0.1,
        attn_d_model: int = 64,
    ):
        """
        Args:
            in_channels : Number of sensor features F (38 for SMD).
            latent_dim  : Latent space dimensionality.
            window_size : Input window length W.
            lstm_hidden : LSTM hidden size per direction. Total = 2 * lstm_hidden
                          after concatenating forward and backward outputs.
            num_layers  : Number of stacked LSTM layers.
            n_heads     : Attention heads in FeatureAttention.
                          Requirement: attn_d_model % n_heads == 0.
            dropout     : Dropout between LSTM layers.
            attn_d_model: FeatureAttention internal dimension. Fixed at 64 to match
                          TCN-VAE and Mamba-VAE for a controlled comparison.
        """
        super().__init__()
        assert attn_d_model % n_heads == 0, (
            f"attn_d_model ({attn_d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.feature_attn = FeatureAttention(
            n_features  = in_channels,
            window_size = window_size,
            d_model     = attn_d_model,
            n_heads     = n_heads,
        )

        self.lstm = nn.LSTM(
            input_size   = in_channels,
            hidden_size  = lstm_hidden,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        flat_dim = lstm_hidden * 2   # concat [fwd_final, bwd_final]
        self.fc_mu      = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, F, W)
        Returns:
            mu, log_var : each (B, latent_dim)
        """
        h = self.feature_attn(x)          # (B, F, W) — cross-sensor context
        h = h.permute(0, 2, 1)            # (B, W, F) — time-major for LSTM

        _, (hn, _) = self.lstm(h)          # hn: (num_layers*2, B, hidden)
        # hn[-2] = last forward layer, hn[-1] = last backward layer
        fwd = hn[-2]                       # (B, hidden)
        bwd = hn[-1]                       # (B, hidden)
        h   = torch.cat([fwd, bwd], dim=1) # (B, 2*hidden)

        return self.fc_mu(h), self.fc_log_var(h)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full LSTM VAE
# ─────────────────────────────────────────────────────────────────────────────

class LSTMVAE(nn.Module):
    """
    VAE with BiLSTM encoder and ConvTranspose decoder.

    Identical interface to VAE in vae_model.py — forward returns
    (x_mu, x_log_var, z_mu, z_log_var).  Uses the same Decoder class
    to keep the comparison between architectures clean.

    Checkpoint format (for load/save compatibility with train_compare.py):
        {
            'model_state_dict': ...,
            'hyperparameters': {
                'in_channels', 'latent_dim', 'window_size',
                'lstm_hidden', 'num_layers', 'n_heads'
            },
            'threshold': float,
        }
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim:  int = 32,
        window_size: int = 64,
        lstm_hidden: int = 128,
        num_layers:  int = 2,
        n_heads:     int = 4,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(
            in_channels = in_channels,
            latent_dim  = latent_dim,
            window_size = window_size,
            lstm_hidden = lstm_hidden,
            num_layers  = num_layers,
            n_heads     = n_heads,
        )
        # Decoder reused from TCN-VAE: (latent_dim → F × W)
        # tcn_hidden acts as the channel width in the ConvTranspose stack.
        # We use lstm_hidden // 2 to roughly match parameter count with TCN-VAE
        # (TCN default tcn_hidden=64, so lstm_hidden=128 → tcn_hidden_equiv=64).
        tcn_equiv = lstm_hidden // 2
        self.decoder = Decoder(
            out_channels = in_channels,
            latent_dim   = latent_dim,
            window_size  = window_size,
            tcn_hidden   = tcn_equiv,
            n_heads      = n_heads,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, F, W)
        Returns:
            x_mu, x_log_var : (B, F, W)
            z_mu, z_log_var : (B, latent_dim)
        """
        z_mu, z_log_var = self.encoder(x)
        z               = reparameterize(z_mu, z_log_var)
        x_mu, x_log_var = self.decoder(z)
        return x_mu, x_log_var, z_mu, z_log_var


# compute_reconstruction_errors is re-exported from vae_model (see import above).
