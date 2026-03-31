"""
vae_model.py  (v2 — TCN + Feature Attention)
---------------------------------------------
Variational Autoencoder with a Temporal Convolutional Network (TCN) backbone
and cross-feature self-attention for multivariate time-series anomaly detection
on the SMD dataset.

Architecture changes from v1 (plain 1D-CNN):
─────────────────────────────────────────────
  v1 Encoder : Conv1d stack (F→32→64→128) → Pool → FC → (μ, log σ²)
  v2 Encoder : FeatureAttention → TCN stack (dilations 1,2,4,8) → Pool → FC → (μ, log σ²)

  v1 Decoder : FC → ConvTranspose1d stack → Upsample
  v2 Decoder : FC → ConvTranspose1d stack → Upsample → FeatureAttention

Why the two additions close the structural gaps in v1:

  1. FeatureAttention (cross-sensor correlation)
     Server anomalies are rarely isolated to one metric. A CPU spike co-occurs
     with memory pressure, I/O wait, and network saturation. The v1 Conv1d
     treated the 38 sensor channels as independent filters — it could not learn
     these inter-sensor relationships. FeatureAttention treats each sensor's
     W-step temporal profile as a token and applies multi-head self-attention
     across all 38 tokens, letting every sensor attend to every other sensor
     before temporal encoding begins.

  2. Dilated TCN (full-window receptive field)
     With kernel=3 and strides 1/2/2, the v1 encoder's effective receptive
     field was ≈12 timesteps at the deepest layer. With dilations [1,2,4,8]
     and kernel=3, the TCN's receptive field is:
         2 × (3−1) × (1+2+4+8) + 1 = 61 timesteps
     covering the entire 64-step window in 4 layers without sacrificing
     parallelism or adding sequential bottlenecks.

Loss function and training are unchanged:
  L = MSE(x, x̂)  +  β · KL[ N(μ, σ²) || N(0,1) ]
  β annealed linearly from 0 → 1 over warmup_epochs.

Author: VAE-SMD Mini-Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cross-Feature Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class FeatureAttention(nn.Module):
    """
    Cross-feature self-attention for multivariate time-series windows.

    Intuition:
        For a window of shape (F, W), treat each of the F sensor channels
        as a "token" whose embedding is its W-step temporal profile projected
        to d_model dimensions. Multi-head self-attention then allows every
        sensor to attend to every other sensor — capturing cross-sensor
        dependencies that Conv1d ignores.

    Input / output:  (B, F, W)  — shape preserved via residual connection.

    Attention complexity: O(F²) per batch — with F=38, this is a 38×38
    attention matrix per head, negligible compared to O(W²) temporal attention.

    Why apply this BEFORE the TCN (in the encoder):
        The TCN receives context-enriched features where each sensor already
        encodes information from its peers. Temporal convolutions then extract
        patterns from correlated, semantically-aware signals rather than raw
        independent channels.

    Why apply this AFTER the ConvTranspose stack (in the decoder):
        The decoder must reconstruct the joint distribution of all 38 sensors.
        Applying feature attention after spatial upsampling lets the decoder
        re-establish inter-sensor consistency in the final reconstruction.
    """

    def __init__(
        self,
        n_features:  int,
        window_size: int,
        d_model:     int = 64,
        n_heads:     int = 4,
        dropout:     float = 0.1
    ):
        """
        Args:
            n_features  : Number of sensor channels F.
            window_size : Input window length W (used for temporal projection).
            d_model     : Internal attention embedding dimension.
                          Must be divisible by n_heads.
            n_heads     : Number of attention heads.
            dropout     : Attention dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Project each feature's W-step temporal profile → d_model embedding.
        # This decouples the attention dimension from window_size, allowing
        # the model to be retrained with different window sizes without
        # changing the attention head count or d_model.
        self.proj_in  = nn.Linear(window_size, d_model)

        # Self-attention: F sensor tokens attend to each other.
        self.attn     = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True    # expects (B, L, E), not PyTorch's old (L, B, E)
        )
        self.norm     = nn.LayerNorm(d_model)

        # Project attended embeddings back to W-step temporal profiles.
        self.proj_out = nn.Linear(d_model, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor, shape (B, F, W).
        Returns:
            Tensor, shape (B, F, W) — cross-sensor context added via residual.
        """
        h = self.proj_in(x)          # (B, F, W) → (B, F, d_model)
        h, _ = self.attn(h, h, h)    # self-attention: F tokens attend to each other
        h = self.norm(h)             # LayerNorm over d_model dimension
        h = self.proj_out(h)         # (B, F, d_model) → (B, F, W)
        return x + h                 # residual: preserve original signal + add context


# ─────────────────────────────────────────────────────────────────────────────
# 2. TCN Residual Block
# ─────────────────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """
    One residual block of a Temporal Convolutional Network (TCN).

    Key properties:
      - Dilated Conv1d expands the receptive field exponentially with depth.
      - With kernel_size=3 and padding=dilation, output length = input length
        (no temporal downsampling inside the TCN stack).
      - Residual connection with 1×1 conv when in_channels ≠ out_channels.

    Receptive field per dilation level (kernel=3):
        dilation=1  →  adds 2 timesteps
        dilation=2  →  adds 4 timesteps
        dilation=4  →  adds 8 timesteps
        dilation=8  →  adds 16 timesteps
        ───────────────────────────────
        4 blocks total receptive field:  2×(3−1)×(1+2+4+8) + 1 = 61 timesteps
        Covers the full 64-step window in a single TCN stack.

    Why not causal (left-padded) convolutions:
        Causal masking is required for *streaming* prediction where future
        timesteps are unavailable. Here the entire window is available at
        inference time (batch mode), so symmetric padding is appropriate and
        gives the model access to both past and future context within the window.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        dilation:     int = 1
    ):
        super().__init__()
        # padding = dilation keeps: output_len = input_len (for kernel_size=3)
        padding = dilation

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        # Residual projection: 1×1 conv when channel count changes, else identity
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    TCN Encoder with cross-feature attention.

    Full pipeline:
        Input (B, F, W)
            ↓
        FeatureAttention          — cross-sensor correlations (B, F, W)
            ↓
        TCN Block (dilation=1)    — short-range local patterns  (B, H, W)
        TCN Block (dilation=2)    — mid-range patterns          (B, H, W)
        TCN Block (dilation=4)    — longer-range patterns       (B, H, W)
        TCN Block (dilation=8)    — full-window context         (B, 2H, W)
            ↓
        AdaptiveAvgPool1d(8)      — fixed temporal summary      (B, 2H, 8)
            ↓
        Flatten                                                  (B, 2H×8)
            ↓
        FC → μ       (B, latent_dim)
        FC → log σ²  (B, latent_dim)

    H = tcn_hidden (default 64), so flat_dim = 128 × 8 = 1024,
    matching v1's flat_dim for a fair parameter-count comparison.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim:  int,
        window_size: int,
        tcn_hidden:  int = 64,
        n_heads:     int = 4,
    ):
        """
        Args:
            in_channels : Number of sensor features F (38 for SMD).
            latent_dim  : Latent space dimensionality.
            window_size : Input window length W.
            tcn_hidden  : Base channel width for the TCN stack.
                          The last TCN block doubles this to tcn_hidden*2.
            n_heads     : Attention heads in FeatureAttention.
                          Requirement: tcn_hidden % n_heads == 0.
        """
        super().__init__()
        self._tcn_hidden = tcn_hidden

        self.feature_attn = FeatureAttention(
            n_features  = in_channels,
            window_size = window_size,
            d_model     = tcn_hidden,
            n_heads     = n_heads,
        )

        self.tcn = nn.Sequential(
            TemporalBlock(in_channels,   tcn_hidden,     dilation=1),
            TemporalBlock(tcn_hidden,    tcn_hidden,     dilation=2),
            TemporalBlock(tcn_hidden,    tcn_hidden,     dilation=4),
            TemporalBlock(tcn_hidden,    tcn_hidden * 2, dilation=8),
        )

        self.pool    = nn.AdaptiveAvgPool1d(output_size=8)
        flat_dim     = (tcn_hidden * 2) * 8   # 1024 with default tcn_hidden=64

        self.fc_mu      = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : Tensor, shape (B, F, W).
        Returns:
            mu      : Tensor, shape (B, latent_dim) — posterior mean.
            log_var : Tensor, shape (B, latent_dim) — posterior log-variance.
        """
        h = self.feature_attn(x)    # (B, F, W)  — cross-sensor context
        h = self.tcn(h)             # (B, 2H, W) — temporal encoding
        h = self.pool(h)            # (B, 2H, 8) — fixed-size temporal summary
        h = h.flatten(start_dim=1)  # (B, 2H×8)

        return self.fc_mu(h), self.fc_log_var(h)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reparameterization Trick  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Sample a latent vector z using the reparameterization trick.

    Rewrites z ~ N(μ, σ²) as a deterministic function of a fixed noise ε:
        z = μ + ε · σ,    ε ~ N(0, I)

    This allows gradients to flow through μ and σ during backpropagation,
    since ε is treated as a non-learned constant rather than a parameter.

    Args:
        mu      : Tensor, shape (B, latent_dim) — posterior mean.
        log_var : Tensor, shape (B, latent_dim) — posterior log-variance.
    Returns:
        z : Tensor, shape (B, latent_dim).
    """
    std = torch.exp(0.5 * log_var)   # σ = exp(0.5 · log σ²)
    eps = torch.randn_like(std)      # ε ~ N(0, I), same device and dtype
    return mu + eps * std


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    TCN-style Decoder with cross-feature attention.

    Full pipeline:
        z (B, latent_dim)
            ↓
        FC(latent_dim → 2H×8)     — project latent vector to spatial seed
            ↓
        Reshape (B, 2H, 8)
            ↓
        ConvTranspose1d(2H → H,  k=4, s=2)  + BN + GELU  →  (B, H, ~16)
        ConvTranspose1d(H  → H/2, k=4, s=2) + BN + GELU  →  (B, H/2, ~32)
        ConvTranspose1d(H/2 → F,  k=3, s=1)               →  (B, F, ~32)
            ↓
        Upsample(size=W, mode='linear')     — exact output size (B, F, W)
            ↓
        FeatureAttention                    — re-establish inter-sensor consistency

    Why FeatureAttention at the end of the decoder:
        The ConvTranspose stack rebuilds temporal structure but operates on
        channels independently. The final FeatureAttention allows the decoder
        to enforce cross-sensor consistency in the reconstruction — e.g. ensuring
        that a reconstructed CPU spike co-occurs with appropriate memory/I/O values.
        This makes the reconstructed x̂ "physically plausible" under the learned
        sensor covariance structure, sharpening the anomaly score contrast.
    """

    def __init__(
        self,
        out_channels: int,
        latent_dim:   int,
        window_size:  int,
        tcn_hidden:   int = 64,
        n_heads:      int = 4,
    ):
        """
        Args:
            out_channels : Number of sensor features F (must match Encoder.in_channels).
            latent_dim   : Latent space dimensionality.
            window_size  : Target output window length W.
            tcn_hidden   : Must match the value used in Encoder.
            n_heads      : Attention heads in FeatureAttention.
        """
        super().__init__()
        self.window_size = window_size
        self._tcn_hidden = tcn_hidden

        flat_dim = (tcn_hidden * 2) * 8   # must match Encoder's flat_dim

        self.fc = nn.Linear(latent_dim, flat_dim)

        self.deconv_stack = nn.Sequential(
            # ── Upsample ×2: 8 → ~16 ─────────────────────────────────────────
            nn.ConvTranspose1d(tcn_hidden * 2, tcn_hidden,     kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(tcn_hidden),
            nn.GELU(),

            # ── Upsample ×2: ~16 → ~32 ───────────────────────────────────────
            nn.ConvTranspose1d(tcn_hidden,     tcn_hidden // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(tcn_hidden // 2),
            nn.GELU(),

            # ── Final feature projection → F output channels ──────────────────
            # No BatchNorm / activation: target is real-valued Z-scored data ∈ ℝ.
            # Any bounded activation (Tanh/Sigmoid) would clip reconstruction values.
            nn.ConvTranspose1d(tcn_hidden // 2, out_channels,  kernel_size=3, stride=1, padding=1),
        )

        # Learnable-free resize to enforce exact (B, F, window_size) output.
        self.upsample = nn.Upsample(size=window_size, mode='linear', align_corners=False)

        # Cross-feature attention: refine inter-sensor consistency in reconstruction.
        self.feature_attn = FeatureAttention(
            n_features  = out_channels,
            window_size = window_size,
            d_model     = tcn_hidden,
            n_heads     = n_heads,
        )

        # Learned per-element log-variance head: (B, F, W) → (B, F, W).
        # Separate from the mean path — variance doesn't need cross-feature attention.
        # Clamped to [-4, 2] in forward() for numerical stability.
        self.log_var_head = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z : Tensor, shape (B, latent_dim).
        Returns:
            x_mu      : Tensor, shape (B, F, W) — reconstruction mean.
            x_log_var : Tensor, shape (B, F, W) — reconstruction log-variance.
        """
        h = self.fc(z)                                   # (B, flat_dim)
        h = h.view(h.size(0), self._tcn_hidden * 2, 8)  # (B, 2H, 8)
        h = self.deconv_stack(h)                         # (B, F, ~W)
        h = self.upsample(h)                             # (B, F, W) — exact size
        x_mu      = self.feature_attn(h)                 # (B, F, W) — cross-sensor refinement
        x_log_var = self.log_var_head(h).clamp(-20, 2)   # (B, F, W) — learned uncertainty
        return x_mu, x_log_var


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full VAE Model
# ─────────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """
    Variational Autoencoder: Encoder + Reparameterization + Decoder.

    Trained on *normal* data only. At inference, anomalous windows produce
    high reconstruction error (MSE) because the latent space encodes only
    normal operational patterns; anomalous sensor behaviour maps to regions
    of latent space the decoder has never been trained to reconstruct.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim:  int = 32,
        window_size: int = 64,
        tcn_hidden:  int = 64,
        n_heads:     int = 4,
    ):
        """
        Args:
            in_channels : Number of sensor/feature channels F.
            latent_dim  : Latent space dimensionality.
                          Too small → underfitting; too large → VAE memorises,
                          losing anomaly sensitivity. 32 is a strong baseline.
            window_size : Input/output window length W.
            tcn_hidden  : TCN base channel width (doubles at last TCN block).
            n_heads     : Attention heads. Requirement: tcn_hidden % n_heads == 0.
        """
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, window_size, tcn_hidden, n_heads)
        self.decoder = Decoder(in_channels, latent_dim, window_size, tcn_hidden, n_heads)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample → decode.

        Args:
            x : Tensor, shape (B, F, W).
        Returns:
            x_mu        : Tensor, shape (B, F, W) — reconstruction mean.
            x_log_var   : Tensor, shape (B, F, W) — reconstruction log-variance.
            z_mu        : Tensor, shape (B, latent_dim) — posterior mean.
            z_log_var   : Tensor, shape (B, latent_dim) — posterior log-variance.
        """
        z_mu, z_log_var = self.encoder(x)
        z               = reparameterize(z_mu, z_log_var)
        x_mu, x_log_var = self.decoder(z)
        return x_mu, x_log_var, z_mu, z_log_var


# ─────────────────────────────────────────────────────────────────────────────
# 7. ELBO Loss with KL Annealing  (v3: Gaussian NLL reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

class ELBOLoss(nn.Module):
    """
    Evidence Lower BOund (ELBO) loss for the VAE with learned decoder variance.

    Minimise:
        L = GaussianNLL(x, x̂_μ, x̂_σ²)  +  β · KL[ N(z_μ, z_σ²) || N(0,1) ]

    Gaussian NLL (reconstruction loss):
        −log p(x|z) = ½ Σ (log σ² + (x − μ)²/σ²)
        The log(2π) constant is dropped (does not affect optimisation).

    This replaces the fixed-variance MSE loss. The decoder now outputs both
    a mean (x̂_μ) and a learned log-variance (x̂_log_var) per element. Benefits:
      - Noisy features learn high variance → naturally downweighted in the loss.
      - Stable features learn low variance → deviations penalised sharply.
      - Anomaly score becomes a true negative log-likelihood.

    KL Divergence (closed form):
        KL = −½ Σ_j (1 + log σ²_j − μ²_j − σ²_j)

    KL Annealing (β warm-up):
        β ramps linearly from 0 → max_beta over warmup_epochs.
    """

    def __init__(self, warmup_epochs: int = 10, max_beta: float = 1.0):
        """
        Args:
            warmup_epochs : Epochs over which β ramps from 0 → max_beta.
                            Set to 0 to disable annealing (β=max_beta always).
            max_beta      : Maximum KL weight. Values < 1.0 keep reconstruction
                            as the dominant loss term, preventing posterior
                            collapse. For anomaly detection, 0.1 works well.
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_beta      = max_beta

    def get_beta(self, current_epoch: int) -> float:
        """Compute annealed β for the current epoch (0-indexed)."""
        if self.warmup_epochs == 0:
            return self.max_beta
        return min(self.max_beta, (current_epoch / max(1, self.warmup_epochs)) * self.max_beta)

    def forward(
        self,
        x:             torch.Tensor,
        x_mu:          torch.Tensor,
        x_log_var:     torch.Tensor,
        z_mu:          torch.Tensor,
        z_log_var:     torch.Tensor,
        current_epoch: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x             : Original input, shape (B, F, W).
            x_mu          : Reconstruction mean, shape (B, F, W).
            x_log_var     : Reconstruction log-variance, shape (B, F, W).
            z_mu          : Posterior mean, shape (B, latent_dim).
            z_log_var     : Posterior log-variance, shape (B, latent_dim).
            current_epoch : Current training epoch (0-based).
        Returns:
            total_loss : Scalar — combined ELBO (minimised during training).
            recon_loss : Scalar — Gaussian NLL reconstruction component.
            kl_loss    : Scalar — KL divergence component.
        """
        # Gaussian NLL: 0.5 * mean(log_var + (x - mu)^2 / exp(log_var))
        # Averaged over B, F, W.  Dropping the log(2π) constant.
        recon_loss = 0.5 * torch.mean(
            x_log_var + (x - x_mu).pow(2) / torch.exp(x_log_var)
        )

        # KL divergence (closed form, mean over batch, sum over latent dims)
        kl_per_dim = 1 + z_log_var - z_mu.pow(2) - z_log_var.exp()  # (B, latent_dim)
        kl_loss    = -0.5 * torch.mean(kl_per_dim.sum(dim=1))       # scalar

        beta       = self.get_beta(current_epoch)
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# 8. Inference: Per-Window Reconstruction Error  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_errors(
    model:      VAE,
    dataloader: torch.utils.data.DataLoader,
    device:     torch.device
) -> torch.Tensor:
    """
    Pass all windows through the frozen trained VAE and return per-window MSE.

    Legacy scoring function — uses MSE on the reconstruction mean only,
    ignoring the learned decoder variance. Kept for backward compatibility
    and for comparing MSE vs NLL scoring. Prefer compute_anomaly_scores().

    Args:
        model      : Trained VAE (set to eval mode internally).
        dataloader : DataLoader with shuffle=False (preserves temporal order).
        device     : torch.device — 'cuda' or 'cpu'.
    Returns:
        errors : 1D Tensor, shape (N,) — one MSE scalar per window.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in dataloader:
            batch              = batch.to(device)
            x_mu, _, _, _      = model(batch)

            # Per-window MSE: mean over (F, W) dimensions, keep batch dim
            mse = F.mse_loss(x_mu, batch, reduction='none')   # (B, F, W)
            mse = mse.mean(dim=[1, 2])                         # (B,)

            errors.append(mse.cpu())

    return torch.cat(errors, dim=0)   # (N,)


def compute_anomaly_scores(
    model:      VAE,
    dataloader: torch.utils.data.DataLoader,
    device:     torch.device,
    n_samples:  int = 20,
    topk:       int = 3,
    kl_weight:  float = 0.1,
) -> torch.Tensor:
    """
    Monte Carlo Gaussian NLL anomaly score with feature-wise top-k aggregation
    and per-window KL divergence.

    For each window:
      1. Encode to posterior q(z|x) = N(z_mu, z_sigma²).
      2. Sample z n_samples times; for each sample, decode to (x_mu, x_log_var).
      3. Compute per-element Gaussian NLL: 0.5 * (log_var + (x - mu)² / exp(log_var)).
      4. Average NLL over the temporal dimension W → per-feature score (B, F).
      5. Take the top-k feature scores and average them → per-window NLL score (B,).
      6. Compute per-window KL divergence from the encoder posterior to the
         standard normal prior: KL = -0.5 * sum(1 + log_var - mu² - exp(log_var)).
      7. Combine: score = nll_topk + kl_weight * kl_per_window.
      8. Average across all MC samples.

    Why add KL divergence to the anomaly score:
        Anomalies may map to unusual regions of latent space (far from the
        N(0,I) prior) even when the decoder reconstructs them reasonably well.
        Adding a weighted KL term catches these "latent-space outliers".
        kl_weight defaults to 0.1 so reconstruction still dominates.

    Why top-k instead of mean over features:
        Real anomalies often affect only a few sensors (e.g. CPU + memory but not
        all 38). Averaging across all features dilutes the signal ~F/k times.
        top-k focuses on the most anomalous channels.

    Args:
        model      : Trained VAE (set to eval mode internally).
        dataloader : DataLoader with shuffle=False (preserves temporal order).
        device     : torch.device — 'cuda' or 'cpu'.
        n_samples  : Number of latent samples per window. Default 20.
        topk       : Number of top per-feature NLL scores to average. Default 3.
        kl_weight  : Weight for the per-window KL divergence term. Default 0.1.
                     Set to 0.0 to disable KL scoring (pure NLL).
    Returns:
        scores : 1D Tensor, shape (N,) — one score per window (higher = more anomalous).
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            batch       = batch.to(device)                     # (B, F, W)
            z_mu, z_lv  = model.encoder(batch)                 # posterior params

            score_sum = torch.zeros(batch.size(0), device=device)
            for _ in range(n_samples):
                z                   = reparameterize(z_mu, z_lv)
                x_mu, x_log_var     = model.decoder(z)

                # Per-element Gaussian NLL (dropping constant log(2π))
                nll = 0.5 * (x_log_var + (batch - x_mu).pow(2) / torch.exp(x_log_var))

                # Per-feature NLL: average over temporal dim W → (B, F)
                nll_per_feature = nll.mean(dim=2)

                # Top-k feature aggregation → (B,)
                k = min(topk, nll_per_feature.size(1))
                nll_topk = nll_per_feature.topk(k, dim=1).values.mean(dim=1)

                # Per-window KL divergence: encoder posterior vs N(0,I) prior
                # kl shape: (B,) — one scalar per window
                kl = -0.5 * (1 + z_lv - z_mu.pow(2) - z_lv.exp()).sum(dim=1)

                # Combined score: reconstruction NLL + weighted KL
                score = nll_topk + kl_weight * kl

                score_sum += score

            scores.append((score_sum / n_samples).cpu())

    return torch.cat(scores, dim=0)   # (N,)
