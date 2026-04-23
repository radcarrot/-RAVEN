"""
vae_model.py  (v3 — TCN + Feature Attention + Learned Variance)
----------------------------------------------------------------
Variational Autoencoder with a Temporal Convolutional Network (TCN) backbone,
cross-feature self-attention, and learned decoder variance for multivariate
time-series anomaly detection on the SMD dataset.

Architecture changes across versions:
──────────────────────────────────────
  v1 Encoder : Conv1d stack (F→32→64→128) → Pool → FC → (μ, log σ²)
  v2 Encoder : FeatureAttention → TCN stack (dilations 1,2,4,8) → Pool → FC → (μ, log σ²)
  v3 Encoder : unchanged from v2

  v1 Decoder : FC → ConvTranspose1d stack → Upsample
  v2 Decoder : FC → ConvTranspose1d stack → Upsample → FeatureAttention
  v3 Decoder : v2 decoder + separate 1×1 conv log_var_head that outputs
               per-element learned reconstruction log-variance (B, F, W)

Why the v2 additions (FeatureAttention + TCN) close the gaps in v1:

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

Why the v3 addition (learned variance):

  3. Learned decoder variance (Gaussian NLL loss)
     The decoder now outputs both x̂_μ and per-element x̂_log_var. Noisy
     features learn high variance (lower penalty for reconstruction error);
     stable features learn low variance (deviations penalised sharply).
     Anomaly scoring uses MSE on x̂_μ only — see compute_reconstruction_errors().

Loss function (v3 — Gaussian NLL reconstruction):
  L = GaussianNLL(x, x̂_μ, x̂_σ²)  +  β · KL[ N(z_μ, z_σ²) || N(0,1) ]
  NLL = ½ mean(log σ² + (x − μ)²/σ²)   [log(2π) constant dropped]
  β annealed linearly from 0 → max_beta over warmup_epochs.

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
        # Clamped to [-20, 2] in forward() for numerical stability.
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

    def __init__(self, warmup_epochs: int = 10, max_beta: float = 1.0, free_bits: float = 0.0):
        """
        Args:
            warmup_epochs : Epochs over which β ramps from 0 → max_beta.
                            Set to 0 to disable annealing (β=max_beta always).
            max_beta      : Maximum KL weight. Values < 1.0 keep reconstruction
                            as the dominant loss term, preventing posterior
                            collapse. For anomaly detection, 0.1 works well.
            free_bits     : Minimum KL per latent dimension (nats). Dimensions
                            whose per-batch-averaged KL falls below this floor
                            are not penalised — gradient to the encoder is zero
                            for those dims. Prevents KL collapse while still
                            regularising active dimensions. Typical: 0.1.
                            Set to 0.0 to disable (default, original behaviour).
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_beta      = max_beta
        self.free_bits     = free_bits

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

        # KL divergence (closed form) — positive per (batch, dim)
        kl_per_dim = -0.5 * (1 + z_log_var - z_mu.pow(2) - z_log_var.exp())  # (B, latent_dim)

        if self.free_bits > 0.0:
            # Average over batch per dim, clamp to free_bits floor, then sum.
            # Dims below the floor contribute a constant — gradient to encoder is
            # zero for those dims, forcing the decoder to use them or stay silent.
            kl_loss = torch.clamp(kl_per_dim.mean(dim=0), min=self.free_bits).sum()
        else:
            kl_loss = kl_per_dim.mean(dim=0).sum()

        beta       = self.get_beta(current_epoch)
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# 8. Inference: Per-Window Reconstruction Error  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_errors(
    model:      nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device:     torch.device,
) -> torch.Tensor:
    """
    Pass all windows through the frozen model and return per-window MSE.

    Works with any model that follows the 4-tensor forward API:
        (x_mu, x_log_var, z_mu, z_log_var) = model(batch)

    Scores on the reconstruction mean only (MSE), ignoring learned variance.
    This prevents the model from hiding anomalies by predicting high variance.

    Args:
        model      : Trained model — VAE, LSTMVAE, or MambaVAE.
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
