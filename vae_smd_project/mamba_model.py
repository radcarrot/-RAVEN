"""
mamba_model.py — Mamba-2 (SSD) VAE for multivariate time-series anomaly detection
------------------------------------------------------------------------------------
Variational Autoencoder with a Mamba-2 / State Space Duality (SSD) encoder.

Designed as a drop-in architectural comparison against:
  - TCN-VAE   (vae_model.py)
  - BiLSTM-VAE (lstm_model.py)

This module implements a pure-PyTorch Mamba-2 approximation. It does NOT require
the `mamba-ssm` CUDA extension — the selective scan runs sequentially on any device.
For production-scale training, install `mamba-ssm` and replace `selective_scan_seq`
with the optimised parallel associative scan.

Background — Mamba and Mamba-2:
    Mamba (Gu & Dao, 2023, arXiv:2312.00752) introduced Selective State Space Models
    (S6): unlike S4 where A, B, C are fixed or time-invariant, Mamba makes B, C, and
    a time-step Δ input-dependent, giving the model a selective memory mechanism.

    Mamba-2 (Dao & Gu, 2024, arXiv:2405.21060) introduced State Space Duality (SSD):
    a structured matrix formulation showing equivalence between SSMs and a form of
    linear attention, enabling more efficient multi-head parallel computation.
    Key SSD simplification: A is scalar per head (not per-dimension), yielding
    head-wise exponential decay controlled by a learned scalar α.

This implementation follows Mamba-2 SSD with:
    - Head-specific scalar A (α decay), input-dependent B, C, Δ
    - Multi-head grouped structure: n_heads groups of d_head = d_model // n_heads
    - Zero-order hold (ZOH) discretisation: Ā = exp(-softplus(α) * Δ), B̄ = Δ * B
    - Sequential scan (parallel scan not needed for W=64 windows)
    - Residual MambaBlock: LayerNorm → expand → SSM → contract → dropout → residual

Architecture:
    Encoder:
        (B, F, W) → FeatureAttention → permute (B, W, F)
                  → MambaBlock × n_layers                 (B, W, F)
                  → mean-pool over W → (B, F)
                  → FC → (z_mu, z_log_var)

    Decoder:
        Identical to TCN-VAE Decoder (ConvTranspose1d + FeatureAttention).

Author: VAE-SMD Research Project
References:
    Gu, A. & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective
        State Spaces. arXiv:2312.00752.
    Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and
        Efficient Algorithms Through Structured State Space Duality. arXiv:2405.21060.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from vae_model import FeatureAttention, Decoder, reparameterize, ELBOLoss  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# 1. Selective State Space (S6 core — Mamba-1 style, Mamba-2 head structure)
# ─────────────────────────────────────────────────────────────────────────────

def selective_scan_seq(
    u:     torch.Tensor,   # (B, L, d_inner)
    delta: torch.Tensor,   # (B, L, d_inner) — discretisation step Δ
    A:     torch.Tensor,   # (d_inner, d_state) — log-A (learned, fixed per channel)
    B:     torch.Tensor,   # (B, L, d_state)    — input-dependent B
    C:     torch.Tensor,   # (B, L, d_state)    — input-dependent C
    D:     torch.Tensor,   # (d_inner,)          — skip connection
) -> torch.Tensor:
    """
    Sequential selective scan — O(L) per sequence, O(B*L*d_inner*d_state) total.

    Implements the ZOH-discretised SSM recurrence:
        Ā_t  = exp(Δ_t * A)          (element-wise; A stored as negative log)
        B̄_t  = Δ_t * B_t             (simplified ZOH — valid when Δ is small)
        h_t  = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t
        y_t  = C_t · h_t  +  D ⊙ u_t

    Args:
        u     : input sequence, (B, L, d_inner)
        delta : input-dependent time step, (B, L, d_inner)
        A     : log of decay rates, shape (d_inner, d_state) — stored as -log|A|
                so actual A < 0 (stable); Ā = exp(Δ * A) ∈ (0, 1)
        B     : input projection (B, L, d_state)
        C     : output projection (B, L, d_state)
        D     : skip connection weight (d_inner,)

    Returns:
        y : output tensor (B, L, d_inner)
    """
    B_sz, L, d_inner = u.shape
    d_state = A.shape[1]

    # Discretise: Ā = exp(Δ · A), B̄ = Δ · B
    # delta: (B, L, d_inner), A: (d_inner, d_state)
    # dA shape: (B, L, d_inner, d_state)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))   # (B, L, d_inner, d_state)
    dB = delta.unsqueeze(-1) * B.unsqueeze(2)                             # (B, L, d_inner, d_state)

    # Sequential scan over time dimension L
    h = torch.zeros(B_sz, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        # h: (B, d_inner, d_state)
        # dA[:, t]: (B, d_inner, d_state)
        # dB[:, t]: (B, d_inner, d_state)
        # u[:, t]:  (B, d_inner)
        h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)

        # y_t = sum over d_state: h * C_t → (B, d_inner)
        # C[:, t]: (B, d_state)
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)   # (B, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)            # (B, L, d_inner)
    y = y + u * D.unsqueeze(0).unsqueeze(0)  # skip connection
    return y


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mamba-2 SSD Block
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Mamba-2 residual block with selective state space layer.

    Structure (following Mamba-2 SSD paper):
        x → LayerNorm
          → Linear expand (d_model → d_inner = expand * d_model)
          → [branch 1] gating with SiLU
          → [branch 2] selective SSM (S6)
          → element-wise product (gated output)
          → Linear contract (d_inner → d_model)
          → Dropout
          + residual

    Input-dependent projections (all linear, applied on the expanded sequence):
        x_proj: (d_inner) → (dt_rank + 2*d_state)
            → splits into [Δ_raw (dt_rank), B (d_state), C (d_state)]
        dt_proj: (dt_rank) → (d_inner)  — promotes rank-dt to full width

    Learnable fixed parameters:
        A_log : (d_inner, d_state) — log decay rates (negative; Ā ∈ (0,1))
        D     : (d_inner,)          — skip connection
    """

    def __init__(
        self,
        d_model:  int,
        d_state:  int = 16,
        d_conv:   int = 4,
        expand:   int = 2,
        dt_rank:  int = None,
        dropout:  float = 0.1,
    ):
        """
        Args:
            d_model : Input/output feature dimension.
            d_state : SSM state size N (latent state per channel).
            d_conv  : Depthwise conv kernel size (short-range local context).
            expand  : Inner expansion factor (d_inner = expand * d_model).
            dt_rank : Rank of the Δ projection. Defaults to ceil(d_model / 16).
            dropout : Dropout on the output projection.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        if dt_rank is None:
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

        # Input projection: d_model → 2 * d_inner (one for SSM input, one for gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short depthwise conv for local context (Mamba-1 uses this before SSM)
        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,
            groups       = self.d_inner,
            bias         = True,
        )

        # x_proj: maps inner dim to (dt_rank + 2*d_state) for B, C, Δ
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)

        # dt_proj: promote dt_rank → d_inner and set dt (log) bias
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialise dt_proj bias so softplus(bias) ≈ Δ_init ∈ [0.001, 0.1]
        dt_init_floor = 1e-4
        dt_log_max    = math.log(0.1)
        dt_log_min    = math.log(dt_init_floor)
        nn.init.uniform_(self.dt_proj.bias, dt_log_min, dt_log_max)

        # A: stable decay; initialised as A_log = log(n) for n in 1..d_state
        A = torch.arange(1, d_state + 1, dtype=torch.float32)            # (d_state,)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()               # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))                           # stored as log(A)
        self.A_log._no_weight_decay = True

        # D: skip connection (one per inner channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, L, d_model)
        Returns:
            y : (B, L, d_model) — residual added inside
        """
        residual = x
        x = self.norm(x)                                   # pre-norm

        # Project and split into SSM input (xz) and gate (z)
        xz = self.in_proj(x)                               # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)                        # each (B, L, d_inner)

        # Depthwise conv for local temporal context
        # Conv1d expects (B, C, L); we have (B, L, C)
        x_ = x_.transpose(1, 2)                            # (B, d_inner, L)
        x_ = self.conv1d(x_)[..., :x.size(1)]              # causal trim
        x_ = x_.transpose(1, 2)                            # (B, L, d_inner)
        x_ = F.silu(x_)

        # Compute input-dependent SSM parameters: Δ, B, C
        # x_proj: (B, L, dt_rank + 2*d_state)
        x_dbc = self.x_proj(x_)
        delta_raw = x_dbc[..., :self.dt_rank]              # (B, L, dt_rank)
        B = x_dbc[..., self.dt_rank : self.dt_rank + self.d_state]   # (B, L, d_state)
        C = x_dbc[..., self.dt_rank + self.d_state:]                  # (B, L, d_state)

        # Promote dt_rank → d_inner; softplus for positivity
        delta = F.softplus(self.dt_proj(delta_raw))        # (B, L, d_inner)

        # Recover A from log: A = -softplus(A_log) ensures A < 0 → Ā ∈ (0,1)
        A = -torch.exp(self.A_log.float())                 # (d_inner, d_state)

        # Selective scan
        y = selective_scan_seq(
            u     = x_.float(),
            delta = delta.float(),
            A     = A,
            B     = B.float(),
            C     = C.float(),
            D     = self.D.float(),
        ).to(x.dtype)                                      # (B, L, d_inner)

        # Gate with SiLU activation
        y = y * F.silu(z)                                  # (B, L, d_inner)

        # Output projection + dropout + residual
        y = self.dropout(self.out_proj(y))                 # (B, L, d_model)
        return y + residual


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mamba Encoder
# ─────────────────────────────────────────────────────────────────────────────

class MambaEncoder(nn.Module):
    """
    Mamba-2 SSD encoder for multivariate time-series windows.

    Pipeline:
        (B, F, W)
            ↓
        FeatureAttention(F, W)          — cross-sensor correlations   (B, F, W)
            ↓
        permute → (B, W, F)             — time-major for Mamba input
            ↓
        MambaBlock × n_layers           — selective temporal encoding  (B, W, F)
            ↓
        mean-pool over W → (B, F)       — fixed-size summary
            ↓
        FC → z_mu        (B, latent_dim)
        FC → z_log_var   (B, latent_dim)
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim:  int,
        window_size: int,
        d_model:     int = 64,
        d_state:     int = 16,
        n_layers:    int = 4,
        expand:      int = 2,
        n_heads:     int = 4,
        dropout:     float = 0.1,
    ):
        """
        Args:
            in_channels : F — number of sensor channels (38 for SMD).
            latent_dim  : Latent space size.
            window_size : W — input window length.
            d_model     : Mamba hidden dimension. Also used as FeatureAttention d_model.
                          Requirement: d_model % n_heads == 0.
            d_state     : SSM state size N per channel.
            n_layers    : Number of stacked MambaBlocks.
            expand      : Inner expansion factor in each MambaBlock.
            n_heads     : Attention heads in FeatureAttention.
            dropout     : Dropout in MambaBlock output projection.
        """
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.feature_attn = FeatureAttention(
            n_features  = in_channels,
            window_size = window_size,
            d_model     = d_model,
            n_heads     = n_heads,
        )

        # Input embedding: project F channels to d_model
        self.input_proj = nn.Linear(in_channels, d_model)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

        self.fc_mu      = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, F, W)
        Returns:
            mu, log_var : each (B, latent_dim)
        """
        h = self.feature_attn(x)           # (B, F, W)
        h = h.permute(0, 2, 1)             # (B, W, F) — time-major
        h = self.input_proj(h)             # (B, W, d_model)

        for layer in self.mamba_layers:
            h = layer(h)                   # (B, W, d_model) — with residual

        h = self.norm_out(h)               # (B, W, d_model)
        h = h.mean(dim=1)                  # (B, d_model) — mean pool over time

        return self.fc_mu(h), self.fc_log_var(h)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Mamba VAE
# ─────────────────────────────────────────────────────────────────────────────

class MambaVAE(nn.Module):
    """
    VAE with Mamba-2 SSD encoder and ConvTranspose decoder.

    Identical interface to VAE (vae_model.py) and LSTMVAE (lstm_model.py):
        forward(x) → (x_mu, x_log_var, z_mu, z_log_var)

    Decoder is the same ConvTranspose + FeatureAttention stack used in TCN-VAE
    and LSTM-VAE, keeping the decoder architecture constant for fair comparison.

    Checkpoint format:
        {
            'model_state_dict': ...,
            'hyperparameters': {
                'in_channels', 'latent_dim', 'window_size',
                'd_model', 'd_state', 'n_layers', 'expand', 'n_heads'
            },
            'threshold': float,
        }
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim:  int = 32,
        window_size: int = 64,
        d_model:     int = 64,
        d_state:     int = 16,
        n_layers:    int = 4,
        expand:      int = 2,
        n_heads:     int = 4,
    ):
        super().__init__()
        self.encoder = MambaEncoder(
            in_channels = in_channels,
            latent_dim  = latent_dim,
            window_size = window_size,
            d_model     = d_model,
            d_state     = d_state,
            n_layers    = n_layers,
            expand      = expand,
            n_heads     = n_heads,
        )
        # Use d_model as tcn_hidden for the decoder (same effective width)
        self.decoder = Decoder(
            out_channels = in_channels,
            latent_dim   = latent_dim,
            window_size  = window_size,
            tcn_hidden   = d_model,
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_errors(
    model:      MambaVAE,
    dataloader: torch.utils.data.DataLoader,
    device:     torch.device,
) -> torch.Tensor:
    """Per-window MSE anomaly scores. Identical API to vae_model version."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            batch         = batch.to(device)
            x_mu, _, _, _ = model(batch)
            mse = F.mse_loss(x_mu, batch, reduction='none').mean(dim=[1, 2])
            errors.append(mse.cpu())
    return torch.cat(errors, dim=0)
