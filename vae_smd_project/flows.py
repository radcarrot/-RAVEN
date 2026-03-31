"""
flows.py
--------
Normalizing flow modules for enhancing the VAE prior.

Implements Planar Flows (Rezende & Mohamed, 2015) that transform the
simple Gaussian posterior into a more expressive distribution. This gives
better density estimation in the latent space, making anomalies easier
to detect via the KL-based component of the anomaly score.

Usage:
    Replace the standard reparameterize() call with a flow-enhanced version:

    flow = PlanarFlow(latent_dim=32, n_flows=8)
    z0 = reparameterize(mu, log_var)        # initial sample from q(z|x)
    zk, log_det = flow(z0)                  # transform through flow
    # Use zk for decoding; add log_det to KL for correct ELBO

Integration with the existing VAE:

    from flows import PlanarFlow
    from vae_model import reparameterize

    # After creating the VAE:
    flow = PlanarFlow(latent_dim=LATENT_DIM, n_flows=8).to(DEVICE)

    # In the training loop, replace:
    #   x_mu, x_log_var, z_mu, z_log_var = model(batch)
    # With:
    #   z_mu, z_log_var = model.encoder(batch)
    #   z0 = reparameterize(z_mu, z_log_var)
    #   zk, log_det = flow(z0)
    #   x_mu, x_log_var = model.decoder(zk)
    #
    # In the loss, subtract the log_det from the KL term:
    #   kl_loss = kl_loss - log_det.mean()
    #
    # The modified ELBO becomes:
    #   L = recon_loss + beta * (kl_loss - log_det.mean())
    #
    # This works because the ELBO with normalizing flows is:
    #   log p(x) >= E_q[log p(x|zK)] - KL(q(z0|x) || p(z0)) + E_q[sum log_det_k]
    #
    # The sum_log_det term corrects the KL divergence for the change of
    # variables introduced by the flow, effectively allowing q(zK|x) to be
    # a more expressive distribution than the original Gaussian q(z0|x).
    #
    # Remember to include flow.parameters() in the optimizer:
    #   optimizer = torch.optim.Adam(
    #       list(model.parameters()) + list(flow.parameters()), lr=LR
    #   )
"""

import torch
import torch.nn as nn
from typing import Tuple


class PlanarTransform(nn.Module):
    """
    Single planar flow transformation:
        f(z) = z + u * tanh(w^T z + b)

    where u, w in R^d, b in R are learnable parameters.

    The log-determinant of the Jacobian is:
        log |det df/dz| = log |1 + u^T * h'(w^T z + b) * w|
    where h' is the derivative of tanh.

    Reference:
        Rezende, D. J. & Mohamed, S. (2015). Variational Inference with
        Normalizing Flows. ICML 2015.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim : Dimensionality of the latent space (must match latent_dim
                  used in the VAE encoder/decoder).
        """
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def _constrained_u(self) -> torch.Tensor:
        """
        Enforce the invertibility condition: w^T u >= -1.

        Without this constraint, the transformation can become non-invertible,
        breaking the change-of-variables formula. Uses the reparameterization
        from Appendix A.1 of Rezende & Mohamed (2015):

            u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2

        where m(x) = -1 + softplus(x) = -1 + log(1 + exp(x)), which ensures
        w^T u_hat >= -1.
        """
        wu = torch.dot(self.w, self.u)
        m_wu = -1 + torch.log1p(torch.exp(wu))  # softplus shifted by -1
        return self.u + (m_wu - wu) * self.w / (torch.dot(self.w, self.w) + 1e-8)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply one planar transformation.

        Args:
            z : Tensor, shape (B, dim) -- input latent vectors.
        Returns:
            z_out   : Tensor, shape (B, dim) -- transformed latent vectors.
            log_det : Tensor, shape (B,) -- log |det Jacobian| per sample.
        """
        u = self._constrained_u()

        # w^T z + b  ->  (B,)
        linear = torch.mv(z, self.w) + self.b                            # (B,)

        # f(z) = z + u * tanh(w^T z + b)
        z_out = z + u.unsqueeze(0) * torch.tanh(linear).unsqueeze(1)     # (B, dim)

        # log |det Jacobian| = log |1 + u^T * (1 - tanh^2(w^T z + b)) * w|
        psi = (1 - torch.tanh(linear).pow(2)).unsqueeze(1) * self.w.unsqueeze(0)  # (B, dim)
        log_det = torch.log(torch.abs(1 + torch.mv(psi, u)) + 1e-8)              # (B,)

        return z_out, log_det


class PlanarFlow(nn.Module):
    """
    Stack of K planar transformations.

    Transforms z0 ~ q(z|x) = N(mu, sigma^2) through K invertible mappings:
        z0 -> z1 -> z2 -> ... -> zK

    The ELBO with flows becomes:
        log p(x) >= E_q[log p(x|zK)] - KL(q(z0|x) || p(z0)) + E_q[sum log_det_k]

    The sum of log-determinants corrects the KL for the flow transformations,
    allowing a more expressive approximate posterior. Each planar transform
    adds a single "bump" to the density, so K transforms can model distributions
    with up to K modes -- substantially more flexible than the unimodal Gaussian
    posterior of the base VAE.

    For anomaly detection, this means:
      - Normal data clusters more tightly in high-density regions of latent space.
      - Anomalies land in lower-density regions with higher confidence.
      - The KL component of the anomaly score becomes more discriminative.

    Args:
        latent_dim : Dimensionality of the latent space (must match the VAE).
        n_flows    : Number of planar transformations to stack (default 8).
                     More flows = more expressive, but diminishing returns
                     beyond ~16 for typical latent_dim=32.
    """

    def __init__(self, latent_dim: int, n_flows: int = 8):
        super().__init__()
        self.transforms = nn.ModuleList([
            PlanarTransform(latent_dim) for _ in range(n_flows)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass z through all K planar transformations sequentially.

        Args:
            z : Tensor, shape (B, latent_dim) -- initial sample from q(z|x).
        Returns:
            zk          : Tensor, shape (B, latent_dim) -- final transformed sample.
            sum_log_det : Tensor, shape (B,) -- total log-determinant across all
                          K flows. Add this to the ELBO to correct the KL term.
        """
        sum_log_det = torch.zeros(z.size(0), device=z.device)
        zk = z

        for transform in self.transforms:
            zk, log_det = transform(zk)
            sum_log_det += log_det

        return zk, sum_log_det
