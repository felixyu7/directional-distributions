"""Von Mises-Fisher distribution: loss function and evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution


def von_mises_fisher_loss(
    n_pred: Tensor, n_true: Tensor, kappa_reg: float = 0.01, eps: float = 1e-8
) -> Tensor:
    """
    von Mises-Fisher loss with decoupled direction and κ.

    Expects n_pred [B,4]: direction = n_pred[:,:3], κ = softplus(n_pred[:,3]).
    The κ head is regularized to prevent explosion while preserving vMF gradient scaling.
    """
    direction = F.normalize(n_pred[:, :3], p=2, dim=1)
    kappa = F.softplus(n_pred[:, 3]) + 0.1
    cos_sim = (direction * n_true).sum(dim=1)
    log_C = -kappa + torch.log((kappa + eps) / (1 - torch.exp(-2 * kappa) + 2 * eps))
    return (-(kappa * cos_sim + log_C) + kappa_reg * kappa).mean()


class VMF(BaseDistribution):
    """Von Mises-Fisher distribution on S².

    The vMF distribution has density:

        f(y | μ, κ) = C(κ) exp(κ μ·y)

    where C(κ) = κ / (4π sinh(κ)) is the normalization constant.

    Args:
        pred: [B, 4] raw network output where pred[:, :3] is the
            (unnormalized) direction and pred[:, 3] is the raw κ
            (before softplus).
    """

    n_params = 4

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def kappa(self) -> Tensor:
        """Concentration parameter κ [B]. Always >= 0.1."""
        return F.softplus(self._pred[:, 3]) + 0.1

    def log_pdf(self, points: Tensor, eps: float = 1e-8) -> Tensor:
        """Evaluate log f(y | μ, κ) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.
            eps: Small constant for numerical stability.

        Returns:
            [B, N] log-probability density.
        """
        mu = self.mean_direction  # [B, 3]
        kappa = self.kappa        # [B]

        cos_sim = points @ mu.T  # [N, B]

        # log C(κ) = log(κ / (4π sinh(κ)))
        #          = log(κ) - log(2π) - κ - log(1 - exp(-2κ))
        # The loss function omits -log(2π) since it's constant w.r.t. parameters,
        # but here we need the proper normalization for a valid PDF.
        log_C = (
            -kappa
            + torch.log((kappa + eps) / (1 - torch.exp(-2 * kappa) + 2 * eps))
            - np.log(2 * np.pi)
        )

        return (log_C[None, :] + kappa[None, :] * cos_sim).T  # [B, N]
