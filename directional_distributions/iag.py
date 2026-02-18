"""Isotropic Angular Gaussian distribution: loss function and evaluation."""

import numpy as np
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _log_M2


def iag_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Isotropic Angular Gaussian (IAG) negative log-likelihood loss.

    The IAG distribution is the angular Gaussian with V = I (identity covariance),
    making it rotationally symmetric about the mean direction. It is a special
    case of ESAG with gamma = (0, 0).

    The density is:
        f_IAG(y) = (1/2π) * exp[0.5 * ((y·μ)² - ||μ||²)] * M_2(y·μ)

    where M_2(α) = (1 + α²)Φ(α) + αφ(α), with φ and Φ being the standard
    normal PDF and CDF respectively.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 3] predicted mean vectors μ. The magnitude ||μ|| controls
              concentration (higher = more peaked), and μ/||μ|| is the mean direction.
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred  # [B, 3]

    # Normalize y_true to ensure unit vectors
    y = F.normalize(y_true, p=2, dim=1)  # [B, 3]

    # Compute terms
    mu_norm_sq = (mu ** 2).sum(dim=1)  # ||μ||² [B]
    y_dot_mu = (y * mu).sum(dim=1)     # y·μ [B]

    # log(M_2(y·μ))
    log_M2 = _log_M2(y_dot_mu)

    # NLL = log(2π) + 0.5*(||μ||² - (y·μ)²) - log(M_2(y·μ))
    nll = np.log(2 * np.pi) + 0.5 * (mu_norm_sq - y_dot_mu ** 2) - log_M2

    return nll.mean()


class IAG(BaseDistribution):
    """Isotropic Angular Gaussian distribution on S².

    The IAG distribution is rotationally symmetric about the mean direction,
    with concentration controlled by ||μ||.

    The density is:
        f_IAG(y) = (1/2π) exp[0.5((y·μ)² - ||μ||²)] M_2(y·μ)

    Reference: Paine et al. (2018), Stat Comput 28:689-697.

    Args:
        pred: [B, 3] raw network output (the mean vector μ).
    """

    n_params = 3

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred, p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ||μ|| [B]. Higher = more peaked."""
        return self._pred.norm(p=2, dim=1)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_IAG(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred  # [B, 3]
        mu_norm_sq = (mu ** 2).sum(dim=1)  # [B]

        y_dot_mu = points @ mu.T  # [N, B]
        log_M2 = _log_M2(y_dot_mu)  # [N, B]

        # log f = -log(2π) + 0.5*((y·μ)² - ||μ||²) + log(M_2(y·μ))
        log_p = -np.log(2 * np.pi) + 0.5 * (y_dot_mu ** 2 - mu_norm_sq[None, :]) + log_M2

        return log_p.T  # [B, N]
