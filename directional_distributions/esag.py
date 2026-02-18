"""Elliptically Symmetric Angular Gaussian distribution: loss function and evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _construct_orthonormal_basis, _log_M2


def esag_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Elliptically Symmetric Angular Gaussian (ESAG) negative log-likelihood loss.

    The ESAG distribution has ellipse-like contours on the sphere, enabling
    modeling of anisotropic directional uncertainty. It generalizes IAG by
    adding shape parameters γ = (γ₁, γ₂) that control the ellipticity.

    The density is:
        f_ESAG(y) = C_3 / (y'V⁻¹y)^(3/2) * exp[0.5*((y·μ)²/(y'V⁻¹y) - ||μ||²)]
                    * M_2(y·μ / √(y'V⁻¹y))

    where V⁻¹ is constructed from μ and γ per Equation (18) of the paper.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 5] predictions where:
              - pred[:, :3] = μ (mean vector, magnitude controls concentration)
              - pred[:, 3:5] = γ = (γ₁, γ₂) (shape parameters for ellipticity)
              Setting γ = (0, 0) recovers the IAG distribution.
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred[:, :3]      # [B, 3]
    gamma1 = pred[:, 3]   # [B]
    gamma2 = pred[:, 4]   # [B]

    # Normalize y_true to ensure unit vectors
    y = F.normalize(y_true, p=2, dim=1)  # [B, 3]

    # Basic terms
    mu_norm_sq = (mu ** 2).sum(dim=1)  # ||μ||² [B]
    y_dot_mu = (y * mu).sum(dim=1)     # y·μ [B]

    # Construct orthonormal basis {ξ₁, ξ₂} perpendicular to μ
    xi1, xi2 = _construct_orthonormal_basis(mu)  # [B, 3] each

    # Projections of y onto the basis vectors
    a = (y * xi1).sum(dim=1)  # y·ξ₁ [B]
    b = (y * xi2).sum(dim=1)  # y·ξ₂ [B]

    # Compute y'V⁻¹y using Equation (18):
    # y'V⁻¹y = 1 + γ₁(a² - b²) + 2γ₂ab + (√(1 + γ₁² + γ₂²) - 1)(a² + b²)
    gamma_sq = gamma1 ** 2 + gamma2 ** 2
    sqrt_term = torch.sqrt(1.0 + gamma_sq)
    a_sq_plus_b_sq = a ** 2 + b ** 2

    y_Vinv_y = (1.0
                + gamma1 * (a ** 2 - b ** 2)
                + 2.0 * gamma2 * a * b
                + (sqrt_term - 1.0) * a_sq_plus_b_sq)

    # Clamp for numerical stability (V⁻¹ is positive definite, so this should be > 0)
    y_Vinv_y = torch.clamp(y_Vinv_y, min=1e-8)

    # Argument for M_2
    sqrt_y_Vinv_y = torch.sqrt(y_Vinv_y)
    alpha = y_dot_mu / sqrt_y_Vinv_y

    # log(M_2(alpha))
    log_M2 = _log_M2(alpha)

    # NLL = log(2π) + 1.5*log(y'V⁻¹y) + 0.5*(||μ||² - (y·μ)²/(y'V⁻¹y)) - log(M_2(α))
    nll = (np.log(2 * np.pi)
           + 1.5 * torch.log(y_Vinv_y)
           + 0.5 * (mu_norm_sq - y_dot_mu ** 2 / y_Vinv_y)
           - log_M2)

    return nll.mean()


class ESAG(BaseDistribution):
    """Elliptically Symmetric Angular Gaussian distribution on S².

    The ESAG distribution generalizes IAG with ellipse-like contours on the
    sphere, controlled by shape parameters γ = (γ₁, γ₂). Setting γ = (0, 0)
    recovers the IAG distribution.

    Reference: Paine et al. (2018), Stat Comput 28:689-697.

    Args:
        pred: [B, 5] raw network output where pred[:, :3] is μ and
            pred[:, 3:5] is γ = (γ₁, γ₂).
    """

    n_params = 5

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ||μ|| [B]. Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def gamma(self) -> Tensor:
        """Ellipticity parameters (γ₁, γ₂) [B, 2]."""
        return self._pred[:, 3:5]

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_ESAG(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]       # [B, 3]
        gamma1 = self._pred[:, 3]    # [B]
        gamma2 = self._pred[:, 4]    # [B]

        mu_norm_sq = (mu ** 2).sum(dim=1)  # [B]

        # Construct orthonormal basis perpendicular to μ
        xi1, xi2 = _construct_orthonormal_basis(mu)  # [B, 3] each

        # y·μ, y·ξ₁, y·ξ₂ for all (point, sample) pairs
        y_dot_mu = points @ mu.T    # [N, B]
        a = points @ xi1.T          # [N, B]  (y·ξ₁)
        b = points @ xi2.T          # [N, B]  (y·ξ₂)

        # y'V⁻¹y (Equation 18)
        gamma_sq = gamma1 ** 2 + gamma2 ** 2        # [B]
        sqrt_term = torch.sqrt(1.0 + gamma_sq)      # [B]
        a_sq_plus_b_sq = a ** 2 + b ** 2             # [N, B]

        y_Vinv_y = (1.0
                    + gamma1[None, :] * (a ** 2 - b ** 2)
                    + 2.0 * gamma2[None, :] * a * b
                    + (sqrt_term - 1.0)[None, :] * a_sq_plus_b_sq)

        y_Vinv_y = torch.clamp(y_Vinv_y, min=1e-8)  # [N, B]

        # M_2 argument
        sqrt_y_Vinv_y = torch.sqrt(y_Vinv_y)
        alpha = y_dot_mu / sqrt_y_Vinv_y

        log_M2 = _log_M2(alpha)  # [N, B]

        # log f = -log(2π) - 1.5*log(y'V⁻¹y) + 0.5*((y·μ)²/(y'V⁻¹y) - ||μ||²) + log(M_2)
        log_p = (-np.log(2 * np.pi)
                 - 1.5 * torch.log(y_Vinv_y)
                 + 0.5 * (y_dot_mu ** 2 / y_Vinv_y - mu_norm_sq[None, :])
                 + log_M2)

        return log_p.T  # [B, N]
