"""Spherical Projected Cauchy family: SIPC, SESPC, and GSPC distributions on S^2.

This module collects the three members of the spherical projected Cauchy
family, obtained by projecting a trivariate Cauchy C(mu, Sigma) onto the
sphere via z -> z/||z||.  They form a nested hierarchy:

    SIPC (3 params)  <  SESPC (5 params)  <  GSPC (8 params)

References
----------
Tsagris & Alzeley (2024), "Circular and Spherical Projected Cauchy
Distributions", arXiv:2302.02468v4.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _construct_orthonormal_basis, _build_cholesky, _sc_log_density


# ---------------------------------------------------------------------------
# Spherical Isotropic Projected Cauchy (SIPC)
# ---------------------------------------------------------------------------

def sipc_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Spherical Isotropic Projected Cauchy (SIPC) negative log-likelihood loss.

    The SIPC is the projected Cauchy with Sigma = I, making it rotationally
    symmetric about the mean direction.  It is the Cauchy analog of IAG.

    The density is:

        f(y; mu) = [B(Gamma^2+1)*sqrt(Delta)*Omega + 2A*Delta] / Delta^2

    where A = y.mu, B = 1, Gamma^2 = ||mu||^2, Delta = Gamma^2+1-A^2.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (20).

    Args:
        pred: [B, 3] predicted mean vectors mu.  The magnitude ||mu|| controls
              concentration (higher = more peaked), and mu/||mu|| is the mean
              direction.
        y_true: [B, 3] true unit direction vectors on S^2.

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred                                       # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)             # [B, 3]

    # With Sigma = I: A = y.mu, B = ||y||^2 = 1, Gamma^2 = ||mu||^2
    A = (y * mu).sum(dim=1)                          # [B]
    B = torch.ones_like(A)                           # [B]
    Gamma_sq = (mu ** 2).sum(dim=1)                  # [B]

    return -_sc_log_density(A, B, Gamma_sq).mean()


class SIPC(BaseDistribution):
    """Spherical Isotropic Projected Cauchy distribution on S^2.

    The SIPC is rotationally symmetric about the mean direction,
    with concentration controlled by ||mu||.  It is the Cauchy analog of IAG.

    The density is derived by projecting a trivariate Cauchy C(mu, I)
    onto the sphere via z -> z/||z||.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (20).

    Args:
        pred: [B, 3] raw network output (the mean vector mu).
    """

    n_params = 3

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred, p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B].  Higher = more peaked."""
        return self._pred.norm(p=2, dim=1)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_SIPC(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred                              # [B, 3]
        Gamma_sq = (mu ** 2).sum(dim=1)              # [B]

        A = points @ mu.T                            # [N, B]
        B = torch.ones_like(A)                       # [N, B]
        Gamma_sq_exp = Gamma_sq[None, :].expand_as(A)  # [N, B]

        return _sc_log_density(A, B, Gamma_sq_exp).T  # [B, N]


# ---------------------------------------------------------------------------
# Spherical Elliptically Symmetric Projected Cauchy (SESPC)
# ---------------------------------------------------------------------------

def sespc_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Spherical Elliptically Symmetric Projected Cauchy (SESPC) NLL loss.

    The SESPC generalises SIPC with ellipse-like contours on the sphere,
    controlled by shape parameters gamma = (gamma_1, gamma_2).  It is the Cauchy analog
    of ESAG.

    The Sigma^{-1} construction follows Paine et al. (2018), identical to ESAG.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (22).

    Args:
        pred: [B, 5] predictions where:
              - pred[:, :3] = mu (mean vector, magnitude controls concentration)
              - pred[:, 3:5] = gamma = (gamma_1, gamma_2) (shape parameters for ellipticity)
              Setting gamma = (0, 0) recovers the SIPC distribution.
        y_true: [B, 3] true unit direction vectors on S^2.

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred[:, :3]                                  # [B, 3]
    gamma1 = pred[:, 3]                               # [B]
    gamma2 = pred[:, 4]                               # [B]

    y = F.normalize(y_true, p=2, dim=1)               # [B, 3]

    # Since Sigma*mu = mu: A = y.mu, Gamma^2 = ||mu||^2
    A = (y * mu).sum(dim=1)                            # [B]
    Gamma_sq = (mu ** 2).sum(dim=1)                    # [B]

    # Construct orthonormal basis perpendicular to mu
    xi1, xi2 = _construct_orthonormal_basis(mu)        # [B, 3] each

    # Projections
    a = (y * xi1).sum(dim=1)                           # [B]
    b = (y * xi2).sum(dim=1)                           # [B]

    # B = y^T Sigma^{-1} y (Paine et al. 2018, Eq. 18)
    gamma_sq = gamma1 ** 2 + gamma2 ** 2
    sqrt_term = torch.sqrt(1.0 + gamma_sq)
    a_sq_plus_b_sq = a ** 2 + b ** 2

    B = (1.0
         + gamma1 * (a ** 2 - b ** 2)
         + 2.0 * gamma2 * a * b
         + (sqrt_term - 1.0) * a_sq_plus_b_sq)

    B = torch.clamp(B, min=1e-8)                      # [B]

    return -_sc_log_density(A, B, Gamma_sq).mean()


class SESPC(BaseDistribution):
    """Spherical Elliptically Symmetric Projected Cauchy distribution on S^2.

    The SESPC generalises SIPC with ellipse-like contours on the sphere,
    controlled by shape parameters gamma = (gamma_1, gamma_2).  Setting gamma = (0, 0)
    recovers the SIPC distribution.  It is the Cauchy analog of ESAG.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (22).

    Args:
        pred: [B, 5] raw network output where pred[:, :3] is mu and
            pred[:, 3:5] is gamma = (gamma_1, gamma_2).
    """

    n_params = 5

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def gamma(self) -> Tensor:
        """Ellipticity parameters (gamma_1, gamma_2) [B, 2]."""
        return self._pred[:, 3:5]

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_SESPC(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                         # [B, 3]
        gamma1 = self._pred[:, 3]                      # [B]
        gamma2 = self._pred[:, 4]                      # [B]
        Gamma_sq = (mu ** 2).sum(dim=1)                # [B]

        xi1, xi2 = _construct_orthonormal_basis(mu)    # [B, 3] each

        # [N, B] intermediates
        A = points @ mu.T                              # [N, B]
        a = points @ xi1.T                             # [N, B]
        b = points @ xi2.T                             # [N, B]

        gamma_sq = gamma1 ** 2 + gamma2 ** 2           # [B]
        sqrt_term = torch.sqrt(1.0 + gamma_sq)         # [B]
        a_sq_plus_b_sq = a ** 2 + b ** 2               # [N, B]

        B = (1.0
             + gamma1[None, :] * (a ** 2 - b ** 2)
             + 2.0 * gamma2[None, :] * a * b
             + (sqrt_term - 1.0)[None, :] * a_sq_plus_b_sq)

        B = torch.clamp(B, min=1e-8)                  # [N, B]

        Gamma_sq_exp = Gamma_sq[None, :].expand_as(A)  # [N, B]

        return _sc_log_density(A, B, Gamma_sq_exp).T   # [B, N]


# ---------------------------------------------------------------------------
# General Spherical Projected Cauchy (GSPC)
# ---------------------------------------------------------------------------

def gspc_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """General Spherical Projected Cauchy (GSPC) negative log-likelihood loss.

    The GSPC is the full 8-parameter projected Cauchy on S^2, with density
    given by Eq. (18) of the reference with |Sigma| = 1.

    Sigma^{-1} is parameterised as LL^T via a log-Cholesky factor with det(L) = 1,
    identical to the GAG parameterisation.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (18).

    Args:
        pred: [B, 9] predictions where:
              - pred[:, :3]  = mu  (mean vector, unconstrained)
              - pred[:, 3:6] = raw log-diagonal of Cholesky factor L
              - pred[:, 6:9] = off-diagonal entries (L_21, L_31, L_32)
        y_true: [B, 3] true unit direction vectors on S^2.

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred[:, :3]                                   # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)                # [B, 3]
    L = _build_cholesky(pred)                          # [B, 3, 3]

    # Transformed vectors: z_y = L^T y,  z_mu = L^T mu
    z_y = torch.einsum('bji,bj->bi', L, y)            # [B, 3]
    z_mu = torch.einsum('bji,bj->bi', L, mu)          # [B, 3]

    B = (z_y ** 2).sum(dim=1)                          # ||z_y||^2 = y^T Sigma^{-1} y  [B]
    Gamma_sq = (z_mu ** 2).sum(dim=1)                  # ||z_mu||^2 = mu^T Sigma^{-1} mu  [B]
    A = (z_y * z_mu).sum(dim=1)                        # z_y.z_mu = y^T Sigma^{-1} mu  [B]

    return -_sc_log_density(A, B, Gamma_sq).mean()


class GSPC(BaseDistribution):
    """General Spherical Projected Cauchy distribution on S^2.

    The GSPC is the full 8-parameter projected Cauchy, generalising SESPC
    by allowing the scatter matrix eigenvectors to be independent of mu.
    This enables asymmetric, non-elliptical contours on the sphere.

    Sigma^{-1} is parameterised via a log-Cholesky factor L with det(L) = 1.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (18).

    Args:
        pred: [B, 9] raw network output where pred[:, :3] is mu,
            pred[:, 3:6] is the raw log-diagonal of L, and
            pred[:, 6:9] is the off-diagonal (L_21, L_31, L_32).
    """

    n_params = 9

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def cholesky_factor(self) -> Tensor:
        """Normalised lower-triangular Cholesky factor L [B, 3, 3].

        Sigma^{-1} = LL^T with det(L) = 1.
        """
        return _build_cholesky(self._pred)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_GSPC(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                          # [B, 3]
        L = _build_cholesky(self._pred)                 # [B, 3, 3]

        # Transform all grid points and mu through L^T
        z_y = torch.einsum('bji,nj->bni', L, points)   # [B, N, 3]
        z_mu = torch.einsum('bji,bj->bi', L, mu)       # [B, 3]

        B = (z_y ** 2).sum(dim=2)                       # [B, N]
        Gamma_sq = (z_mu ** 2).sum(dim=1)               # [B]
        A = torch.einsum('bni,bi->bn', z_y, z_mu)      # [B, N]

        Gamma_sq_exp = Gamma_sq[:, None].expand_as(A)   # [B, N]

        return _sc_log_density(A, B, Gamma_sq_exp)      # [B, N]
