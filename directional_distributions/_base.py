"""Base class, shared utilities, and grid generation for directional distributions on S²."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._plotting import plot_mollweide

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Sphere grid
# ---------------------------------------------------------------------------

@dataclass
class SphereGrid:
    """A lat/lon grid of points on the unit sphere S².

    Attributes:
        points: [N, 3] unit vectors (Cartesian coordinates).
        lon: [N] longitude in radians, range (-π, π].
        lat: [N] latitude in radians, range [-π/2, π/2].
        shape: (n_lat, n_lon) for reshaping flat arrays into 2D grids.
    """

    points: Tensor
    lon: Tensor
    lat: Tensor
    shape: tuple


def make_grid(
    n_lat: int = 181, n_lon: int = 360, device: torch.device | None = None
) -> SphereGrid:
    """Generate a uniform lat/lon grid of points on S².

    Args:
        n_lat: Number of latitude bins (pole to pole).
        n_lon: Number of longitude bins.
        device: Torch device for the output tensors.

    Returns:
        SphereGrid with n_lat * n_lon points.
    """
    lat = torch.linspace(-np.pi / 2, np.pi / 2, n_lat, device=device)
    lon = torch.linspace(-np.pi, np.pi, n_lon + 1, device=device)[:-1]

    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.reshape(-1)
    lon_flat = lon_grid.reshape(-1)

    cos_lat = torch.cos(lat_flat)
    x = cos_lat * torch.cos(lon_flat)
    y = cos_lat * torch.sin(lon_flat)
    z = torch.sin(lat_flat)

    points = torch.stack([x, y, z], dim=1)

    return SphereGrid(points=points, lon=lon_flat, lat=lat_flat, shape=(n_lat, n_lon))


# ---------------------------------------------------------------------------
# Shared math utilities (used by IAG / ESAG)
# ---------------------------------------------------------------------------

def _log_M2(alpha: Tensor) -> Tensor:
    """Compute log(M_2(alpha)) numerically stably.

    M_2(alpha) = (1 + alpha^2) * Phi(alpha) + alpha * phi(alpha)

    where Phi is the standard normal CDF and phi is the standard normal PDF.

    For large negative alpha, direct computation suffers from catastrophic
    cancellation. We use torch.special.log_ndtr for the tail, which provides
    correct values and gradients even for alpha << 0.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (4).
    """
    # Direct computation (accurate for moderate alpha)
    log_phi = -0.5 * alpha ** 2 - 0.5 * np.log(2 * np.pi)
    phi = torch.exp(log_phi)
    Phi = 0.5 * (1.0 + torch.erf(alpha / np.sqrt(2)))
    M2_direct = (1.0 + alpha ** 2) * Phi + alpha * phi

    # Stable computation for large negative alpha:
    #   M₂(α) = Φ(α) · [(1+α²) + α · φ(α)/Φ(α)]
    #   log M₂ = log Φ(α) + log[(1+α²) + α · exp(log φ(α) - log Φ(α))]
    #
    # The inner term (1+α²) + α·φ/Φ ≈ 2/α² for large |α|, computed as the
    # difference of two ~α²-sized quantities. Float32 loses all significant
    # digits around |α| > 26, so we upcast to float64 for this subtraction.
    #
    # log_ndtr is not implemented for half/bfloat16 on CPU, so upcast if needed.
    compute_dtype = torch.float64 if alpha.dtype != torch.float64 else torch.float64
    alpha_hi = alpha.to(compute_dtype)
    log_phi_hi = -0.5 * alpha_hi ** 2 - 0.5 * np.log(2 * np.pi)
    log_Phi_hi = torch.special.log_ndtr(alpha_hi)
    ratio_hi = alpha_hi * torch.exp(log_phi_hi - log_Phi_hi)
    inner_hi = (1.0 + alpha_hi ** 2) + ratio_hi
    M2_stable = (log_Phi_hi + torch.log(torch.clamp(inner_hi, min=1e-300))).to(alpha.dtype)

    # Use direct for alpha >= -3.5, stable form for alpha < -3.5.
    # The direct branch suffers catastrophic cancellation in M2_direct for
    # alpha < ~-3.8 (float32), while the stable branch (computed in float64)
    # is accurate for all alpha < 0.
    return torch.where(alpha >= -3.5, torch.log(torch.clamp(M2_direct, min=1e-40)), M2_stable)


def _sc_log_density(A: Tensor, B: Tensor, Gamma_sq: Tensor) -> Tensor:
    """Compute the log-density kernel shared by all spherical projected Cauchy distributions.

    Given the three intermediate quantities from the projected Cauchy on S²,
    returns the log-PDF assuming |Σ| = 1.

    The density (Tsagris & Alzeley, 2024, Eq. 18 with |Σ|=1) is:

        log f = -log(4π²) - log(B) - 1.5·log(Δ) + log[B(Γ²+1)·Ω + 2A√Δ]

    where Δ = B(Γ²+1) - A²  and  Ω = 2(π - atan2(√Δ, A)).

    **Numerically stable formulation.** Direct computation of Δ = B·C - A²
    suffers catastrophic cancellation when B and Γ² are large (e.g. extreme
    Cholesky eigenvalues), since both terms overflow to inf before subtraction.

    We instead factor via the Cauchy-Schwarz identity:
        A = √B · √Γ² · cos θ      (where θ is the angle in the transformed space)
        Δ = B · (Γ² · sin²θ + 1)   (no cancellation; always ≥ B)

    Then factor B out of inner:
        Δ/B = r²  where  r² = Γ²·sin²θ + 1
        inner/B = (Γ²+1)·Ω + 2·√Γ²·cosθ·r

    Final log-density:
        log f = -log(4π²) - 1.5·log(B) - 1.5·log(r²) + log(inner_reduced)

    Reference: Tsagris & Alzeley (2024), "Circular and Spherical Projected
    Cauchy Distributions", arXiv:2302.02468v4, Equation (18).

    Args:
        A: [...] y⊤Σ⁻¹μ.
        B: [...] y⊤Σ⁻¹y (positive).
        Gamma_sq: [...] μ⊤Σ⁻¹μ (non-negative).

    Returns:
        [...] log-probability density values (same shape as inputs).
    """
    # Normalized quantities that stay O(1) regardless of scale
    sqrt_B = torch.sqrt(torch.clamp(B, min=1e-30))
    sqrt_G = torch.sqrt(torch.clamp(Gamma_sq, min=0.0))

    # cos θ = A / (√B · √Γ²), clamped to [-1, 1] for numerical safety
    denom = sqrt_B * sqrt_G
    # When Gamma_sq ≈ 0, cos_theta is irrelevant (sin²θ·Γ² → 0 anyway)
    cos_theta = torch.where(
        denom > 1e-15,
        torch.clamp(A / denom, min=-1.0, max=1.0),
        torch.zeros_like(A),
    )
    sin_sq_theta = 1.0 - cos_theta ** 2

    # r² = Γ²·sin²θ + 1  (always ≥ 1, no cancellation)
    r_sq = Gamma_sq * sin_sq_theta + 1.0
    r = torch.sqrt(r_sq)

    # Ω = 2(π - atan2(√Δ, A)) with √Δ = √B · r, A = √B · √Γ² · cosθ
    # atan2(√B · r, √B · √Γ² · cosθ) = atan2(r, √Γ² · cosθ)  [√B cancels]
    Omega = 2.0 * (np.pi - torch.atan2(r, sqrt_G * cos_theta))

    # inner_reduced = (Γ²+1)·Ω + 2·√Γ²·cosθ·r   [B factored out]
    inner_reduced = (Gamma_sq + 1.0) * Omega + 2.0 * sqrt_G * cos_theta * r
    inner_reduced = torch.clamp(inner_reduced, min=1e-30)

    return (-np.log(4.0 * np.pi ** 2)
            - 1.5 * torch.log(torch.clamp(B, min=1e-30))
            - 1.5 * torch.log(r_sq)
            + torch.log(inner_reduced))


def _construct_orthonormal_basis(mu: Tensor) -> Tuple[Tensor, Tensor]:
    """Construct two orthonormal vectors perpendicular to mu using Gram-Schmidt.

    Avoids the singularity when mu is aligned with the x-axis.

    Args:
        mu: [B, 3] mean direction vectors (not necessarily normalized).

    Returns:
        xi1, xi2: [B, 3] orthonormal vectors perpendicular to mu.
    """
    B = mu.shape[0]
    device, dtype = mu.device, mu.dtype
    mu_norm_len = mu.norm(p=2, dim=1, keepdim=True)
    mu_norm = mu / mu_norm_len.clamp_min(1e-12)

    # For zero-mean vectors, pick a deterministic direction so the basis remains valid.
    zero_mu = (mu_norm_len.squeeze(1) <= 1e-12).unsqueeze(1)
    fallback_mu = torch.zeros(B, 3, device=device, dtype=dtype)
    fallback_mu[:, 2] = 1.0
    mu_norm = torch.where(zero_mu, fallback_mu, mu_norm)

    ref1 = torch.zeros(B, 3, device=device, dtype=dtype)
    ref1[:, 0] = 1.0
    ref2 = torch.zeros(B, 3, device=device, dtype=dtype)
    ref2[:, 1] = 1.0

    dot1 = torch.abs((mu_norm * ref1).sum(dim=1))
    use_ref2 = dot1 > 0.9
    ref = torch.where(use_ref2.unsqueeze(1), ref2, ref1)

    xi1 = ref - (ref * mu_norm).sum(dim=1, keepdim=True) * mu_norm
    xi1 = F.normalize(xi1, p=2, dim=1)
    xi2 = torch.cross(mu_norm, xi1, dim=1)
    xi2 = F.normalize(xi2, p=2, dim=1)

    return xi1, xi2


# ---------------------------------------------------------------------------
# Cholesky parameterization utilities (used by GAG / GSPC)
# ---------------------------------------------------------------------------

def _build_cholesky(pred: Tensor) -> Tensor:
    """Construct the normalised lower-triangular Cholesky factor L from raw
    network outputs.

    Args:
        pred: [B, 9] where pred[:, 3:6] are raw log-diagonal entries and
              pred[:, 6:9] are off-diagonal entries (L₂₁, L₃₁, L₃₂).

    Returns:
        L: [B, 3, 3] lower-triangular with det(L) = 1 and V⁻¹ = LLᵀ SPD.
    """
    B = pred.shape[0]
    device, dtype = pred.device, pred.dtype

    raw_log_diag = pred[:, 3:6]   # [B, 3]
    off_diag = pred[:, 6:9]       # [B, 3]  → L₂₁, L₃₁, L₃₂

    # Centre log-diagonal so that sum = 0  ⟹  det(L) = exp(0) = 1
    log_diag = raw_log_diag - raw_log_diag.mean(dim=1, keepdim=True)
    diag = torch.exp(log_diag)    # [B, 3], always positive, product = 1

    # Assemble L  (lower triangular)
    L = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    L[:, 0, 0] = diag[:, 0]
    L[:, 1, 1] = diag[:, 1]
    L[:, 2, 2] = diag[:, 2]
    L[:, 1, 0] = off_diag[:, 0]   # L₂₁
    L[:, 2, 0] = off_diag[:, 1]   # L₃₁
    L[:, 2, 1] = off_diag[:, 2]   # L₃₂

    return L


# ---------------------------------------------------------------------------
# Base distribution class
# ---------------------------------------------------------------------------

class BaseDistribution:
    """Base class for directional distributions on the unit sphere S².

    Subclasses must set :attr:`n_params` and implement
    :attr:`mean_direction` and :meth:`log_pdf`.
    """

    n_params: int

    def __init__(self, pred: Tensor) -> None:
        if pred.shape[-1] != self.n_params:
            raise ValueError(
                f"{type(self).__name__} expects pred with last dim "
                f"{self.n_params}, got {pred.shape[-1]}"
            )
        self._pred = pred

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        raise NotImplementedError

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log-PDF at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        raise NotImplementedError

    def pdf(self, points: Tensor) -> Tensor:
        """Evaluate PDF at points on S²."""
        return self.log_pdf(points).exp()

    def plot_mollweide(
        self,
        idx: int = 0,
        n_lat: int = 181,
        n_lon: int = 360,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plot PDF on a Mollweide projection for a single sample.

        Args:
            idx: Batch index to plot.
            n_lat: Number of latitude bins.
            n_lon: Number of longitude bins.
            **kwargs: Forwarded to :func:`plot_mollweide`.

        Returns:
            ``(fig, ax)`` tuple.
        """
        grid = make_grid(n_lat, n_lon, device=self._pred.device)
        with torch.no_grad():
            vals = self.pdf(grid.points)[idx]
        vals_2d = vals.reshape(grid.shape).cpu()
        return plot_mollweide(grid, vals_2d, **kwargs)
