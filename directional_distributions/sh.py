"""Spherical Harmonic (SOS) distribution: loss function and evaluation.

Implements a non-parametric density on S² via Sum-of-Squares of Spherical
Harmonic expansions.  The density is:

    f(y) = (ε + Σ_k g_k(y)²) / Z_ε

where each g_k is a real SH expansion up to degree *L* and
Z_ε = ε·4π + Σ_k ‖c_k‖² is the closed-form normalization constant
(via Parseval's theorem + offset regularization).

See ``sh_proposal.md`` for the full mathematical derivation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution
from ._sh import real_spherical_harmonics


# ---------------------------------------------------------------------------
# Analytical coordinate multiplication matrices for real SH
# ---------------------------------------------------------------------------


def _coord_mult_matrices(
    L_max: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build coordinate multiplication matrices R_x, R_y, R_z analytically.

    ``R_i[d, b]`` is the coefficient of ``Y_d`` in the expansion of
    ``y_i · Y_b``, computed from the three-term recurrence relations
    for real spherical harmonics.

    Convention: orthonormal real SH with Condon-Shortley phase,
    flat index ``p = ℓ² + ℓ + m``.

    Args:
        L_max: Maximum SH degree for the square matrices.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``[3, P, P]`` tensor where ``P = (L_max + 1)²`` and
        axis 0 is (x, y, z).
    """
    P = (L_max + 1) ** 2

    def idx(l: int, m: int) -> int:
        return l * l + l + m

    # Coupling coefficients (Codex notation)
    def _A(l: int, n: int) -> float:
        return math.sqrt((l + n + 1) * (l + n + 2) / ((2 * l + 1) * (2 * l + 3)))

    def _B(l: int, n: int) -> float:
        denom = (2 * l - 1) * (2 * l + 1)
        if denom <= 0:
            return 0.0
        return math.sqrt(max((l - n) * (l - n - 1), 0) / denom)

    def _C(l: int, n: int) -> float:
        return math.sqrt((l - n + 1) * (l - n + 2) / ((2 * l + 1) * (2 * l + 3)))

    def _D(l: int, n: int) -> float:
        denom = (2 * l - 1) * (2 * l + 1)
        if denom <= 0:
            return 0.0
        return math.sqrt(max((l + n) * (l + n - 1), 0) / denom)

    def _az(l: int, m: int) -> float:
        """z-coupling coefficient a_{l,m}."""
        denom = 4 * l * l - 1
        if denom <= 0:
            return 0.0
        return math.sqrt(max(l * l - m * m, 0) / denom)

    sqrt2 = math.sqrt(2.0)

    # Collect (coord_idx, target, source, value) triples
    triples: list[tuple[int, int, int, float]] = []

    def _add(ci: int, l_t: int, m_t: int, b: int, val: float) -> None:
        if val == 0.0 or l_t < 0 or l_t > L_max or abs(m_t) > l_t:
            return
        triples.append((ci, idx(l_t, m_t), b, val))

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            b = idx(l, m)
            n = abs(m)

            # ---- z coupling (preserves m) ----
            _add(2, l + 1, m, b, _az(l + 1, m))
            _add(2, l - 1, m, b, _az(l, m))

            # ---- x, y coupling (shifts m by ±1) ----
            # Formulas from Codex, then negated for our CS convention
            A, B, C, D = _A(l, n), _B(l, n), _C(l, n), _D(l, n)

            if m == 0:
                _add(0, l + 1, 1, b, A / sqrt2)
                _add(0, l - 1, 1, b, -B / sqrt2)
                _add(1, l + 1, -1, b, A / sqrt2)
                _add(1, l - 1, -1, b, -B / sqrt2)

            elif m == 1:
                _add(0, l - 1, 0, b, D / sqrt2)
                _add(0, l + 1, 0, b, -C / sqrt2)
                _add(0, l - 1, 2, b, -B / 2)
                _add(0, l + 1, 2, b, A / 2)
                _add(1, l - 1, -2, b, -B / 2)
                _add(1, l + 1, -2, b, A / 2)

            elif m > 1:
                _add(0, l - 1, m - 1, b, D / 2)
                _add(0, l - 1, m + 1, b, -B / 2)
                _add(0, l + 1, m - 1, b, -C / 2)
                _add(0, l + 1, m + 1, b, A / 2)
                _add(1, l - 1, 1 - m, b, -D / 2)
                _add(1, l - 1, -m - 1, b, -B / 2)
                _add(1, l + 1, 1 - m, b, C / 2)
                _add(1, l + 1, -m - 1, b, A / 2)

            elif m == -1:
                _add(0, l - 1, -2, b, -B / 2)
                _add(0, l + 1, -2, b, A / 2)
                _add(1, l - 1, 0, b, D / sqrt2)
                _add(1, l + 1, 0, b, -C / sqrt2)
                _add(1, l - 1, 2, b, B / 2)
                _add(1, l + 1, 2, b, -A / 2)

            else:  # m < -1
                _add(0, l - 1, m + 1, b, D / 2)
                _add(0, l - 1, m - 1, b, -B / 2)
                _add(0, l + 1, m + 1, b, -C / 2)
                _add(0, l + 1, m - 1, b, A / 2)
                _add(1, l - 1, -m - 1, b, D / 2)
                _add(1, l - 1, -m + 1, b, B / 2)
                _add(1, l + 1, -m - 1, b, -C / 2)
                _add(1, l + 1, -m + 1, b, -A / 2)

    # Materialize dense matrices
    R = torch.zeros(3, P, P, dtype=dtype, device=device)
    for ci, d, b, val in triples:
        R[ci, d, b] += val

    # Sign correction for our CS convention: x and y coupling are negated
    R[0] = -R[0]
    R[1] = -R[1]

    return R


def sh_nll_loss(
    pred: Tensor,
    y_true: Tensor,
    L: int,
    K: int,
    eps: float = 1e-6,
) -> Tensor:
    """Spherical-harmonic sum-of-squares negative log-likelihood loss.

    The density is ``f(y) = (eps + sum_k g_k(y)^2) / Z_eps`` where
    ``g_k(y) = sum_{lm} c_{k,lm} Y_l^m(y)`` and
    ``Z_eps = eps * 4*pi + sum_k ||c_k||^2`` (Parseval).

    Args:
        pred: ``[B, K*(L+1)^2]`` raw network output (SH coefficients).
        y_true: ``[B, 3]`` true unit direction vectors on S².
        L: Maximum spherical harmonic degree.
        K: Number of SOS channels.
        eps: Offset regularization for numerical safety.

    Returns:
        Scalar mean NLL loss over the batch.
    """
    P = (L + 1) ** 2

    # Reshape coefficients: [B, K*P] -> [B, K, P]
    coeffs = pred.reshape(-1, K, P)

    # Normalize y_true to ensure unit vectors
    y = F.normalize(y_true, p=2, dim=1)  # [B, 3]

    # Evaluate SH at observation points
    Y = real_spherical_harmonics(y, L)  # [B, P]

    # g_k(y) for each channel: [B, K]
    g = (coeffs * Y.unsqueeze(1)).sum(dim=-1)  # [B, K]

    # Sum of squares at observation points
    sum_g2 = (g ** 2).sum(dim=-1)  # [B]

    # Normalization constant (Parseval + offset)
    coeffs_norm_sq = (coeffs ** 2).sum(dim=(-1, -2))  # [B]
    Z_eps = eps * 4.0 * math.pi + coeffs_norm_sq  # [B]

    # NLL: -log(eps + sum_g2) + log(Z_eps)
    nll = -torch.log(eps + sum_g2) + torch.log(Z_eps)

    return nll.mean()


class SH(BaseDistribution):
    """Sum-of-Squares Spherical Harmonic distribution on S².

    A non-parametric density on the unit sphere using *K* channels of
    real spherical harmonic expansions up to degree *L*.  The density is
    guaranteed non-negative and has a closed-form normalization constant
    via Parseval's theorem.

    Unlike parametric distributions (VMF, IAG, ESAG), the number of
    parameters depends on the configuration: ``n_params = K * (L+1)²``.

    Args:
        pred: ``[B, K*(L+1)²]`` raw network output (SH coefficients).
        L: Maximum spherical harmonic degree.
        K: Number of SOS channels (default 1).
        eps: Offset regularization (default 1e-6).
    """

    def __init__(
        self,
        pred: Tensor,
        L: int,
        K: int = 1,
        eps: float = 1e-6,
    ) -> None:
        self.L = L
        self.K = K
        self.eps = eps
        # Instance attribute shadows class attribute so BaseDistribution.__init__
        # sees the correct value for validation.
        self.n_params = K * (L + 1) ** 2
        super().__init__(pred)

    @property
    def coeffs(self) -> Tensor:
        """SH coefficients reshaped to ``[B, K, (L+1)²]``."""
        P = (self.L + 1) ** 2
        return self._pred.reshape(-1, self.K, P)

    @property
    def mean_direction(self) -> Tensor:
        """Closed-form mean direction via Gaunt coefficients ``[B, 3]``.

        Computes the exact mean resultant vector E[y] using the identity
        that Cartesian coordinates are proportional to ℓ=1 real SH.
        The integral reduces to a quadratic form in the SH coefficients
        mediated by the Gaunt coupling matrices (triple-SH integrals).

        Returns the normalized mean direction E[y] / ||E[y]||.
        """
        return F.normalize(self._mean_vector(), p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Mean resultant length ||E[y]|| ``[B]``.

        Ranges from 0 (uniform) to 1 (point mass).  Higher values
        indicate a more peaked distribution.  For VMF, this is a
        monotonic function of κ.
        """
        return self._mean_vector().norm(p=2, dim=1)

    @property
    def second_moment_matrix(self) -> Tensor:
        """Second moment matrix E[y_i y_j] ``[B, 3, 3]``.

        Closed-form via the coordinate multiplication matrices R_i.
        Since R_i[a,b] = ∫ Y_a y_i Y_b dω (by orthonormality), the
        second-moment coupling is M2[i,j] = R_i @ R_j through an
        extended basis (degree L+1) to avoid truncation error.

        The trace equals 1 for ε→0 (since ||y||² = 1 on S²).
        """
        return self._second_moment()

    @property
    def covariance_matrix(self) -> Tensor:
        """Covariance matrix Cov[y_i, y_j] ``[B, 3, 3]``.

        Defined as E[y y^T] - E[y] E[y]^T in the R³ embedding.
        The eigenvalues and eigenvectors describe the shape and
        orientation of the directional uncertainty: the eigenvector
        with the largest eigenvalue is the direction of greatest
        angular spread (analogous to ESAG's γ parameters).
        """
        mean = self._mean_vector()  # [B, 3]
        M2 = self._second_moment()  # [B, 3, 3]
        return M2 - torch.einsum("bi,bj->bij", mean, mean)

    def _Z_eps(self) -> Tensor:
        """Normalization constant Z_ε ``[B]``."""
        coeffs = self.coeffs
        coeffs_norm_sq = (coeffs ** 2).sum(dim=(-1, -2))
        return self.eps * 4.0 * math.pi + coeffs_norm_sq

    def _mean_vector(self) -> Tensor:
        """Compute the mean resultant vector E[y] ``[B, 3]`` (unnormalized).

        Uses the Gaunt integral identity:

            E[y_i] = (1/Z_ε) Σ_k c_k^T M_i c_k

        where M_i[a,b] = ∫ Y_a(y) Y_b(y) y_i dω(y) are the Gaunt
        coupling matrices for the three Cartesian coordinates.  These
        are precomputed once per *L* via :meth:`_moment_matrices`.
        """
        M1, _ = self._moment_matrices(self.L, self._pred.device)
        M1 = M1.to(self._pred.dtype)
        coeffs = self.coeffs  # [B, K, P]
        Z = self._Z_eps()  # [B]

        # E[y_i] = (1/Z) Σ_k c_k^T M1[i] c_k
        # M1: [3, P, P], coeffs: [B, K, P]
        vals = torch.einsum("bkp,ipq,bkq->bi", coeffs, M1, coeffs)  # [B, 3]
        return vals / Z.unsqueeze(1)

    def _second_moment(self) -> Tensor:
        """Compute E[y_i y_j] ``[B, 3, 3]``.

        Uses the identity:

            E[y_i y_j] = (ε/Z_ε)(4π/3)δ_ij + (1/Z_ε) Σ_k c_k^T N_ij c_k

        where N_ij[a,b] = ∫ Y_a(y) Y_b(y) y_i y_j dω(y).  The ε term
        accounts for the uniform offset: ∫ y_i y_j dω = (4π/3) δ_ij.
        """
        _, M2 = self._moment_matrices(self.L, self._pred.device)
        M2 = M2.to(self._pred.dtype)
        coeffs = self.coeffs  # [B, K, P]
        Z = self._Z_eps()  # [B]
        B = coeffs.shape[0]

        # Quadratic form: Σ_k c_k^T N_ij c_k for each (i,j)
        vals = torch.einsum("bkp,ijpq,bkq->bij", coeffs, M2, coeffs)  # [B, 3, 3]
        result = vals / Z.view(B, 1, 1)

        # Add epsilon contribution: (ε/Z)(4π/3) δ_ij
        eps_term = self.eps * 4.0 * math.pi / 3.0 / Z  # [B]
        result = result + eps_term.view(B, 1, 1) * torch.eye(
            3, device=self._pred.device, dtype=self._pred.dtype
        )

        return result

    # ------------------------------------------------------------------
    # Precomputed moment matrices (cached per L, device)
    # ------------------------------------------------------------------

    _moment_cache: dict[
        tuple[int, torch.device | None],
        tuple[Tensor, Tensor],
    ] = {}

    @classmethod
    def _moment_matrices(
        cls, L: int, device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        """Precompute (or retrieve cached) Gaunt coupling matrices for
        first and second moments.

        Returns ``(M1, M2)`` where:

        - ``M1``: ``[3, P, P]`` — first-moment matrices (E[y_i]).
          ``M1[i, a, b] = ∫ Y_a Y_b y_i dω = R_i[a, b]``.

        - ``M2``: ``[3, 3, P, P]`` — second-moment matrices (E[y_i y_j]).
          ``M2[i, j] = R_i[:P, :P'] @ R_j[:P', :P]`` via extended basis.

        The matrices R_i are the coordinate multiplication operators
        built analytically from the spherical harmonic recurrence
        relations.  Matrices are computed in float64 for precision and
        cast to the caller's dtype at usage time.  Results are cached
        per ``(L, device)`` pair.
        """
        key = (L, device)
        if key in cls._moment_cache:
            return cls._moment_cache[key]

        P = (L + 1) ** 2
        P1 = (L + 2) ** 2

        # Build in float64 for precision; cast at usage time
        R = _coord_mult_matrices(L + 1, device=device, dtype=torch.float64)

        M1 = R[:, :P, :P].clone()

        # M2[i,j] = R_i[:P,:P1] @ R_j[:P1,:P] through extended basis
        M2 = torch.einsum("iad,jdb->ijab", R[:, :P, :P1], R[:, :P1, :P])

        cls._moment_cache[key] = (M1, M2)
        return M1, M2

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f(y) at points on S².

        Args:
            points: ``[N, 3]`` unit vectors on the sphere.

        Returns:
            ``[B, N]`` log-probability density.
        """
        coeffs = self.coeffs  # [B, K, P]

        # Evaluate SH at all grid points
        Y = real_spherical_harmonics(points, self.L)  # [N, P]

        # g_k(y) for all (batch, point, channel) combinations
        # coeffs: [B, K, P], Y: [N, P]
        # g[b, n, k] = sum_p coeffs[b, k, p] * Y[n, p]
        g = torch.einsum("bkp,np->bnk", coeffs, Y)  # [B, N, K]

        # Sum of squares: [B, N]
        sum_g2 = (g ** 2).sum(dim=-1)  # [B, N]

        # Normalization constant: [B]
        coeffs_norm_sq = (coeffs ** 2).sum(dim=(-1, -2))  # [B]
        Z_eps = self.eps * 4.0 * math.pi + coeffs_norm_sq  # [B]

        # log f = log(eps + sum_g2) - log(Z_eps)
        log_p = torch.log(self.eps + sum_g2) - torch.log(Z_eps).unsqueeze(1)

        return log_p  # [B, N]
