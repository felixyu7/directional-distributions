"""Real spherical harmonic evaluation on S² in PyTorch.

Provides a single function :func:`real_spherical_harmonics` that evaluates
orthonormal real SH up to degree *L* at batches of Cartesian unit vectors.
The implementation uses associated Legendre polynomial three-term recurrence
and is fully differentiable through PyTorch autograd.

Convention
----------
- **Orthonormal** real SH with Condon-Shortley phase.
- Flat index ``p = l² + l + m`` for ``m ∈ [-l, l]``, giving ``(L+1)²`` total
  basis functions for maximum degree *L*.
- Satisfies ``∫_{S²} Y_l^m(y) Y_{l'}^{m'}(y) dω(y) = δ_{ll'} δ_{mm'}``.

Stability
---------
The associated Legendre recurrence is numerically stable for ``L ≤ ~20``
with standard float64 factorial pre-computation.  For higher degrees a
log-space or scaled recurrence would be needed (not implemented here).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def real_spherical_harmonics(points: Tensor, L: int) -> Tensor:
    """Evaluate real spherical harmonics up to degree *L*.

    Args:
        points: ``[N, 3]`` unit vectors in Cartesian coordinates ``(x, y, z)``.
        L: Maximum spherical harmonic degree (``L >= 0``).

    Returns:
        ``[N, (L+1)²]`` tensor of real SH values at each point.
    """
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]  # cos(theta)

    N = points.shape[0]
    P = (L + 1) ** 2
    Y = torch.zeros(N, P, dtype=points.dtype, device=points.device)

    # sin(theta) = sqrt(x² + y²), with clamp to avoid NaN gradient at poles
    # (d/dx sqrt(x²+y²) is undefined at x=y=0; clamping makes grad = 0 there)
    rho_sq = x * x + y * y
    rho = torch.sqrt(rho_sq.clamp(min=1e-20))

    # Azimuth angle with pole guard to prevent NaN gradients from atan2(0, 0)
    phi = torch.where(rho_sq < 1e-20, torch.zeros_like(x), torch.atan2(y, x))

    # ----- Associated Legendre polynomials P_l^m(z) for m >= 0 -----
    # We store P_l^m values in a list-of-lists indexed by [m][l - m].
    # The recurrence is run for each m separately, vectorized over N points.

    # plm[m] is a list of tensors: [P_m^m, P_{m+1}^m, ..., P_L^m]
    plm: list[list[Tensor]] = []

    for m in range(L + 1):
        col: list[Tensor] = []

        if m == 0:
            # P_0^0 = 1
            pmm = torch.ones(N, dtype=points.dtype, device=points.device)
        else:
            # Diagonal seed: P_m^m = (-1)^m (2m-1)!! sin_theta^m
            # Build incrementally to avoid computing large double factorials
            # P_m^m = -(2m-1) * sin_theta * P_{m-1}^{m-1}
            pmm = -(2 * m - 1) * rho * plm[m - 1][0]

        col.append(pmm)

        if m < L:
            # First off-diagonal: P_{m+1}^m = z (2m+1) P_m^m
            pm1m = z * (2 * m + 1) * pmm
            col.append(pm1m)

            # General three-term recurrence for l = m+2 .. L
            for l in range(m + 2, L + 1):
                pl = ((2 * l - 1) * z * col[-1] - (l + m - 1) * col[-2]) / (l - m)
                col.append(pl)

        plm.append(col)

    # ----- Assemble real SH from P_l^m and trig functions -----

    for l in range(L + 1):
        for m in range(-l, l + 1):
            p_idx = l * l + l + m  # flat index
            abs_m = abs(m)

            # Normalization constant
            norm = math.sqrt(
                (2 * l + 1) / (4 * math.pi)
                * math.factorial(l - abs_m)
                / math.factorial(l + abs_m)
            )

            # Associated Legendre value: plm[abs_m][l - abs_m]
            p_val = plm[abs_m][l - abs_m]

            if m > 0:
                Y[:, p_idx] = math.sqrt(2) * norm * p_val * torch.cos(m * phi)
            elif m == 0:
                Y[:, p_idx] = norm * p_val
            else:  # m < 0
                Y[:, p_idx] = math.sqrt(2) * norm * p_val * torch.sin(abs_m * phi)

    return Y
