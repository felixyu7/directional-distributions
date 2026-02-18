"""Mollweide projection plotting for directional distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor

    from ._base import SphereGrid


def plot_mollweide(
    grid: SphereGrid,
    values: Tensor | np.ndarray,
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    **pcolormesh_kwargs,
) -> tuple[Figure, Axes]:
    """Plot a scalar field on S² using a Mollweide projection.

    Args:
        grid: SphereGrid from :func:`make_grid`.
        values: [n_lat, n_lon] array of scalar values (e.g. PDF values).
        ax: Optional Matplotlib axes with ``projection='mollweide'``.
            Created automatically if *None*.
        cmap: Colormap name.
        colorbar: Whether to add a colorbar.
        **pcolormesh_kwargs: Forwarded to ``ax.pcolormesh``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    import matplotlib.pyplot as plt

    n_lat, n_lon = grid.shape

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    else:
        fig = ax.figure

    # Convert to numpy
    lon = grid.lon.detach().cpu().numpy().reshape(n_lat, n_lon)
    lat = grid.lat.detach().cpu().numpy().reshape(n_lat, n_lon)
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    vals = np.asarray(values).reshape(n_lat, n_lon)

    mesh = ax.pcolormesh(lon, lat, vals, cmap=cmap, shading="auto", **pcolormesh_kwargs)
    if colorbar:
        fig.colorbar(mesh, ax=ax, shrink=0.6)

    ax.grid(True, alpha=0.3)

    return fig, ax
