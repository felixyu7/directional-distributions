"""Directional distribution loss functions and evaluation utilities for PyTorch."""

from .vmf import von_mises_fisher_loss, VMF
from .iag import iag_nll_loss, IAG
from .esag import esag_nll_loss, ESAG
from ._base import SphereGrid, make_grid
from ._plotting import plot_mollweide

__all__ = [
    # Loss functions
    "von_mises_fisher_loss",
    "iag_nll_loss",
    "esag_nll_loss",
    # Distribution classes
    "VMF",
    "IAG",
    "ESAG",
    # Grid utilities
    "SphereGrid",
    "make_grid",
    # Plotting
    "plot_mollweide",
]
