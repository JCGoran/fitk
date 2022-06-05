"""
FITK - Fisher Information ToolKit

A Python package for manipulating and plotting Fisher-like objects.

Notes
--------
Througout the documentation, it is assumed that the modules are imported in the
following way:

>>> import numpy as np
>>> from fitk import FisherMatrix, FisherPlotter
"""

__all__ = [
    "FisherMatrix",
    "FisherFigure1D",
    "FisherFigure2D",
    "FisherPlotter",
    "bayes_factor",
    "kl_divergence",
    "plot_curve_1d",
    "plot_curve_2d",
]

from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_operations import bayes_factor, kl_divergence
from fitk.fisher_plotter import (
    FisherFigure1D,
    FisherFigure2D,
    FisherPlotter,
    plot_curve_1d,
    plot_curve_2d,
)
from fitk.fisher_utils import (
    float_to_latex,
    get_index_of_other_array,
    is_positive_semidefinite,
    is_square,
    is_symmetric,
    reindex_array,
)
