"""
FITK - Fisher Information ToolKit, version $VERSION

A Python package for computing, manipulating, and plotting Fisher-like
objects.

Notes
-----
Througout the documentation, it is assumed that the modules are imported
in the following way:

>>> import numpy as np
>>> from fitk import FisherMatrix, FisherFigure1D, FisherFigure2D, D, FisherDerivative
"""

from pathlib import Path

__version__ = (
    (Path(__file__).resolve().parent / "VERSION.txt")
    .read_text(encoding="utf-8")
    .strip()
)

from fitk import fisher_operations, fisher_utils, interfaces
from fitk.fisher_derivative import D, FisherDerivative
from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_operations import bayes_factor, kl_divergence, kl_matrix
from fitk.fisher_plotter import FisherFigure1D, FisherFigure2D
