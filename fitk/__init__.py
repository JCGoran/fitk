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

from fitk.fisher_utils import \
    float_to_latex, \
    get_index_of_other_array, \
    is_positive_semidefinite, \
    is_square, \
    is_symmetric, \
    reindex_array
from fitk.fisher_matrix import \
    FisherMatrix
from fitk.fisher_plotter import \
    FisherPlotter, \
    FisherFigure1D, \
    FisherFigure2D
