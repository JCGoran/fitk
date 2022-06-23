"""
FITK - Fisher Information ToolKit

A Python package for computing, manipulating, and plotting Fisher-like
objects.

Notes
--------
Througout the documentation, it is assumed that the modules are imported
in the following way:

>>> import numpy as np
>>> from fitk import FisherMatrix, FisherPlotter, D, FisherDerivative
"""

from fitk import fisher_operations, fisher_utils, interfaces
from fitk.fisher_derivative import D, FisherDerivative
from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_operations import bayes_factor, kl_divergence, kl_matrix
from fitk.fisher_plotter import FisherPlotter
