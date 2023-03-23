"""
Helper utilities for tests
"""

from __future__ import annotations

import numpy as np

from fitk.derivatives import FisherDerivative


def validate_signal_and_covariance(item: FisherDerivative):
    """
    Check that the signal and covariance have standard properties, that is:

    * they are shape-compatible
    * the covariance is an NxN matrix
    * the covariance has positive eigenvalues
    * the covariance is invertible
    """
    signal = item.signal()
    cov = item.covariance()
    assert np.allclose(cov.T, cov)
    assert cov.ndim == 2
    assert np.all(np.linalg.eigvalsh(cov) >= 0)
    return np.linalg.inv(cov) @ signal
