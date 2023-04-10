"""Helper functions for testing."""

from __future__ import annotations

import numpy as np

from fitk import FisherDerivative


def get_signal_and_covariance(item: FisherDerivative):
    signal = item.signal()
    cov = item.covariance()
    return signal, cov


def validate_covariance(cov):
    assert cov.ndim == 2
    assert np.all(np.diag(cov) > 0)
    assert np.all(np.linalg.eigvalsh(cov) >= 0)
    assert np.allclose(cov.T, cov)


def validate_signal_and_covariance(signal, cov):
    validate_covariance(cov)
    assert signal.shape[0] == cov.shape[0] == cov.shape[1]
    return np.linalg.inv(cov) @ signal
