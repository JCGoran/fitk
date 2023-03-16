"""
Tests for the classy interfaces to FITK
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import pytest
from scipy.linalg import block_diag

from fitk import D, FisherDerivative, FisherMatrix
from fitk.interfaces.classy_interfaces import ClassyCMBDerivative
from fitk.utilities import find_diff_weights

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"


def helper(cosmo: FisherDerivative):
    signal = cosmo.signal()
    cov = cosmo.covariance()
    assert np.allclose(cov.T, cov)
    return np.linalg.inv(cov) @ signal


def test_outputs():
    cosmo_all = ClassyCMBDerivative(config={"output": "tCl,pCl"})
    helper(cosmo_all)

    cosmo_t = ClassyCMBDerivative(config={"output": "tCl"})
    helper(cosmo_t)

    cosmo_p = ClassyCMBDerivative(config={"output": "pCl"})
    helper(cosmo_p)


def test_another():
    lmax = 999
    cosmo_lmax = ClassyCMBDerivative(config={"output": "tCl", "l_max_scalars": lmax})

    assert cosmo_lmax.signal().shape == (lmax - 1,)


def test_from_file():
    cosmo = ClassyCMBDerivative.from_file(DATADIR_INPUT / "classy_parameter_file.ini")
    cosmo.signal()
    cosmo.covariance()
