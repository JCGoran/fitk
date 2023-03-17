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
from fitk.interfaces.classy_interfaces import (
    ClassyCMBDerivative,
    ClassyGalaxyCountsDerivative,
)
from fitk.utilities import find_diff_weights

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"


def helper(cosmo: FisherDerivative):
    signal = cosmo.signal()
    cov = cosmo.covariance()
    assert np.allclose(cov.T, cov)
    assert cov.ndim == 2
    assert np.all(np.linalg.eigvalsh(cov) >= 0)
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


def test_derivative():
    cosmo = ClassyCMBDerivative.from_file(DATADIR_INPUT / "classy_parameter_file.ini")
    cosmo.derivative("signal", D(name="Omega_cdm", fiducial=0.3, abs_step=1e-3))


def test_fisher_matrix():
    parameters = [
        D("Omega_cdm", 0.3, 1e-3, latex_name=r"$\Omega_\mathrm{cdm}$"),
        D("Omega_b", 0.04, 1e-3, latex_name=r"$\Omega_\mathrm{b}$"),
        D("h", 0.6, 1e-3, latex_name=r"$h$"),
        D("n_s", 0.96, 1e-3, latex_name=r"$n_\mathrm{s}$"),
        D("ln10^{10}A_s", 3.0980, 1e-3, latex_name=r"$\log (10^{10} A_\mathrm{s})$"),
    ]
    cosmo_t = ClassyCMBDerivative(config={"output": "tCl"})
    fm_t = cosmo_t.fisher_matrix(*parameters)


def test_galaxy_counts():
    cosmo = ClassyGalaxyCountsDerivative(
        config={
            "output": "nCl",
            "selection_mean": "0.1, 0.5",
            "non_diagonal": 1,
            "l_max_lss": 3,
        }
    )
    helper(cosmo)
