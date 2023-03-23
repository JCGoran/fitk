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
    assert cov.ndim == 2
    assert np.all(np.diag(cov) > 0)
    assert np.all(np.linalg.eigvalsh(cov) >= 0)
    assert signal.shape[0] == cov.shape[0] == cov.shape[1]
    assert np.allclose(cov.T, cov)
    return np.linalg.inv(cov) @ signal


class TestClassy:
    @pytest.mark.parametrize("output", ["tCl,pCl", "tCl", "pCl"])
    def test_outputs(self, output):
        cosmo = ClassyCMBDerivative(config={"output": output})
        helper(cosmo)

    @pytest.mark.parametrize("lmax", [10, 100, 999])
    def test_lmax(self, lmax: int):
        cosmo = ClassyCMBDerivative(config={"output": "tCl", "l_max_scalars": lmax})

        assert cosmo.signal().shape == (lmax - 1,)
        helper(cosmo)

    def test_from_file(self):
        cosmo = ClassyCMBDerivative.from_file(
            DATADIR_INPUT / "classy_parameter_file.ini"
        )
        helper(cosmo)

    @pytest.mark.parametrize(
        "output,benchmark",
        [
            (["tCl", "pCl"], {"temperature": True, "polarization": True}),
            ("tCl,pCl", {"temperature": True, "polarization": True}),
            ("tCl", {"temperature": True, "polarization": False}),
            ("pCl", {"temperature": False, "polarization": True}),
            (["tCl"], {"temperature": True, "polarization": False}),
            (["pCl"], {"temperature": False, "polarization": True}),
        ],
    )
    def test_parse_outputs(self, output, benchmark):
        cosmo = ClassyCMBDerivative(config={"output": output})
        result = cosmo._parse_outputs()
        assert result == benchmark
