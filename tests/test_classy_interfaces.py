"""Tests for classy interfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitk import D, FisherDerivative, P
from fitk.interfaces.classy_interfaces import ClassyCMBDerivative

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
        cosmo = ClassyCMBDerivative(config={"output": output, "l_max_scalars": 50})
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
        for key in benchmark:
            assert result[key] == benchmark[key]

    @pytest.mark.parametrize("output", ["tCl,pCl", "tCl", "pCl"])
    def test_fisher_matrix(self, output):
        parameters = [
            D(P("Omega_cdm", 0.3, latex_name=r"$\Omega_\mathrm{cdm}$"), 1e-3),
            D(P("Omega_b", 0.04, latex_name=r"$\Omega_\mathrm{b}$"), 1e-3),
            D(P("h", 0.6, latex_name=r"$h$"), 1e-3),
            D(P("n_s", 0.96, latex_name=r"$n_\mathrm{s}$"), 1e-3),
            D(
                P("ln10^{10}A_s", 3.0980, latex_name=r"$\log (10^{10} A_\mathrm{s})$"),
                1e-3,
            ),
        ]
        cosmo = ClassyCMBDerivative(config={"output": output, "l_max_scalars": 100})
        fm = cosmo.fisher_matrix(*parameters)

        assert fm.is_valid()
