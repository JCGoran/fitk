"""Tests for classy interfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import get_signal_and_covariance, validate_signal_and_covariance

from fitk import D, P
from fitk.interfaces.classy_interfaces import (
    ClassyCMBDerivative,
    ClassyGalaxyCountsDerivative,
)

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"


class TestCMB:
    @pytest.mark.parametrize(
        "output,benchmark",
        [
            ({}, "tCl"),
            ({"output": "tCl,pCl"}, "tCl,pCl"),
            ({"output": "tCl"}, "tCl"),
            ({"output": "pCl"}, "pCl"),
        ],
    )
    def test_outputs(self, output, benchmark):
        cosmo = ClassyCMBDerivative(config={"l_max_scalars": 50, **output})
        assert cosmo.config["output"] == benchmark
        signal, cov = get_signal_and_covariance(cosmo)
        validate_signal_and_covariance(signal, cov)

    @pytest.mark.parametrize("lmax", [10, 100, 999])
    def test_lmax(self, lmax: int):
        cosmo = ClassyCMBDerivative(config={"output": "tCl", "l_max_scalars": lmax})
        signal, cov = get_signal_and_covariance(cosmo)

        assert signal.shape == (lmax - 1,)
        validate_signal_and_covariance(signal, cov)

    def test_from_file(self):
        cosmo = ClassyCMBDerivative.from_file(
            DATADIR_INPUT / "classy_parameter_file.ini"
        )
        signal, cov = get_signal_and_covariance(cosmo)
        validate_signal_and_covariance(signal, cov)

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
        cosmo = ClassyCMBDerivative(
            config={
                "output": output,
                "l_max_scalars": 100,
                **{_.parameter.name: _.parameter.fiducial for _ in parameters},
            }
        )
        fm = cosmo.fisher_matrix(*parameters)

        assert fm.is_valid()


class TestGalaxyCounts:
    @pytest.mark.parametrize(
        "output,benchmark",
        [
            ({}, "nCl"),
            ({"output": "nCl"}, "nCl"),
        ],
    )
    def test_outputs(self, output, benchmark):
        cosmo = ClassyGalaxyCountsDerivative(config={"l_max_lss": 50, **output})
        assert cosmo.config["output"] == benchmark
        signal, cov = get_signal_and_covariance(cosmo)
        validate_signal_and_covariance(signal, cov)

    @pytest.mark.parametrize(
        "config",
        [
            {"selection_mean": "0.1"},
            {"selection_mean": "0.1, 0.5", "non_diagonal": 0},
            pytest.param(
                {"selection_mean": "0.1, 0.5", "non_diagonal": 1},
                marks=pytest.mark.xfail(reason="The covariance matrix is singular"),
            ),
        ],
    )
    def test_signal_and_covariance(self, config):
        cosmo = ClassyGalaxyCountsDerivative(
            config={
                "l_max_lss": 3,
                **config,
            }
        )
        signal, cov = get_signal_and_covariance(cosmo)
        validate_signal_and_covariance(signal, cov)

    @pytest.mark.parametrize(
        "config",
        [
            {"selection_mean": "0.1"},
            {"selection_mean": "0.1, 0.5", "non_diagonal": 0},
            {"selection_mean": "0.1, 0.5", "non_diagonal": 1},
        ],
    )
    def test_fisher_matrix(self, config, l_max: int = 100):
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

        cosmo = ClassyGalaxyCountsDerivative(
            config={
                "l_max_lss": l_max,
                **config,
                **{_.parameter.name: _.parameter.fiducial for _ in parameters},
            }
        )

        fm = cosmo.fisher_matrix(*parameters, use_pinv=True)

        assert fm.is_valid()
