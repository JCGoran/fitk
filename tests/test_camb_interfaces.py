"""Tests for CAMB interfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from helpers import get_signal_and_covariance, validate_signal_and_covariance

from fitk import D, FisherFigure2D, P
from fitk.interfaces.camb_interfaces import CambCMBDerivative
from fitk.interfaces.classy_interfaces import ClassyCMBDerivative

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"


class TestCMB:
    @pytest.mark.parametrize("lmax", [100, 999])
    def test_lmax(self, lmax: int):
        cosmo = CambCMBDerivative(
            config={
                "omch2": 0.121203,
                "H0": 67,
                "max_l": lmax,
            }
        )
        signal, cov = get_signal_and_covariance(cosmo)

        # the 3 is here because CAMB outputs TT, EE, and TE by default
        assert signal.shape == (3 * (lmax - 1),)
        validate_signal_and_covariance(signal, cov)

    def test_fisher_matrix(self):
        parameters = [
            D(P("omch2", 0.27 * 0.67**2, latex_name=r"$\omega_\mathrm{cdm}$"), 1e-3),
            D(P("ombh2", 0.04 * 0.67**2, latex_name=r"$\omega_\mathrm{b}$"), 1e-3),
            D(P("H0", 60, latex_name=r"$H_0$"), 1e-3),
        ]
        cosmo = CambCMBDerivative(
            config={
                "max_l": 100,
                **{_.parameter.name: _.parameter.fiducial for _ in parameters},
            }
        )
        fm = cosmo.fisher_matrix(*parameters)

        assert fm.is_valid()
