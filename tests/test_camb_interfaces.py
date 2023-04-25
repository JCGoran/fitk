"""Tests for CAMB interfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from helpers import get_signal_and_covariance, validate_signal_and_covariance

from fitk import D, P
from fitk.interfaces.camb_interfaces import CambCMBDerivative
from fitk.interfaces.classy_interfaces import ClassyCMBDerivative

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"


class TestCMB:
    def test_signal(self):
        cosmo = CambCMBDerivative(
            config={
                "omch2": 0.121203,
                "ombh2": 0.02244,
                "H0": 67,
                "ns": 0.96,
                "omk": 0,
                "As": 2e-9,
                "nnu": 3.046,
            }
        )
        signal = cosmo.signal()

        c = ClassyCMBDerivative(
            config={
                "omega_cdm": 0.121203,
                "omega_b": 0.02244,
                "H0": 67,
                "n_s": 0.96,
                "Omega_k": 0,
                "A_s": 2e-9,
                "N_ur": 3.046,
                "output": "tCl,pCl",
            }
        )
        s = c.signal()

        r = slice(2500, 4000)

        s = s[r]
        signal = signal[r]

        print(signal / s)

        fig, ax = plt.subplots(nrows=2, sharex=True)

        ax[0].plot(signal, label="CAMB")
        ax[0].plot(s, label="CLASS")
        ax[0].set_xscale("log")
        ax[0].legend()
        ax[1].plot(100 * (1 - signal / s))
        fig.savefig("camb_vs_class.pdf", dpi=300, bbox_inches="tight")
