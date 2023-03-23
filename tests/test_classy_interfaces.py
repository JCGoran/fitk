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
from helpers import validate_signal_and_covariance
from scipy.linalg import block_diag

from fitk import D, FisherDerivative, FisherMatrix
from fitk.interfaces.classy_interfaces import ClassyCMBDerivative
from fitk.utilities import find_diff_weights

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"


class TestClassy:
    def test_outputs(self):
        cosmo_all = ClassyCMBDerivative(config={"output": "tCl,pCl"})
        validate_signal_and_covariance(cosmo_all)

        cosmo_t = ClassyCMBDerivative(config={"output": "tCl"})
        validate_signal_and_covariance(cosmo_t)

        cosmo_p = ClassyCMBDerivative(config={"output": "pCl"})
        validate_signal_and_covariance(cosmo_p)

    def test_another(self):
        lmax = 999
        cosmo_lmax = ClassyCMBDerivative(
            config={"output": "tCl", "l_max_scalars": lmax}
        )

        assert cosmo_lmax.signal().shape == (lmax - 1,)

    def test_from_file(self):
        cosmo = ClassyCMBDerivative.from_file(
            DATADIR_INPUT / "classy_parameter_file.ini"
        )
        cosmo.signal()
        cosmo.covariance()

    def test_parse_outputs1(self):
        cosmo = ClassyCMBDerivative(config={"output": ["tCl", "pCl"]})
        result = cosmo._parse_outputs()
        assert result == {"temperature": True, "polarization": True}

    def test_parse_outputs2(self):
        cosmo = ClassyCMBDerivative(config={"output": "tCl,pCl"})
        result = cosmo._parse_outputs()
        assert result == {"temperature": True, "polarization": True}

    def test_parse_outputs3(self):
        cosmo = ClassyCMBDerivative(config={"output": "tCl"})
        result = cosmo._parse_outputs()
        assert result == {"temperature": True, "polarization": False}

    def test_parse_outputs4(self):
        cosmo = ClassyCMBDerivative(config={"output": "pCl"})
        result = cosmo._parse_outputs()
        assert result == {"temperature": False, "polarization": True}

    def test_parse_outputs5(self):
        cosmo = ClassyCMBDerivative(config={"output": ["pCl"]})
        result = cosmo._parse_outputs()
        assert result == {"temperature": False, "polarization": True}
