"""
Tests for the various third-party interfaces to FITK
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
from scipy.linalg import block_diag

try:
    import coffe
except Exception:
    pass
from fitk import D, FisherMatrix
from fitk.fisher_utils import find_diff_weights
from fitk.interfaces.coffe_interfaces import CoffeMultipolesDerivative

COFFE_SETTINGS = {
    "omega_m": 0.32,
    "omega_baryon": 0.05,
    "h": 0.67,
    "z_mean": [1.0],
    "deltaz": [0.1],
    "sep": np.linspace(10, 110, 5),
    "l": [0],
    "number_density1": [1e-3],
    "number_density2": [2e-3],
    "fsky": [0.2],
    "pixelsize": [5],
    "has_density": True,
    "has_rsd": True,
}
COFFE_REDSHIFTS = np.array([0, 0.5, 1, 1.5, 2])
COFFE_GALAXY_BIAS1 = np.array([np.sqrt(1 + _) for _ in COFFE_REDSHIFTS])
COFFE_GALAXY_BIAS2 = np.array([np.sqrt(1 + _ / 2) for _ in COFFE_REDSHIFTS])


def _split_array(item, index: int):
    """
    Splits an array of size `n` at indices `index` and returns the result
    """
    return np.array([item[_ : _ + index] for _ in range(0, len(item), index)])


class TestCoffeInterfaces:
    """
    Tests for the interfaces of the COFFE code
    """

    def test_signal_multipoles(self):
        """
        Testing multipoles of COFFE
        """
        cosmo = coffe.Coffe(**COFFE_SETTINGS)
        cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
        cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)

        benchmark = np.array([_.value for _ in cosmo.compute_multipoles_bulk()])

        d = CoffeMultipolesDerivative(
            config={
                "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                **COFFE_SETTINGS,
            }
        )

        result = d.signal()

        assert np.allclose(benchmark, result, rtol=1e-3)

    def test_signal_multipoles_derivative(self):
        """
        Testing the derivatives of the multipoles of COFFE
        """
        parameters = [
            D(name=_, fiducial=COFFE_SETTINGS[_], abs_step=1e-3, accuracy=2)
            for _ in [
                "omega_m",
            ]
        ]
        for parameter in parameters:
            weights = find_diff_weights(parameter.stencil)
            benchmarks: list[Collection[float]] = []
            for value in parameter.fiducial + parameter.stencil * parameter.abs_step:

                cosmo = coffe.Coffe(**{**COFFE_SETTINGS, parameter.name: value})
                cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
                cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)

                benchmarks.append(
                    np.array([_.value for _ in cosmo.compute_multipoles_bulk()])
                )

            benchmark = np.sum(
                [b * w / parameter.abs_step for b, w in zip(benchmarks, weights)],
                axis=0,
            )

            d = CoffeMultipolesDerivative(
                config={
                    "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                    "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                    **COFFE_SETTINGS,
                }
            )

            result = d.derivative("signal", parameter)

            assert np.allclose(benchmark, result, rtol=1e-3)

    def test_covariance_multipoles(self):
        """
        Testing the covariance of multipoles of COFFE
        """
        cosmo = coffe.Coffe(**COFFE_SETTINGS)
        cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
        cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)

        covariance = cosmo.compute_covariance_bulk()
        covariance_per_bin = _split_array(
            [_.value for _ in covariance],
            (len(cosmo.sep) * len(cosmo.l)) ** 2,
        )

        benchmark = block_diag(
            *[
                np.reshape(
                    _,
                    (len(cosmo.sep) * len(cosmo.l), len(cosmo.sep) * len(cosmo.l)),
                )
                for _ in covariance_per_bin
            ]
        )

        d = CoffeMultipolesDerivative(
            config={
                "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                **COFFE_SETTINGS,
            }
        )

        result = d.covariance()

        assert np.allclose(benchmark, result, rtol=1e-3)

    def test_covariance_multipoles_derivative(self):
        """
        Testing the derivatives of the covariance of multipoles of COFFE
        """
        parameters = [
            D(name=_, fiducial=COFFE_SETTINGS[_], abs_step=1e-3, accuracy=2)
            for _ in [
                "omega_m",
            ]
        ]
        for parameter in parameters:
            weights = find_diff_weights(parameter.stencil)
            benchmarks: list[Collection[float]] = []
            for value in parameter.fiducial + parameter.stencil * parameter.abs_step:

                cosmo = coffe.Coffe(**{**COFFE_SETTINGS, parameter.name: value})
                cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
                cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)

                covariance = cosmo.compute_covariance_bulk()
                covariance_per_bin = _split_array(
                    [_.value for _ in covariance],
                    (len(cosmo.sep) * len(cosmo.l)) ** 2,
                )

                benchmark = block_diag(
                    *[
                        np.reshape(
                            _,
                            (
                                len(cosmo.sep) * len(cosmo.l),
                                len(cosmo.sep) * len(cosmo.l),
                            ),
                        )
                        for _ in covariance_per_bin
                    ]
                )

                benchmarks.append(benchmark)

            benchmark = np.sum(
                [b * w / parameter.abs_step for b, w in zip(benchmarks, weights)],
                axis=0,
            )

            d = CoffeMultipolesDerivative(
                config={
                    "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                    "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                    **COFFE_SETTINGS,
                }
            )

            result = d.derivative("covariance", parameter)

            assert np.allclose(benchmark, result, rtol=1e-3)

    def test_fisher_matrix(self):
        """
        Tests the full Fisher matrix obtained using multipoles from COFFE
        """
        parameters = [
            D(name=_, fiducial=COFFE_SETTINGS[_], abs_step=1e-3, accuracy=2)
            for _ in [
                "omega_m",
                "h",
            ]
        ]

        derivative: dict[str, Any] = {}

        cosmo = coffe.Coffe(**COFFE_SETTINGS)
        cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
        cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)

        cov = cosmo.compute_covariance_bulk()
        covariance_per_bin = _split_array(
            [_.value for _ in cov],
            (len(cosmo.sep) * len(cosmo.l)) ** 2,
        )

        covariance = block_diag(
            *[
                np.reshape(
                    _,
                    (len(cosmo.sep) * len(cosmo.l), len(cosmo.sep) * len(cosmo.l)),
                )
                for _ in covariance_per_bin
            ]
        )

        for parameter in parameters:
            weights = find_diff_weights(parameter.stencil)
            benchmarks: list[Collection[float]] = []
            for value in parameter.fiducial + parameter.stencil * parameter.abs_step:

                cosmo = coffe.Coffe(**{**COFFE_SETTINGS, parameter.name: value})
                cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
                cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)

                benchmarks.append(
                    np.array([_.value for _ in cosmo.compute_multipoles_bulk()])
                )

            derivative[parameter.name] = np.sum(
                [b * w / parameter.abs_step for b, w in zip(benchmarks, weights)],
                axis=0,
            )

        fisher_matrix = [
            value1 @ np.linalg.inv(covariance) @ value2
            for value1 in derivative.values()
            for value2 in derivative.values()
        ]

        d = CoffeMultipolesDerivative(
            config={
                "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                **COFFE_SETTINGS,
            }
        )

        result = d.fisher_matrix(*parameters, parameter_dependence="signal")

        assert result == FisherMatrix(
            np.array(fisher_matrix).reshape(len(parameters), len(parameters)),
            names=[_.name for _ in parameters],
            fiducials=[_.fiducial for _ in parameters],
        )
