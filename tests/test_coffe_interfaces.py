"""Tests for COFFE interfaces."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pytest
from scipy.linalg import block_diag

try:
    import coffe
except ImportError:
    pass
from fitk import D, FisherMatrix
from fitk.interfaces.coffe_interfaces import (
    CoffeMultipolesBiasDerivative,
    CoffeMultipolesDerivative,
    CoffeMultipolesTildeDerivative,
)
from fitk.utilities import P, find_diff_weights

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
    "has_flatsky_local": True,
    "N_ncdm": 0,
    "A_s": 2.2e-09,
}
COFFE_REDSHIFTS = np.array([0, 0.5, 1, 1.5, 2])
COFFE_GALAXY_BIAS1 = np.array([np.sqrt(1 + _) for _ in COFFE_REDSHIFTS])
COFFE_GALAXY_BIAS2 = np.array([np.sqrt(1 + _ / 2) for _ in COFFE_REDSHIFTS])


def _split_array(item, index: int):
    """
    Splits an array of size `n` at indices `index` and returns the result
    """
    return np.array([item[_ : _ + index] for _ in range(0, len(item), index)])


class TestMultipolesInterface:
    """Tests for the `CoffeMultipolesDerivative` class."""

    def test_signal(self):
        """Test multipoles of COFFE."""
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

    def test_signal_derivative(self):
        """Test the derivatives of the multipoles of COFFE."""
        parameters = [
            D(P(name=_, fiducial=COFFE_SETTINGS[_]), abs_step=1e-3, accuracy=2)
            for _ in [
                "omega_m",
            ]
        ]
        for parameter in parameters:
            weights = find_diff_weights(parameter.stencil)
            benchmarks: list[Collection[float]] = []
            for value in (
                parameter.parameter.fiducial + parameter.stencil * parameter.abs_step
            ):
                cosmo = coffe.Coffe(
                    **{**COFFE_SETTINGS, parameter.parameter.name: value}
                )
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

    def test_covariance(self):
        """Test the covariance of multipoles of COFFE."""
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

    def test_covariance_derivative(self):
        """Test the derivatives of the covariance of multipoles of COFFE."""
        parameters = [
            D(P(name=_, fiducial=COFFE_SETTINGS[_]), abs_step=1e-3, accuracy=2)
            for _ in [
                "omega_m",
            ]
        ]
        for parameter in parameters:
            weights = find_diff_weights(parameter.stencil)
            benchmarks: list[Collection[float]] = []
            for value in (
                parameter.parameter.fiducial + parameter.stencil * parameter.abs_step
            ):
                cosmo = coffe.Coffe(
                    **{**COFFE_SETTINGS, parameter.parameter.name: value}
                )
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
        """Test the full Fisher matrix obtained using multipoles from COFFE."""
        parameters = [
            D(P(name=_, fiducial=COFFE_SETTINGS[_]), abs_step=1e-3, accuracy=2)
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
            for value in (
                parameter.parameter.fiducial + parameter.stencil * parameter.abs_step
            ):
                cosmo = coffe.Coffe(
                    **{**COFFE_SETTINGS, parameter.parameter.name: value}
                )
                cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
                cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)

                benchmarks.append(
                    np.array([_.value for _ in cosmo.compute_multipoles_bulk()])
                )

            derivative[parameter.parameter.name] = np.sum(
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
            names=[_.parameter.name for _ in parameters],
            fiducials=[_.parameter.fiducial for _ in parameters],
        )


class TestTildeInterface:
    """Tests for the `CoffeMultipolesTildeDerivative` class."""

    @staticmethod
    def monopole_derivative_f(b: float, f: float, cosmo):
        return (
            (2 * b / 3 + 2 * f / 5)
            * np.array([cosmo.integral(r=_, l=0, n=0) for _ in cosmo.sep])
            / cosmo.sigma8**2
        )

    @staticmethod
    def monopole_derivative_b(b: float, f: float, cosmo):
        return (
            (2 * b + 2 * f / 3)
            * np.array([cosmo.integral(r=_, l=0, n=0) for _ in cosmo.sep])
            / cosmo.sigma8**2
        )

    @staticmethod
    def quadrupole_derivative_f(b: float, f: float, cosmo):
        return (
            -(4 * b / 3 + 8 * f / 7)
            * np.array([cosmo.integral(r=_, l=2, n=0) for _ in cosmo.sep])
            / cosmo.sigma8**2
        )

    @staticmethod
    def quadrupole_derivative_b(b: float, f: float, cosmo):
        return (
            -(4 * f / 3)
            * np.array([cosmo.integral(r=_, l=2, n=0) for _ in cosmo.sep])
            / cosmo.sigma8**2
        )

    @staticmethod
    def hexadecapole_derivative_f(b: float, f: float, cosmo):
        return (
            (16 * f / 35)
            * np.array([cosmo.integral(r=_, l=4, n=0) for _ in cosmo.sep])
            / cosmo.sigma8**2
        )

    @staticmethod
    def hexadecapole_derivative_b(b: float, f: float, cosmo):
        return [0 for _ in cosmo.sep]

    def test_derivative(self):
        """Test the `derivative` method."""
        multipoles = [0, 2, 4]
        z_mean = [1, 1.2, 1.5]
        deltaz = [0.1, 0.1, 0.1]
        cosmo = coffe.Coffe(**COFFE_SETTINGS)
        cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
        cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)
        cosmo.l = multipoles
        cosmo.z_mean = z_mean
        cosmo.deltaz = deltaz

        derivative = CoffeMultipolesTildeDerivative(
            config={
                "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                **COFFE_SETTINGS,
                "l": multipoles,
                "z_mean": z_mean,
                "deltaz": deltaz,
            }
        )

        for index, redshift in enumerate(cosmo.z_mean):
            for name, value in zip(
                ["b", "f"],
                [
                    cosmo.galaxy_bias1(redshift)
                    * cosmo.growth_factor(redshift)
                    * cosmo.sigma8,
                    cosmo.growth_rate(redshift)
                    * cosmo.growth_factor(redshift)
                    * cosmo.sigma8,
                ],
            ):
                result = derivative.derivative(
                    "signal",
                    D(
                        P(
                            f"{name}{index + 1}",
                            value,
                            rf"${name}_{index + 1}",
                        ),
                        abs_step=1e-4,
                        accuracy=2,
                    ),
                    rounding_threshold=1e-10,
                )

                benchmark = np.concatenate(
                    [
                        getattr(self, f"{multipole}_derivative_{name}")(
                            cosmo.galaxy_bias1(redshift)
                            * cosmo.growth_factor(redshift)
                            * cosmo.sigma8,
                            cosmo.growth_rate(redshift)
                            * cosmo.growth_factor(redshift)
                            * cosmo.sigma8,
                            cosmo,
                        )
                        for multipole in ["monopole", "quadrupole", "hexadecapole"]
                    ]
                )

                assert np.allclose(
                    result[np.nonzero(result)],
                    benchmark[np.nonzero(benchmark)],
                    rtol=1e-3,
                )


class TestBiasInterface:
    """Tests for the `CoffeMultipolesBiasDerivative` class."""

    def test_derivative(self):
        """Test the `derivative` method."""
        multipoles = [0, 2]
        z_mean = [1, 1.2, 1.5]
        deltaz = [0.1, 0.1, 0.1]
        cosmo = coffe.Coffe(**COFFE_SETTINGS)
        cosmo.set_galaxy_bias1(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1)
        cosmo.set_galaxy_bias2(COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2)
        cosmo.l = multipoles
        cosmo.z_mean = z_mean
        cosmo.deltaz = deltaz

        derivative = CoffeMultipolesBiasDerivative(
            config={
                "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                **COFFE_SETTINGS,
                "l": multipoles,
                "z_mean": z_mean,
                "deltaz": deltaz,
            }
        )

        for index, redshift in enumerate(cosmo.z_mean):
            result = derivative.derivative(
                "signal",
                D(
                    P(
                        f"b{index + 1}",
                        cosmo.galaxy_bias1(redshift),
                        latex_name=rf"$b_{index + 1}$",
                    ),
                    abs_step=1e-4,
                    accuracy=2,
                ),
                rounding_threshold=1e-10,
            )

            benchmark = np.concatenate(
                [
                    cosmo.growth_factor(redshift) ** 2
                    * (
                        2 * cosmo.galaxy_bias1(redshift)
                        + 2 * cosmo.growth_rate(redshift) / 3
                    )
                    * np.array([cosmo.integral(r=_, l=0, n=0) for _ in cosmo.sep]),
                    -cosmo.growth_factor(redshift) ** 2
                    * (4 * cosmo.growth_rate(redshift) / 3)
                    * np.array([cosmo.integral(r=_, l=2, n=0) for _ in cosmo.sep]),
                ]
            )

            assert np.allclose(
                result[np.nonzero(result)],
                benchmark,
                rtol=1e-3,
            )

    def test_warning_flatsky_local(self):
        """
        Test local flatsky warning.

        Check that the code shows a warning when not using the flat-sky
        approximation (for local terms).
        """
        derivative = CoffeMultipolesBiasDerivative(
            config={
                "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                **COFFE_SETTINGS,
                "has_flatsky_local": False,
            }
        )

        with pytest.warns(UserWarning):
            derivative.signal()

    def test_warning_flatsky_nonlocal(self):
        """
        Test nonlocal flatsky warning.

        Check that the code shows a warning when not using the flat-sky
        approximation (for non-local terms).
        """

        derivative = CoffeMultipolesBiasDerivative(
            config={
                "galaxy_bias1": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS1),
                "galaxy_bias2": (COFFE_REDSHIFTS, COFFE_GALAXY_BIAS2),
                **COFFE_SETTINGS,
                "has_lensing": True,
                "has_flatsky_nonlocal": False,
                "has_density": False,
                "has_rsd": False,
            }
        )

        with pytest.warns(UserWarning):
            derivative.signal()
