"""
Tests for derivatives using FITK
"""

from __future__ import annotations

# standard library imports
from itertools import product

# third party imports
import numpy as np
import pytest

# first party imports
from fitk.derivatives import D, FisherDerivative, matrix_element_from_input
from fitk.interfaces.misc_interfaces import SupernovaDerivative
from fitk.utilities import ValidationError, _expansion_coefficient


def test_expansion_coefficient():
    assert np.allclose(_expansion_coefficient(1, 1), 1 / 2)
    assert np.allclose(_expansion_coefficient(2, 1), 1 / 2)
    assert np.allclose(_expansion_coefficient(2, 2), 1 / 8)
    assert np.allclose(_expansion_coefficient(1, 3), 1 / 6)
    assert np.allclose(_expansion_coefficient(3, 2), 1 / 12)
    assert np.allclose(_expansion_coefficient(3, 3), 1 / 72)


class TestMatrixElementFromInput:
    """
    Tests for `matrix_element_from_input`
    """

    def test_no_inputs(self):
        """
        Check that it raises an error when we only specify the covariance and
        no derivatives.
        """
        with pytest.raises(ValueError):
            matrix_element_from_input(
                np.diag([1, 2]),
            )

    def test_only_one_input(self):
        """
        Check that it raises an error if we only specify one derivative and not
        the other.
        """
        covariance = np.linalg.inv(np.diag([1, 2]))
        signal_derivative = np.array([1, 2, 3])
        covariance_derivative = np.array([[1, 2], [2, 1]])

        # only one signal derivative
        with pytest.raises(ValueError):
            matrix_element_from_input(
                covariance,
                signal_derivative1=signal_derivative,
            )

        with pytest.raises(ValueError):
            matrix_element_from_input(
                covariance,
                signal_derivative1=signal_derivative,
                covariance_derivative1=covariance_derivative,
                covariance_derivative2=covariance_derivative,
            )

        # only one covariance derivative
        with pytest.raises(ValueError):
            matrix_element_from_input(
                covariance,
                covariance_derivative1=covariance_derivative,
            )

        with pytest.raises(ValueError):
            matrix_element_from_input(
                covariance,
                signal_derivative1=signal_derivative,
                signal_derivative2=signal_derivative,
                covariance_derivative1=covariance_derivative,
            )

    def test_mismatching_sizes(self):
        """
        Check that it raises an error if the input has mismatching size
        """
        covariance = np.linalg.inv(np.diag([1, 2]))
        signal_derivative = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            matrix_element_from_input(
                covariance,
                signal_derivative1=signal_derivative,
                signal_derivative2=signal_derivative,
            )


class LinearDerivative(FisherDerivative):
    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        for arg in args:
            name, value = arg
            if name == "x":
                x = value

        return np.array([x])


class GaussianDerivative(FisherDerivative):
    def __init__(self, config=None):
        self.config = config if config is not None else {"mu": 1.0, "sigma": 1.0}

    def validate_parameter(self, arg: D):
        if arg.name not in ["mu", "sigma"]:
            return False

        return True

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        mu = self.config["mu"]
        sigma = self.config["sigma"]

        for name, value in args:
            if name == "mu":
                mu = value
            elif name == "sigma":
                sigma = value
            else:
                raise ValueError

        result = np.exp(-(mu**2 / 2 / sigma**2)) / sigma / np.sqrt(2 * np.pi)

        return np.array([result])

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        mu = self.config["mu"]
        sigma = self.config["sigma"]

        for name, value in args:
            if name == "mu":
                mu = value
            elif name == "sigma":
                sigma = value
            else:
                raise ValueError

        result = np.diag([1])

        return result

    def first_derivative_wrt_mu(
        self,
        mu: float,
        sigma: float,
    ):
        return -self.signal(("mu", mu), ("sigma", sigma)) * mu / sigma**2

    def second_derivative_wrt_mu(
        self,
        mu: float,
        sigma: float,
    ):
        return (
            self.signal(("mu", mu), ("sigma", sigma))
            * (mu**2 - sigma**2)
            / sigma**4
        )

    def first_derivative_wrt_sigma(
        self,
        mu: float,
        sigma: float,
    ):
        return (
            self.signal(("mu", mu), ("sigma", sigma))
            * (mu**2 - sigma**2)
            / sigma**3
        )

    def mixed_derivative(
        self,
        mu: float,
        sigma: float,
    ):
        return (
            -self.signal(("mu", mu), ("sigma", sigma))
            * mu
            * (mu**2 - 3 * sigma**2)
            / sigma**5
        )


class TestFisherDerivative:
    """
    Tests for Fisher derivative
    """

    def test_D(self):
        # unknown kind of difference
        with pytest.raises(ValueError):
            D("a", 1, 1e-3, kind="asdf")

        # abs_step < 0
        with pytest.raises(ValueError):
            D("a", 1, -1e-3)

        # accuracy not at least first order
        with pytest.raises(ValueError):
            D("a", 1, 1e-3, accuracy=0)

        # stencil is not an iterable
        with pytest.raises(TypeError):
            D("a", 1, 1e-3, stencil=1)

        # stencil is not in increasing order
        with pytest.raises(ValueError):
            D("a", 1, 1e-3, stencil=[0, -1, 3, 4])

        # valid stencil
        D("a", 1, 1e-3, stencil=[-3, -1, 0, 2])

    def test_abc(self):
        # we cannot use `FisherDerivative` class (but can instantiate it)
        fd = FisherDerivative()
        with pytest.raises(NotImplementedError):
            fd.derivative("signal", D("omega_m", 0.32, abs_step=1e-3))

    def test_first_derivative(self):
        lin = LinearDerivative()

        lin.derivative("signal", D("x", 1, 1e-3))

        g = GaussianDerivative({"mu": 1, "sigma": 1})

        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g.derivative("signal", D(name="mu", fiducial=value, abs_step=1e-4)),
                g.first_derivative_wrt_mu(value, 1),
            )

        # forward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g.derivative(
                    "signal",
                    D(name="mu", fiducial=value, abs_step=1e-4, kind="forward"),
                ),
                g.first_derivative_wrt_mu(value, 1),
            )

        # backward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g.derivative(
                    "signal",
                    D(name="mu", fiducial=value, abs_step=1e-4, kind="backward"),
                ),
                g.first_derivative_wrt_mu(value, 1),
            )

        # forward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g.derivative(
                    "signal",
                    D(name="sigma", fiducial=value, abs_step=1e-4, kind="forward"),
                ),
                g.first_derivative_wrt_sigma(1, value),
            )

        # backward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g.derivative(
                    "signal",
                    D(name="sigma", fiducial=value, abs_step=1e-4, kind="backward"),
                ),
                g.first_derivative_wrt_sigma(1, value),
            )

        # non-existing parameter (implementation dependent!)
        with pytest.raises(ValueError):
            g.derivative("signal", D("a", 1, 1e-2))

        # same as above, but for covariance
        with pytest.raises(ValueError):
            g.derivative("covariance", D("a", 1, 1e-2))

        # it should not be possible to call the derivative function itself
        with pytest.raises(ValueError):
            g.derivative("derivative", D(name="sigma", fiducial=1, abs_step=1e-4))

    def test_second_derivative(self):
        g = GaussianDerivative({"mu": 1, "sigma": 1})

        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g.derivative(
                    "signal", D(name="mu", fiducial=value, abs_step=1e-4, order=2)
                ),
                g.second_derivative_wrt_mu(value, 1),
            )
            # same thing, but we set the derivative
            assert np.allclose(
                g.derivative(
                    "signal",
                    D(name="mu", fiducial=value, abs_step=1e-4),
                    D(name="mu", fiducial=value, abs_step=1e-4),
                ),
                g.second_derivative_wrt_mu(value, 1),
            )

        for mu, sigma in product(np.linspace(-3, 3, 100), repeat=2):
            assert np.allclose(
                g.derivative(
                    "signal",
                    D(name="mu", fiducial=mu, abs_step=1e-5, order=1, accuracy=2),
                    D(name="sigma", fiducial=sigma, abs_step=1e-5, order=1, accuracy=2),
                ),
                g.mixed_derivative(mu, sigma),
                rtol=1e-3,
            )

        for mu, sigma in product(np.linspace(-3, 3, 50), repeat=2):
            assert np.allclose(
                g.derivative(
                    "signal",
                    D(
                        name="mu",
                        fiducial=mu,
                        abs_step=1e-5,
                        order=1,
                        kind="forward",
                        accuracy=2,
                    ),
                    D(
                        name="sigma",
                        fiducial=sigma,
                        abs_step=1e-5,
                        accuracy=2,
                        order=1,
                        kind="center",
                    ),
                ),
                g.mixed_derivative(mu, sigma),
                rtol=1e-3,
            )

        # order of the combined derivative requested is too high
        with pytest.raises(ValueError):
            g.derivative("signal", D(name="mu", fiducial=2, abs_step=1e-5, order=11))

    def test_fisher_matrix(self):
        lin = LinearDerivative()

        # cannot compute the Fisher matrix if covariance is not implemented
        with pytest.raises(NotImplementedError):
            lin.fisher_matrix(D("x", 1, 1e-3))

        g = GaussianDerivative({"mu": 1, "sigma": 1})

        assert np.allclose(
            g.derivative("covariance", D(name="mu", fiducial=1, abs_step=1e-3)), 0
        )

        assert g.fisher_matrix(
            D(name="mu", fiducial=1, abs_step=1e-3),
            D(name="sigma", fiducial=0.5, abs_step=1e-3),
            parameter_dependence="signal",
        ) == g.fisher_matrix(
            D(name="mu", fiducial=1, abs_step=1e-3),
            D(name="sigma", fiducial=0.5, abs_step=1e-3),
            parameter_dependence="both",
        )

        # an object of type subclassed by `FisherDerivative` should work if
        # `external_covariance` is defined, but `covariance` was not overridden
        assert np.allclose(
            lin.fisher_matrix(
                D(name="x", fiducial=1, abs_step=1e-3),
                external_covariance=[[1]],
            ).values,
            lin.derivative(
                "signal",
                D(name="x", fiducial=1, abs_step=1e-3),
            ),
        )

        # wrong dimension of `external_covariance`
        with pytest.raises(ValueError):
            lin.fisher_matrix(
                D(name="x", fiducial=1, abs_step=1e-3),
                external_covariance=[[1, 2, 3], [4, 5, 6]],
            )

        # wrong dimension of `external_covariance`
        with pytest.raises(ValueError):
            lin.fisher_matrix(
                D(name="x", fiducial=1, abs_step=1e-3),
                external_covariance=[1, 2],
            )

        with pytest.warns(UserWarning):
            g.fisher_matrix(
                D(name="mu", fiducial=1, abs_step=1e-3),
                parameter_dependence="both",
                external_covariance=[[1]],
            )

    def test_validate_parameters(self):
        """
        Test to make sure `validate_parameters` method works.
        """
        g = GaussianDerivative({"mu": 1, "sigma": 1})

        with pytest.raises(ValidationError):
            g.fisher_matrix(D(name="asdf", fiducial=0, abs_step=1e-3))

    def test_rounding_threshold_validation(self):
        """
        Test to make sure that the validation of `rounding_threshold` kwarg
        works.
        """
        g = GaussianDerivative({"mu": 1, "sigma": 1})
        with pytest.raises(ValueError):
            g.derivative("signal", D("mu", 1.2, 1e-3), rounding_threshold=-1)


class TestMiscDerivatives:
    def test_supernova_derivative(self):
        sn = SupernovaDerivative(config=dict(omega_m=0.3))
        sn.config = dict(z=[0.5, 1, 1.5], sigma=[1, 1, 1])
        fm = sn.fisher_matrix(
            D("omega_m", 0.32, 1e-3),
            kwargs_signal={"sample_kwarg": "sample_value"},
            kwargs_covariance={"sample_kwarg": "sample_value"},
        )

        with pytest.raises(KeyError):
            sn.config = dict(q="asdf")

        with pytest.raises(ValueError):
            sn.config = dict(z=[1], sigma=[1, 2])
