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
from fitk.fisher_derivative import D, FisherDerivative
from fitk.interfaces.misc_interfaces import SupernovaDerivative


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

        return x


class GaussianDerivative(FisherDerivative):
    def __init__(self, config=None):
        self.config = config if config is not None else {"mu": 1.0, "sigma": 1.0}

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
        # we cannot instantiate the abstract `FisherDerivative` class
        with pytest.raises(TypeError):
            fd = FisherDerivative()

    def test_first_derivative(self):
        lin = LinearDerivative()

        lin("signal", D("x", 1, 1e-3))

        g = GaussianDerivative({"mu": 1, "sigma": 1})

        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g("signal", D(name="mu", value=value, abs_step=1e-4)),
                g.first_derivative_wrt_mu(value, 1),
            )

        # forward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g("signal", D(name="mu", value=value, abs_step=1e-4, kind="forward")),
                g.first_derivative_wrt_mu(value, 1),
            )

        # backward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g(
                    "signal",
                    D(name="mu", value=value, abs_step=1e-4, kind="backward"),
                ),
                g.first_derivative_wrt_mu(value, 1),
            )

        # forward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g(
                    "signal",
                    D(name="sigma", value=value, abs_step=1e-4, kind="forward"),
                ),
                g.first_derivative_wrt_sigma(1, value),
            )

        # backward method
        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g(
                    "signal",
                    D(name="sigma", value=value, abs_step=1e-4, kind="backward"),
                ),
                g.first_derivative_wrt_sigma(1, value),
            )

        # non-existing parameter (implementation dependent!)
        with pytest.raises(ValueError):
            g("signal", D("a", 1, 1e-2))

        # same as above, but for covariance
        with pytest.raises(ValueError):
            g("covariance", D("a", 1, 1e-2))

    def test_second_derivative(self):
        g = GaussianDerivative({"mu": 1, "sigma": 1})

        for value in np.linspace(-3, 3, 100):
            assert np.allclose(
                g("signal", D(name="mu", value=value, abs_step=1e-4, order=2)),
                g.second_derivative_wrt_mu(value, 1),
            )

        for mu, sigma in product(np.linspace(-3, 3, 100), repeat=2):
            assert np.allclose(
                g(
                    "signal",
                    D(name="mu", value=mu, abs_step=1e-5, order=1, accuracy=2),
                    D(name="sigma", value=sigma, abs_step=1e-5, order=1, accuracy=2),
                ),
                g.mixed_derivative(mu, sigma),
                rtol=1e-3,
            )

        for mu, sigma in product(np.linspace(-3, 3, 50), repeat=2):
            assert np.allclose(
                g(
                    "signal",
                    D(
                        name="mu",
                        value=mu,
                        abs_step=1e-5,
                        order=1,
                        kind="forward",
                        accuracy=2,
                    ),
                    D(
                        name="sigma",
                        value=sigma,
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
            g("signal", D(name="mu", value=2, abs_step=1e-5, order=11))

    def test_fisher_matrix(self):
        lin = LinearDerivative()

        # cannot compute the Fisher matrix if covariance is not implemented
        with pytest.raises(NotImplementedError):
            lin.fisher_matrix(D("x", 1, 1e-3))

        g = GaussianDerivative({"mu": 1, "sigma": 1})

        assert np.allclose(g("covariance", D(name="mu", value=1, abs_step=1e-3)), 0)

        assert g.fisher_matrix(
            D(name="mu", value=1, abs_step=1e-3),
            D(name="sigma", value=0.5, abs_step=1e-3),
            constant_covariance=True,
        ) == g.fisher_matrix(
            D(name="mu", value=1, abs_step=1e-3),
            D(name="sigma", value=0.5, abs_step=1e-3),
            constant_covariance=False,
        )


class TestMiscDerivatives:
    def test_supernova_derivative(self):
        sn = SupernovaDerivative(config=dict(omega_m=0.3))
        sn.config = dict(z=[0.5, 1, 1.5], sigma=[1, 1, 1])
        fm = sn.fisher_matrix(
            D("omega_m", 0.32, 1e-3),
        )

        with pytest.raises(KeyError):
            sn.config = dict(q="asdf")

        with pytest.raises(ValueError):
            sn.config = dict(z=[1], sigma=[1, 2])