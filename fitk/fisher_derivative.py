"""
Module for computing derivatives using finite differences
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Tuple

# third party imports
import numpy as np

from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_utils import find_diff_weights


def validate_derivatives(
    *args: D,
):
    """
    Checks whether we are able to compute the requested derivatives

    Raises
    ------
    * `ValueError` if we are unable to compute the requested derivatives
    """
    upper_limit = 10

    if np.sum([arg.order for arg in args]) > upper_limit:
        raise ValueError(
            f"Unable to compute derivatives with combined order > {upper_limit}"
        )


@dataclass
class D:
    """
    Class for describing information about a derivative of a parameter using
    finite differences

    Parameters
    ----------
    name
        the name of the parameter w.r.t. which we take the derivative

    value
        the point where we want to compute the derivative

    abs_step
        the absolute step size for computing the derivative

    order
        the order of the derivative (default: 1)

    accuracy
        the accuracy requested for the derivative (default: 4)

    kind
        the kind of difference to use (default: 'center'). Available options
        are 'center', 'forward', 'backward'.
    """

    name: str
    value: float
    abs_step: float
    order: int = 1
    accuracy: int = 4
    kind: str = "center"

    def __post_init__(self):
        allowed_kinds = ["center", "forward", "backward"]

        if self.kind not in allowed_kinds:
            raise ValueError(
                f"Only the following derivative kinds are allowed: {allowed_kinds}"
            )

        if self.abs_step <= 0:
            raise ValueError("The value of `abs_step` must be positive")

        if self.accuracy < 1:
            raise ValueError("The accuracy must be at least first-order")


class FisherDerivative(ABC):
    """
    Abstract class for computing derivatives using finite differences.

    Notes
    -----
    The user must at least implement the `signal` method to be able to
    instantiate a subclass.
    Furthermore, the `covariance` method, if implemented, needs to return an
    array that is shape-compatible with the output of the `signal` method.
    """

    @abstractmethod
    def signal(
        self,
        *args: Tuple[str, float],
        **kwargs,
    ):
        """
        The method to implement for computing the signal.

        Parameters
        ----------
        args
            the name(s) of the parameter(s) and their respective value(s)

        kwargs
            any keyword arguments to be passed

        Returns
        -------
        array-like of floats
        """
        return NotImplemented

    def covariance(
        self,
        *args: Tuple[str, float],
        **kwargs,
    ):
        """
        The (optional) method to implement for computing the covariance.

        Parameters
        ----------
        args
            the name(s) of the parameter(s) and their respective value(s)

        kwargs
            any keyword arguments to be passed

        Returns
        -------
        array-like of floats with a square shape
        """
        return NotImplemented

    def __call__(
        self,
        method: str,
        *args: D,
    ):
        r"""
        Evaluates the derivative.

        Parameters
        ----------
        method
            the method to use for computing the derivative. Can be either
            'signal' or 'covariance'.

        arg: D
            the derivative
        """
        validate_derivatives(*args)

        weights_arr = []
        points_arr = []

        for arg in args:
            if arg.kind == "center":
                npoints = (2 * np.floor((arg.order + 1) / 2) - 2 + arg.accuracy) // 2
                stencil = np.arange(-npoints, npoints + 1, 1)
            elif arg.kind == "forward":
                npoints = arg.accuracy + arg.order
                stencil = np.arange(0, npoints + 1)
            else:
                npoints = arg.accuracy + arg.order
                stencil = np.arange(-npoints, 1)

            points_arr.append(stencil)
            weights_arr.append(find_diff_weights(stencil, order=arg.order))

        denominator = np.prod([arg.abs_step**arg.order for arg in args])

        stencils = np.array(list(product(*points_arr)))
        weights = np.array([np.prod(_) for _ in product(*weights_arr)])

        # remove any zero-like elements
        weights = np.array([_ if np.abs(_) > 1e-10 else 0 for _ in weights])

        return np.array(
            np.sum(
                [
                    getattr(self, method)(
                        *[
                            (arg.name, arg.value + arg.abs_step * p)
                            for arg, p in zip(args, point)
                        ]
                    )
                    * weight
                    / denominator
                    for point, weight in zip(stencils, weights)
                ],
                axis=0,
            )
        )

    def fisher_tensor(
        self,
        *args: D,
        constant_covariance: bool = True,
    ):
        r"""
        Computes the Fisher matrix, $\mathsf{F}$, using finite differences.

        Parameters
        ----------
        args
            the derivatives (see description of `D`) for which we want to
            compute the derivatives

        constant_covariance
            whether or not to treat the covariance as constant (default: true)

        Returns
        -------
        instance of `FisherMatrix` with corresponding names and fiducials

        Notes
        -----
        The derivative order is automatically set to 1 for each parameter, and
        is ignored if passed to `D`.
        """
        signal_derivative = {}

        # TODO parallelize
        for arg in args:
            signal_derivative[arg.name] = self(
                "signal",
                D(
                    name=arg.name,
                    value=arg.value,
                    abs_step=arg.abs_step,
                    kind=arg.kind,
                    accuracy=arg.accuracy,
                ),
            )

        names = np.array([_.name for _ in args])
        values = np.array([_.value for _ in args])

        covariance_matrix = self.covariance(*zip(names, values))
        if len(covariance_matrix) == 1:
            inverse_covariance_matrix = np.array([covariance_matrix])
        else:
            inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        covariance_shape = np.shape(inverse_covariance_matrix)

        covariance_derivative = {}

        # TODO parallelize
        for arg in args:
            if not constant_covariance:
                covariance_derivative[arg.name] = self(
                    "covariance",
                    D(
                        name=arg.name,
                        value=arg.value,
                        abs_step=arg.abs_step,
                        kind=arg.kind,
                        accuracy=arg.accuracy,
                    ),
                )
            else:
                covariance_derivative[arg.name] = np.zeros(covariance_shape)

        fisher_matrix = np.zeros([len(args)] * 2)

        for (i, arg1), (j, arg2) in product(enumerate(args), repeat=2):
            fisher_matrix[i, j] = (
                signal_derivative[arg1.name]
                @ inverse_covariance_matrix
                @ signal_derivative[arg2.name]
                + inverse_covariance_matrix
                @ covariance_derivative[arg1.name]
                @ inverse_covariance_matrix
                @ covariance_derivative[arg2.name]
                / 2
            )

        return FisherMatrix(fisher_matrix, names=names, fiducials=values)
