"""
Module for computing derivatives using finite differences
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Collection, Optional, Tuple

# third party imports
import numpy as np

from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_utils import find_diff_weights, is_iterable


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

    stencil
        the custom stencil used for computing the derivative (default: None).
        If specified, the arguments `accuracy` and `kind` are ignored.

    Raises
    ------
    * `ValueError` if the value of `abs_step` is not positive
    * `TypeError` if the value of `stencil` is not an iterable
    * `ValueError` if the value of `stencil` is not strictly monotonically
    increasing
    * `ValueError` if the value of `accuracy` is not at least 1
    * `ValueError` if the value of `kind` is not one of: 'center', 'forward',
    'backward'
    """

    name: str
    value: float
    abs_step: float
    order: int = 1
    accuracy: int = 4
    kind: str = "center"
    stencil: Optional[Collection[float]] = None

    def __post_init__(self):
        if self.abs_step <= 0:
            raise ValueError("The value of `abs_step` must be positive")

        # if we specify a stencil, we don't bother checking `kind` and
        # `accuracy`
        if self.stencil is not None:
            if not is_iterable(self.stencil):
                raise TypeError(
                    f"The value of the argument `stencil` ({self.stencil}) is not an iterable"
                )

            if not np.allclose(sorted(self.stencil), self.stencil):
                raise ValueError(
                    f"The value of the argument `stencil` ({self.stencil}) "
                    "should be an increasing list of numbers"
                )

            return

        allowed_kinds = ["center", "forward", "backward"]

        if self.kind not in allowed_kinds:
            raise ValueError(
                f"Only the following derivative kinds are allowed: {allowed_kinds}"
            )

        if self.accuracy < 1:
            raise ValueError("The accuracy must be at least first-order")

        if self.kind == "center":
            npoints = (2 * np.floor((self.order + 1) / 2) - 2 + self.accuracy) // 2
            self.stencil = np.arange(-npoints, npoints + 1, 1)
        elif self.kind == "forward":
            npoints = self.accuracy + self.order
            self.stencil = np.arange(0, npoints + 1)
        else:
            npoints = self.accuracy + self.order
            self.stencil = np.arange(-npoints, 1)


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

        Raises
        ------
        * `NotImplementedError` if the user has not explicitly overridden the
        method
        """
        raise NotImplementedError(
            "The `covariance` method must be implemented first in order to be used"
        )

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
            points_arr.append(arg.stencil)
            weights_arr.append(find_diff_weights(arg.stencil, order=arg.order))

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
        # first we attempt to compute the covariance; if that fails, it means
        # it hasn't been implemented, so we fail fast and early
        names = np.array([_.name for _ in args])
        values = np.array([_.value for _ in args])

        covariance_matrix = self.covariance(*zip(names, values))
        if len(covariance_matrix) == 1:
            inverse_covariance_matrix = 1 / np.array([covariance_matrix])
        else:
            inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        covariance_shape = np.shape(inverse_covariance_matrix)

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
