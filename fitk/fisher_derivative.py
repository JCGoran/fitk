"""
Module for computing derivatives using finite differences
"""

# for compatibility with Python 3.7
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Tuple

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
    * `ValueError` if there are duplicate derivatives (such as `D('a',
    abs_step=1e-2), D('a', abs_step=1e-3)`)
    """
    upper_limit = 2

    if np.sum([arg.order for arg in args]) > upper_limit:
        raise ValueError

    if len(set(arg.name for arg in args)) != len([arg.name for arg in args]):
        raise ValueError


def find_diff_weights_mixed(
    *args: D,
):
    """
    Finds weights for mixed partial derivatives (max order: 2)

    Parameters
    ----------
    args
        the derivatives which we want to compute

    Returns
    -------
    array-like of floats
    """
    # all possible combinations of d^2f / dx / dy
    # the missing ones (say, `center_forward`) can be obtained by just swapping
    # elements of the `points` array
    points_center_center = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    weights_center_center = np.array([1, -1, -1, 1]) / 4

    points_backward_center = np.array([[0, -1], [0, 1], [-1, -1], [1, 1]])
    weights_backward_center = np.array([-1, 1, 1, -1]) / 2

    points_backward_backward = np.array([[0, 0], [0, -1], [-1, 0], [-1, -1]])
    weights_backward_backward = np.array([1, -1, -1, 1])

    points_forward_center = np.array([[0, -1], [0, 1], [1, -1], [1, 1]])
    weights_forward_center = np.array([1, -1, -1, 1]) / 2

    points_forward_forward = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    weights_forward_forward = np.array([1, -1, -1, 1])

    points_forward_backward = np.array([[0, 0], [0, -1], [1, 0], [1, -1]])
    weights_forward_backward = np.array([-1, 1, 1, -1])

    mapping = {
        (2, "center", "center"): (points_center_center, weights_center_center),
        (2, "backward", "center"): (points_backward_center, weights_backward_center),
        (2, "center", "backward"): (
            np.flip(points_backward_center, axis=1),
            weights_backward_center,
        ),
        (2, "backward", "backward"): (
            points_backward_backward,
            weights_backward_backward,
        ),
        (2, "backward", "forward"): (
            np.flip(points_forward_backward, axis=1),
            weights_forward_backward,
        ),
        (2, "forward", "backward"): (points_forward_backward, weights_forward_backward),
        (2, "forward", "center"): (points_forward_center, weights_forward_center),
        (2, "center", "forward"): (
            np.flip(points_forward_center, axis=1),
            weights_forward_center,
        ),
        (2, "forward", "forward"): (points_forward_forward, weights_forward_forward),
    }

    order = np.sum([arg.order for arg in args], dtype=int)
    kinds = np.array([arg.kind for arg in args], dtype=str)

    return mapping[(order, *kinds)]


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

    kind
        the kind of difference to use (default: 'center'). Available options
        are 'center', 'forward', 'backward'.
    """

    name: str
    value: float
    abs_step: float
    order: int = 1
    kind: str = "center"

    def __post_init__(self):
        allowed_orders = [1, 2, 3]
        allowed_kinds = ["center", "forward", "backward"]

        if self.order not in allowed_orders:
            raise ValueError(
                f"Only the following derivative orders are allowed: {allowed_orders}"
            )

        if self.kind not in allowed_kinds:
            raise ValueError(
                f"Only the following derivative kinds are allowed: {allowed_kinds}"
            )

        if self.abs_step <= 0:
            raise ValueError("The value of `abs_step` must be positive")


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

        Notes
        -----
        For mixed derivatives (such as $\partial_x \partial_y f(x, y)$), the
        accuracy is always set to $\mathcal{O(h^n)}$, where $n$ is the order of
        the derivative.
        """
        validate_derivatives(*args)

        if len(args) == 1:
            arg = args[0]
            # we keep the accuracy for single derivatives fixed
            single_derivative_accuracy = 4

            if arg.kind == "center":
                npoints = (
                    2 * np.floor((arg.order + 1) / 2) - 2 + single_derivative_accuracy
                ) // 2
                stencil = np.arange(-npoints, npoints + 1, 1)
            elif arg.kind == "forward":
                npoints = single_derivative_accuracy + arg.order
                stencil = np.arange(0, npoints + 1)
            else:
                npoints = single_derivative_accuracy + arg.order
                stencil = np.arange(-npoints, 1)

            weights = find_diff_weights(stencil, order=arg.order)

            return np.array(
                np.sum(
                    [
                        getattr(self, method)(
                            (arg.name, arg.value + arg.abs_step * point)
                        )
                        * weight
                        / arg.abs_step**arg.order
                        for point, weight in zip(stencil, weights)
                    ],
                    axis=0,
                )
            )

        stencil, weights = find_diff_weights_mixed(*args)

        denominator = np.prod([arg.abs_step**arg.order for arg in args])

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
                    for point, weight in zip(stencil, weights)
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

        names = np.array([_.name for _ in args])
        values = np.array([_.value for _ in args])

        # TODO parallelize
        for arg in args:
            signal_derivative[arg.name] = self(
                "signal",
                D(
                    name=arg.name,
                    value=arg.value,
                    abs_step=arg.abs_step,
                    kind=arg.kind,
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
                    ),
                )
            else:
                covariance_derivative[arg.name] = np.zeros(covariance_shape)

        fisher_matrix = np.zeros([len(args)] * 2)

        for (i, name1), (j, name2) in product(enumerate(names), repeat=2):
            fisher_matrix[i, j] = (
                signal_derivative[name1]
                @ inverse_covariance_matrix
                @ signal_derivative[name2]
                + inverse_covariance_matrix
                @ covariance_derivative[name1]
                @ inverse_covariance_matrix
                @ covariance_derivative[name2]
                / 2
            )

        return FisherMatrix(fisher_matrix, names=names, fiducials=values)
