"""
Submodule for computing derivatives using finite differences.
See here for documentation of `D` and `FisherDerivative`.
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from collections.abc import Collection
from dataclasses import dataclass
from itertools import product
from typing import Optional

# third party imports
import numpy as np

# first party imports
from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_utils import ValidationError, find_diff_weights, is_iterable


def _validate_derivatives(
    *args: D,
):
    """
    Checks whether we are able to compute the requested derivatives

    Raises
    ------
    ValueError
        if we are unable to compute the requested derivatives
    """
    upper_limit = 10

    if np.sum([arg.order for arg in args]) > upper_limit:
        raise ValueError(
            f"Unable to compute derivatives with combined order > {upper_limit}"
        )


def _parse_derivatives(*args: D):
    """
    Parses the derivatives
    """
    parsed_args: list[D] = []
    for arg in args:
        # in case the argument repeats, we just increase the order of the
        # derivative
        if arg in parsed_args:
            parsed_args[parsed_args.index(arg)].order += arg.order
        else:
            parsed_args.append(arg)

    return tuple(parsed_args)


@dataclass
class D:
    """
    Class for describing information about a derivative of a parameter using
    finite differences

    Parameters
    ----------
    name
        the name of the parameter w.r.t. which we take the derivative

    fiducial
        the point where we want to compute the derivative

    abs_step
        the absolute step size for computing the derivative

    order, optional
        the order of the derivative (default: 1)

    accuracy, optional
        the accuracy requested for the derivative (default: 4)

    kind : {'center', 'forward', 'backward'}
        the kind of difference to use (default: 'center')

    stencil, optional
        the custom stencil used for computing the derivative (default: None).
        If specified, the arguments `accuracy` and `kind` are ignored.

    latex_name, optional
        the display name of the parameter. If not specified, equals to name
        (default: None)

    Raises
    ------
    ValueError
        is raised in one of the following situations:
        * if the value of `abs_step` is not positive
        * if the value of `stencil` is not strictly monotonically
        increasing
        * if the value of `accuracy` is not at least 1
        * if the value of `kind` is not one of: 'center', 'forward', 'backward'

    TypeError
        if the value of `stencil` is not an iterable
    """

    name: str
    fiducial: float
    abs_step: float
    order: int = 1
    accuracy: int = 4
    kind: str = "center"
    stencil: Optional[Collection[float]] = None
    latex_name: Optional[str] = None

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.name == other.name
            and np.allclose(self.fiducial, other.fiducial)
            and np.allclose(self.abs_step, other.abs_step)
            and np.allclose(self.order, other.order)
            and self.kind == other.kind
            and np.allclose(self.stencil, other.stencil)
        )

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

        if not self.latex_name:
            self.latex_name = self.name


class FisherDerivative:
    r"""
    Abstract class for computing derivatives using finite differences.

    Notes
    -----
    The user must implement the `signal` or the `covariance` method (or both)
    to be able to use a subclass.
    Furthermore, the `covariance` method, if implemented, needs to return an
    array that is shape-compatible with the output of the `signal` method.
    """

    def __init__(self, *args, **kwargs):
        """
        Placeholder constructor that does nothing. The user should override
        this method if they wish to perform custom initialization.
        """

    def validate_parameter(
        self,
        arg: D,
    ) -> bool:
        """
        Placeholder method used for validating a parameter when calling
        `fisher_matrix`. The user should override this method if they wish to
        perform custom validation.

        Parameters
        ----------
        arg
            the parameter (see description of `D`) which we want to validate

        Returns
        -------
        bool
            the outcome of the validation (default: True)
        """
        return True

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        """
        The method to implement for computing the signal.

        Parameters
        ----------
        *args : tuple
            the name(s) of the parameter(s) and their respective value(s)

        **kwargs
            any keyword arguments to be passed

        Returns
        -------
        array_like : float
            the values of the signal as a numpy array

        Raises
        ------
        NotImplementedError
            if the user has not explicitly overridden the method
        """
        raise NotImplementedError(
            "The `signal` method must be implemented first in order to be used"
        )

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        """
        The (optional) method to implement for computing the covariance.

        Parameters
        ----------
        *args : tuple
            the name(s) of the parameter(s) and their respective value(s)

        **kwargs
            any keyword arguments to be passed

        Returns
        -------
        array_like : float
            the values of the covariance as a numpy array

        Raises
        ------
        NotImplementedError
            if the user has not explicitly overridden the method
        """
        raise NotImplementedError(
            "The `covariance` method must be implemented first in order to be used"
        )

    def derivative(
        self,
        method: str,
        *args: D,
        **kwargs,
    ):
        r"""
        Evaluates the derivative of `method` with respect to arguments `args`.

        Parameters
        ----------
        method : {'signal', 'covariance'}
            the object (method) for which we want to compute the derivative

        *args
            the parameters (see description of `D`) for which we want to
            compute the derivatives

        **kwargs
            any keyword arguments passed to the method

        Returns
        -------
        array_like of float
            the resulting derivative as a numpy array

        Raises
        ------
        ValueError
            if `method` is not one of {'signal', 'covariance'}
        """
        # check that `method` is valid
        valid_methods = ["signal", "covariance"]
        if method not in valid_methods:
            raise ValueError(
                f"The method `{method}` is not one of: {valid_methods}",
            )
        _validate_derivatives(*args)

        parsed_args = _parse_derivatives(*args)

        weights_arr = [
            find_diff_weights(arg.stencil, order=arg.order) for arg in parsed_args
        ]
        points_arr = [arg.stencil for arg in parsed_args]

        denominator = np.prod([arg.abs_step**arg.order for arg in parsed_args])

        stencils = np.array(list(product(*points_arr)))
        weights = np.array([np.prod(_) for _ in product(*weights_arr)])

        # zero-out elements which are close to zero (they are non-zero due to
        # floating-point math)
        weights = np.array([_ if np.abs(_) > 1e-10 else 0 for _ in weights])

        # remove them from the stencil
        stencils = stencils[np.nonzero(weights)]
        weights = weights[np.nonzero(weights)]

        return np.array(
            np.sum(
                [
                    getattr(self, method)(
                        *[
                            (arg.name, arg.fiducial + arg.abs_step * p)
                            for arg, p in zip(parsed_args, point)
                        ],
                        **kwargs,
                    )
                    * weight
                    / denominator
                    for point, weight in zip(stencils, weights)
                ],
                axis=0,
            )
        )

    def fisher_matrix(
        self,
        *args: D,
        parameter_dependence: str = "signal",
        **kwargs,
    ):
        r"""
        Computes the Fisher matrix, $\mathsf{F}$, using finite differences.
        The element $\mathsf{F}_{ij}$ for parameters $(\theta_i, \theta_j)$ is
        defined as:
        $$
            \frac{\partial \mathbf{S}^T}{\partial \theta_i}
            \mathsf{C}^{-1}
            \frac{\partial \mathbf{S}}{\partial \theta_j}
            +
            \frac{1}{2}
            \mathrm{Tr}\left(
            \frac{\partial \mathsf{C}}{\partial \theta_i}
            \mathsf{C}^{-1}
            \frac{\partial \mathsf{C}}{\partial \theta_j}
            \mathsf{C}^{-1}
            \right)
        $$
        where $\mathbf{S}$ is the signal vector, $\mathsf{C}$ is the covariance
        matrix, and $\mathrm{Tr}(X)$ denotes the trace of the quantity $X$.

        Parameters
        ----------
        *args
            the parameters (see description of `D`) for which we want to
            compute the derivatives

        parameter_dependence : {'signal', 'covariance', 'both'}
            where the parameter dependence is located, in the signal, the
            covariance, or both (default: 'signal')

        **kwargs
            any other keyword arguments that should be passed to `signal` and
            `covariance`

        Returns
        -------
        fitk.fisher_matrix.FisherMatrix
            the Fisher object with corresponding names and fiducials

        Raises
        ------
        NotImplementedError
            if either `signal` or `covariance` have not been implemented, and
            the user set `parameter_dependence` to be in one of those

        ValidationError
            if the parameter validation failed

        Notes
        -----
        The `order` parameter is ignored if passed to `D`.
        """
        # first we attempt to compute the covariance; if that fails, it means
        # it hasn't been implemented, so we fail fast and early
        names = np.array([_.name for _ in args])
        fiducials = np.array([_.fiducial for _ in args])
        latex_names = np.array([_.latex_name for _ in args])

        for arg in args:
            if not self.validate_parameter(arg):
                raise ValidationError(arg)

        covariance_matrix = self.covariance(*zip(names, fiducials), **kwargs)
        inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        covariance_shape = np.shape(inverse_covariance_matrix)

        signal_derivative = {}

        # TODO parallelize
        for arg in args:
            if parameter_dependence in ["signal", "both"]:
                signal_derivative[arg.name] = self.derivative(
                    "signal",
                    D(
                        name=arg.name,
                        fiducial=arg.fiducial,
                        abs_step=arg.abs_step,
                        kind=arg.kind,
                        accuracy=arg.accuracy,
                        stencil=arg.stencil,
                    ),
                    **kwargs,
                )

        covariance_derivative = {}

        # TODO parallelize
        for arg in args:
            if parameter_dependence in ["covariance", "both"]:
                covariance_derivative[arg.name] = self.derivative(
                    "covariance",
                    D(
                        name=arg.name,
                        fiducial=arg.fiducial,
                        abs_step=arg.abs_step,
                        kind=arg.kind,
                        accuracy=arg.accuracy,
                        stencil=arg.stencil,
                    ),
                    **kwargs,
                )
            else:
                covariance_derivative[arg.name] = np.zeros(covariance_shape)

        fisher_matrix = np.zeros([len(args)] * 2)

        for (i, arg1), (j, arg2) in product(enumerate(args), repeat=2):
            fisher_matrix[i, j] = (
                signal_derivative[arg1.name]
                @ inverse_covariance_matrix
                @ signal_derivative[arg2.name]
                + np.trace(
                    inverse_covariance_matrix
                    @ covariance_derivative[arg1.name]
                    @ inverse_covariance_matrix
                    @ covariance_derivative[arg2.name]
                )
                / 2
            )

        return FisherMatrix(
            fisher_matrix,
            names=names,
            fiducials=fiducials,
            latex_names=latex_names,
        )
