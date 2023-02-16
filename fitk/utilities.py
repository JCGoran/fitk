"""
Various helper utilities for Fisher objects.
"""

# needed for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
import json
import math
from collections.abc import Collection, Sequence
from math import factorial
from typing import Optional, Union

# third party imports
import numpy as np


class FisherEncoder(json.JSONEncoder):
    """
    Class for custom conversion of numpy objects as JSON
    """

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def is_iterable(value):
    """
    Checks whether an object is iterable.
    """
    try:
        _ = iter(value)
        return True
    except (TypeError, NotImplementedError):
        return False


# TODO make sure that the representation doesn't exceed some fixed number of digits
def float_to_latex(
    value: float,
    sigdigits: int = 3,
):
    """
    Format a float into a useful LaTeX representation.

    Parameters
    ----------
    sigdigits : int, default = 3
        number of significant digits
    """

    # annoying case of small exponents
    if 1e-4 <= np.abs(value) < 1e-2:
        specifier = "e"
    else:
        specifier = "g"

    fmt_string = f"{{:.{sigdigits}{specifier}}}"
    float_str = fmt_string.format(value)

    if "e" in float_str:
        base, exponent = float_str.split("e")
        if np.isclose(float(base), 1):
            return f"10^{{{int(exponent)}}}"
        # remove any trailing zeros and decimal points
        base = base.rstrip("0.")
        return f"{base} \\times 10^{{{int(exponent)}}}"

    return float_str


class ParameterNotFoundError(ValueError):
    """
    Error raised when a parameter is not found in the Fisher object.
    """

    def __init__(
        self,
        name: str,
        names: Collection[str],
    ):
        self.message = f"Parameter '{name}' not found in array {names}"
        super().__init__(self.message)


class ValidationError(Exception):
    """
    Error raised when
    `fitk.derivatives.FisherDerivative.validate_parameter` fails.
    """

    def __init__(self, parameter):
        self.message = f"Parameter '{parameter.name}' contains invalid values"
        super().__init__(self.message)


class MismatchingValuesError(ValueError):
    """
    Error raised when values of either fiducials or names do not match.
    """

    def __init__(
        self,
        name: str,
        values1,
        values2,
    ):
        self.message = f"Incompatible {name}(s): {values1} and {values2}"
        super().__init__(self.message)


class MismatchingSizeError(ValueError):
    """
    Error for handling a list of arrays that have mismatching sizes.
    """

    def __init__(
        self,
        *args,
    ):
        sizes = [str(len(arg)) for arg in args]
        self.message = f'Mismatching sizes: {", ".join(sizes)}'
        super().__init__(self.message)


class HTMLWrapper:
    """
    A wrapper class for pretty printing objects in a Jupyter notebook.
    """

    def __init__(self, string):
        self._string = string

    def _repr_html_(self):
        return self._string


def make_html_table(
    values: Collection,
    names: Optional[Collection[str]] = None,
    fmt_values: str = "{}",
    fmt_names: str = "{}",
    title: Optional[str] = None,
):
    """
    Makes a HTML formatted table with names (optional) and values.

    Parameters
    ----------
    values : Collection
        the values in the table

    names : Collection[str], default = None
        the names of the values in the table. Must have the same length as `values`

    fmt_values : str, default = '{}'
        the format string for the values

    fmt_names : str, default = '{}'
        the format string for the names

    title : Optional[str], default = None
        the title of the table
    """
    if title is not None:
        title = f'<tr><th align="left">{title}</th></tr>'
    else:
        title = ""

    temp = (
        ("<tr>" + (f"<th>{fmt_names}</th>" * len(names)).format(*names) + "</tr>")
        if names is not None
        else ""
    )

    header = f"<thead>{title}{temp}</thead>"

    body = (
        "<tbody>"
        + (f"<td>{fmt_values}</td>" * len(values)).format(*values)
        + "</tbody>"
    )

    return f"<table>{header}{body}</table>"


def make_default_names(
    size: int,
    character: str = "p",
):
    """
    Returns the array with default names `p1, ..., pn`
    """
    if size < 0:
        raise ValueError
    return np.array([f"{character}{_ + 1}" for _ in range(size)], dtype=object)


def is_square(values):
    """
    Checks whether a numpy array-able object is square.
    """
    try:
        values = np.array(values, dtype=float)
    except ValueError:
        return False

    if np.ndim(values) <= 1:
        return False

    shape = np.shape(values)
    return all(shape[0] == _ for _ in shape)


def is_symmetric(values):
    """
    Checks whether a numpy array-able object is symmetric.
    """
    return np.allclose(np.transpose(values), values)


def is_positive_semidefinite(values):
    """
    Checks whether a numpy array-able object is positive semi-definite.
    """
    return np.all(np.linalg.eigvalsh(values) >= 0)


def get_index_of_other_array(A, B):
    """
    Returns the index (an array) which is such that `B[index] == A`.
    """
    A, B = np.array(A), np.array(B)
    xsorted = np.argsort(B)

    return xsorted[np.searchsorted(B[xsorted], A)]


def reindex_array(values, index):
    """
    Returns the array sorted according to (1D) index `index`.
    """
    for dim in range(np.ndim(values)):
        values = np.swapaxes(np.swapaxes(values, 0, dim)[index], dim, 0)

    return values


def get_default_rcparams():
    """
    Returns a dictionary with default parameters used in FITK.

    Returns
    -------
    `dict`
    """
    return {
        "mathtext.fontset": "cm",
        "font.family": "serif",
    }


def process_units(arg: str) -> float:
    """
    Processes unit arguments

    Returns
    -------
    float
    """
    # we only accept bits and bytes
    allowed_units = ["b", "B"]

    allowed_units_multiples = [1, 8]

    allowed_units_dict = dict(zip(allowed_units, allowed_units_multiples))

    # standard SI convention
    allowed_prefixes = ["k", "M", "G", "T", "P", "E", "Z", "Y"]

    allowed_prefixes += [f"{prefix}i" for prefix in allowed_prefixes]

    # taken from:
    # https://en.wikipedia.org/wiki/Units_of_information#Systematic_multiples
    allowed_prefixes_powers = [10**power for power in range(3, 25, 3)]
    allowed_prefixes_powers += [2**power for power in range(10, 61, 10)]

    allowed_prefixes_dict = dict(zip(allowed_prefixes, allowed_prefixes_powers))

    if arg in allowed_units:
        return allowed_units_dict[arg]

    prefix, unit = arg[:-1], arg[-1]

    if unit not in allowed_units:
        raise ValueError(f"The unit must be one of {allowed_units}")

    if prefix not in allowed_prefixes:
        raise ValueError(f"The prefix must be one of {allowed_prefixes}")

    return 1 / allowed_prefixes_dict[prefix] / allowed_units_dict[unit]


def find_diff_weights(
    stencil: Collection[float],
    order: int = 1,
):
    """
    Finds the weights for computing derivatives using finite differences.

    Parameters
    ----------
    stencil : array_like of float
        the points where the derivative should be evaluated

    order : int, optional
        the order of the derivative (default: 1)

    Raises
    ------
    ValueError
        if `order >= len(stencil)`

    Returns
    -------
    array_like : float
        the weights for computing the derivative of the specified order at the
        specified stencil

    Raises
    ------
    ValueError
        if the number of stencil points is smaller or equal to the derivative
        order
    """
    if order >= len(stencil):
        raise ValueError(
            "The number of points in the stencil must be larger than the derivative order"
        )
    matrix = np.vstack([np.array(stencil) ** n for n in range(len(stencil))])
    vector = np.zeros(len(stencil))
    vector[order] = math.factorial(order)

    return np.linalg.inv(matrix) @ vector


def _expansion_coefficient(n1: int, n2: int):
    """
    Returns the expansion coefficient formed with $n_1$ and $n_2$ derivatives.
    """
    if n1 != n2:
        return 1 / factorial(n1) / factorial(n2)

    return 1 / 2 / factorial(n1) ** 2


def math_mode(
    arg: Union[str, Sequence[str]],
) -> Union[str, list[str]]:
    """
    Returns the argument with surrounding math characters (`$...$`).

    Parameters
    ----------
    arg : str or array_like of str
        the string or list of strings which we want to convert to math mode

    Returns
    -------
    str or list of str
        the input converted to math mode

    Raises
    ------
    TypeError
        if the argument is not iterable

    Examples
    --------
    >>> math_mode('a')
    '$a$'
    >>> math_mode(['a', 'b'])
    ['$a$', '$b$']
    """
    if isinstance(arg, str):
        return f"${arg}$"

    try:
        iter(arg)
    except TypeError as err:
        raise TypeError(err) from err

    return [f"${_}$" for _ in arg]
