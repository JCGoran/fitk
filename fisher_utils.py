"""
Various helper utilities for Fisher objects.
"""


from __future__ import annotations

from typing import \
    AnyStr, \
    Iterable, \
    Optional

import numpy as np



def is_iterable(value):
    """
    Checks whether an object is iterable.
    """
    try:
        _ = iter(value)
        return True
    except TypeError:
        return False



# TODO make sure that the representation doesn't exceed some fixed number of digits
def float_to_latex(
    value : float,
    sigdigits : int = 3,
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
        specifier = 'e'
    else:
        specifier = 'g'

    fmt_string = f'{{:.{sigdigits}{specifier}}}'
    float_str = fmt_string.format(value)

    if 'e' in float_str:
        base, exponent = float_str.split('e')
        if np.isclose(float(base), 1):
            return f'10^{{{int(exponent)}}}'
        # remove any trailing zeros and decimal points
        base = base.rstrip('0.')
        return f'{base} \\times 10^{{{int(exponent)}}}'

    return float_str



class ParameterNotFoundError(ValueError):
    """
    Error raised when a parameter is not found in the Fisher object.
    """
    def __init__(
        self,
        name : AnyStr,
        names : Iterable[AnyStr],
    ):
        self.message = f'Parameter \'{name}\' not found in array {names}'
        super().__init__(self.message)



class MismatchingSizeError(Exception):
    """
    Error for handling a list of arrays that have mismatching sizes.
    """
    def __init__(
        self,
        *args,
    ):
        sizes = [str(len(arg)) for arg in args]
        self.message = f'Mismatching lengths: {", ".join(sizes)}'
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
    values : Iterable,
    names : Optional[Iterable[AnyStr]] = None,
    fmt_values : str = '{}',
    fmt_names : str = '{}',
    title : Optional[AnyStr] = None,
):
    """
    Makes a HTML formatted table with names (optional) and values.

    Parameters
    ----------
    values : Iterable
        the values in the table

    names : Iterable, default = None
        the names of the values in the table. Must have the same length as `values`

    fmt_values : str, default = '{}'
        the format string for the values

    fmt_names : str, default = '{}'
        the format string for the names

    title : Optional[AnyStr], default = None
        the title of the table
    """
    if title is not None:
        title = f'<tr><th align="left">{title}</th></tr>'
    else:
        title = ''

    temp = (
        '<tr>' + (f'<th>{fmt_names}</th>' * len(names)).format(*names) + '</tr>'
    ) if names is not None else ''

    header = f'<thead>{title}{temp}</thead>'

    body = '<tbody>' + (
        f'<td>{fmt_values}</td>' * len(values)
    ).format(*values) + '</tbody>'

    return f'<table>{header}{body}</table>'



def make_default_names(
    size : int,
    character : str = 'p',
):
    """
    Returns the array with default names `p1, ..., pn`
    """
    if size < 0:
        raise ValueError
    return np.array([f'{character}{_ + 1}' for _ in range(size)], dtype=object)



def is_square(values):
    """
    Checks whether a numpy array-able object is square.
    """
    try:
        values = np.array(values, dtype=float)
    except ValueError:
        return False

    shape = np.shape(values)
    return all(shape[0] == _ for _ in shape)



def is_symmetric(values):
    """
    Checks whether a numpy array-able object is symmetric.
    """
    return np.allclose(
        np.transpose(values),
        values
    )



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



def jsonify(data):
    """
    Converts arbitrary data into a JSON-compatible format.
    """
    try:
        return data.tolist()
    except AttributeError:
        return data



def reindex_array(values, index):
    """
    Returns the array sorted according to (1D) index `index`.
    """
    for dim in range(np.ndim(values)):
        values = np.swapaxes(
            np.swapaxes(values, 0, dim)[index],
            dim, 0
        )

    return values
