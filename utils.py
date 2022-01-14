"""
Various helper utilities for Fisher objects.
"""


from __future__ import annotations

from typing import \
    AnyStr, \
    Iterable

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



def float_to_latex(value : float):
    """
    Format a float into a useful LaTeX representation.
    """
    float_str = f"{value:.2g}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return f"{base} \\times 10^{{{int(exponent)}}}"

    return float_str



class ParameterNotFoundError(Exception):
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
        sizes = [len(arg) for arg in args]
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
    names : Iterable = None,
    fmt_values : str = '{}',
    fmt_names : str = '{}',
    title : str = '',
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

    title : str, default = ''
        the title of the table
    """

    if title:
        title = f'<tr><th align="left">{title}</th></tr>'

    temp = (
        '<tr>' + (f'<th>{fmt_names}</th>' * len(names)).format(*names) + '</tr>'
    ) if names else ''
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
    return np.array([f'{character}{_ + 1}' for _ in range(size)])



def is_square(values):
    """
    Checks whether a numpy array-able object is square.
    """

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



def has_positive_diagonal(values):
    """
    Checks whether all of the values on the diagonal are positive.
    """
    return np.all(np.diag(values) >= 0)
