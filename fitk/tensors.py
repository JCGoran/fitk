"""
Submodule for performing operations on Fisher objects.
See here for documentation of `FisherMatrix`.
"""

# needed for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
import copy
import json
from collections.abc import Collection, Mapping
from itertools import permutations, product
from numbers import Number
from pathlib import Path
from typing import Any, Optional, Union

# third party imports
import numpy as np
from scipy.special import erfinv  # pylint: disable=no-name-in-module

# first party imports
from fitk.utilities import (
    FisherEncoder,
    MismatchingSizeError,
    MismatchingValuesError,
    ParameterNotFoundError,
    get_index_of_other_array,
    is_iterable,
    is_positive_semidefinite,
    is_square,
    is_symmetric,
    make_default_names,
    reindex_array,
)


def _process_fisher_mapping(value: Mapping) -> dict[str, Union[str, float]]:
    """
    Processes a mapping/dict and returns the sanitized output.
    """
    if "name" not in value:
        raise ValueError("The mapping/dict must contain at least the key `name`")

    name = value["name"]
    latex_name = value.get("latex_name", name)
    fiducial = value.get("fiducial", 0)

    return dict(
        name=name,
        latex_name=latex_name,
        fiducial=fiducial,
    )


class FisherMatrix:
    r"""
    Class for handling Fisher objects.

    Examples
    --------
    Specify a Fisher object with default names and fiducials:
    >>> fm = FisherMatrix(np.diag([5, 4]), latex_names=['$p_1$', '$p_2$'])

    The object has a friendly representation in the interactive session:
    >>> fm
    FisherMatrix(
        array([[5., 0.],
           [0., 4.]]),
        names=array(['p1', 'p2'], dtype=object),
        latex_names=array(['$p_1$', '$p_2$'], dtype=object),
        fiducials=array([0., 0.]))

    In a Jupyter notebook, it looks similar to the below (you can hover over
    the $\LaTeX$ label to show the corresponding name and fiducial of that
    parameter):

    <table>
    <thead>
    <tr>
    <th></th>
    <th title="name = 'p1', fiducial = 0.0">$p_1$</th>
    <th title="name = 'p2', fiducial = 0.0">$p_2$</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <th title="name = 'p1', fiducial = 0.0">$p_1$</th>
    <td>5.000</td>
    <td>0.000</td>
    </tr>
    <tr>
    <th title="name = 'p2', fiducial = 0.0">$p_2$</th>
    <td>0.000</td>
    <td>4.000</td>
    </tr>
    </tbody>
    </table>

    List the names:
    >>> fm.names
    array(['p1', 'p2'], dtype=object)

    List the values of the Fisher object:
    >>> fm.values
    array([[5., 0.],
           [0., 4.]])

    List the values of the fiducials:
    >>> fm.fiducials
    array([0., 0.])

    Names can be changed in bulk (ditto for fiducials and values; dimension
    must of course match the original):
    >>> fm = FisherMatrix(np.diag([5, 4]))
    >>> fm.names = ['x', 'y']
    >>> fm.latex_names = [r'$\mathbf{X}$', r'$\mathbf{Y}$']
    >>> fm
    FisherMatrix(
        array([[5., 0.],
           [0., 4.]]),
        names=array(['x', 'y'], dtype=object),
        latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object),
        fiducials=array([0., 0.]))

    We can get and set individual elements of the matrix using dict-like notation:
    >>> fm = FisherMatrix(np.diag([5, 4]), names=['x', 'y'])
    >>> fm['x', 'x']
    5.0

    The off-diagonal elements are automatically updated when using the setter:
    >>> fm = FisherMatrix(np.diag([5, 4]), names=['x', 'y'])
    >>> fm['x', 'y'] = -2
    >>> fm
    FisherMatrix(
        array([[ 5., -2.],
           [-2.,  4.]]),
        names=array(['x', 'y'], dtype=object),
        latex_names=array(['x', 'y'], dtype=object),
        fiducials=array([0., 0.]))

    We can select submatrices by index:
    >>> fm = FisherMatrix(np.diag([1, 2, 3]), names=['x', 'y', 'z'])
    >>> fm[1:]
    FisherMatrix(
        array([[2., 0.],
           [0., 3.]]),
        names=array(['y', 'z'], dtype=object),
        latex_names=array(['y', 'z'], dtype=object),
        fiducials=array([0., 0.]))

    Fisher object with parameter names:
    >>> fm = FisherMatrix(np.diag([5, 4]),
    ... names=['x', 'y'], latex_names=['$\\mathbf{X}$', '$\\mathbf{Y}$'])
    >>> fm_with_names = FisherMatrix(np.diag([1, 2]), names=['x', 'y'])

    We can add Fisher objects (ordering of names is taken care of):
    >>> fm + fm_with_names # doctest: +SKIP
    FisherMatrix(
        array([[6., 0.],
           [0., 6.]]),
        names=array(['x', 'y'], dtype=object),
        latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object),
        fiducials=array([0., 0.]))

    We can also do element-wise multiplication or division:
    >>> fm * fm_with_names
    FisherMatrix(
        array([[5., 0.],
           [0., 8.]]),
        names=array(['x', 'y'], dtype=object),
        latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object),
        fiducials=array([0., 0.]))

    Furthermore, we can do matrix multiplication:
    >>> fm @ fm_with_names
    FisherMatrix(
        array([[5., 0.],
           [0., 8.]]),
        names=array(['x', 'y'], dtype=object),
        latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object),
        fiducials=array([0., 0.]))

    We can also perform standard matrix operations like the trace, eigenvalues, determinant:
    >>> fm.trace()
    9.0
    >>> fm.eigenvalues()
    array([4., 5.])
    >>> fm.determinant()
    19.999999999999996

    We can also get the inverse (the covariance matrix):
    >>> fm.inverse()
    array([[0.2 , 0.  ],
           [0.  , 0.25]])

    We can drop parameters from the object:
    >>> fm.drop('x')
    FisherMatrix(
        array([[4.]]),
        names=array(['y'], dtype=object),
        latex_names=array(['$\\mathbf{Y}$'], dtype=object),
        fiducials=array([0.]))

    We can save it to a file (the returned value is the dictionary that was saved):
    >>> pprint(fm.to_file('example_matrix.json'), sort_dicts=False) # doctest: +SKIP
    {'values': [[5.0, 0.0], [0.0, 4.0]],
     'names': ['x', 'y'],
     'latex_names': ['$\\mathbf{X}$', '$\\mathbf{Y}$'],
     'fiducials': [0.0, 0.0]}

    Loading is performed by a class method `from_file`:
    >>> fm_new = FisherMatrix.from_file('example_matrix.json') # doctest: +SKIP
    """

    def __init__(
        self,
        values,
        names: Optional[Collection[str]] = None,
        latex_names: Optional[Collection[str]] = None,
        fiducials: Optional[Collection[float]] = None,
    ):
        r"""
        Constructor for Fisher object.

        Parameters
        ----------
        values : array_like of float
            The values of the Fisher object.

        names : array_like of str, optional
            The names of the parameters (default: None).
            If not specified, defaults to `p1, ..., pn`.

        latex_names : array_like of str, optional
            The LaTeX names of the parameters (default: None).
            If not specified, defaults to `names`.

        fiducials : array_like of float, optional
            The fiducial values of the parameters (default: None). If not
            specified, default to 0 for all parameters.

        Raises
        ------
        ValueError
            if the input array has the wrong dimensionality (not 2)

        ValueError
            if the object is not square-like

        MismatchingSizeError
            if the sizes of the array of names, values, LaTeX names and
            fiducials do not match

        Examples
        --------
        Specify a Fisher object without names:
        >>> FisherMatrix(np.diag([1, 2, 3]))
        FisherMatrix(
            array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]]),
            names=array(['p1', 'p2', 'p3'], dtype=object),
            latex_names=array(['p1', 'p2', 'p3'], dtype=object),
            fiducials=array([0., 0., 0.]))

        Specify a Fisher object with names, LaTeX names, and fiducials:
        >>> FisherMatrix(np.diag([1, 2]), names=['alpha', 'beta'],
        ... latex_names=[r'$\alpha$', r'$\beta$'], fiducials=[-3, 2])
        FisherMatrix(
            array([[1., 0.],
               [0., 2.]]),
            names=array(['alpha', 'beta'], dtype=object),
            latex_names=array(['$\\alpha$', '$\\beta$'], dtype=object),
            fiducials=array([-3.,  2.]))
        """
        if np.ndim(values) != 2:
            raise ValueError(f"The object {values} is not 2-dimensional")

        if not is_square(values):
            raise ValueError(f"The object {values} is not square-like")

        # try to treat it as an array-like object
        self._values = np.array(values, dtype=float)

        self._size = np.shape(self._values)[0]
        self._ndim = np.ndim(self._values)

        # setting the fiducials
        if fiducials is None:
            self._fiducials = np.zeros(self._size, dtype=float)
        else:
            self._fiducials = np.array(fiducials, dtype=float)

        # setting the names
        if names is None:
            self._names = make_default_names(self._size)
        else:
            # check they're unique
            if len(set(names)) != len(names):
                raise MismatchingSizeError(set(names), names)

            self._names = np.array(names, dtype=object)

        # setting the pretty names (LaTeX)
        if latex_names is None:
            self._latex_names = copy.deepcopy(self._names)
        else:
            self._latex_names = np.array(latex_names, dtype=object)

        # check sizes of inputs
        if not all(
            len(_) == self._size
            for _ in (self._names, self._fiducials, self._latex_names)
        ):
            raise MismatchingSizeError(
                self._values[0],
                self._names,
                self._fiducials,
                self._latex_names,
            )

    @property
    def matrix(self):
        """
        Returns the values of the Fisher matrix (the Gaussian contribution to
        the likelihood) as a numpy array.
        """
        return self._values

    @property
    def values(self):
        """
        Returns the values in the Fisher object as a numpy array.

        Raises
        ------
        MismatchingSizeError
            if the new values do not have the same size as the original ones

        ValueError
            if the shape of the new values is not square

        TypeError
            if the new values do not have the same type as the original ones

        Notes
        -----
        Currently returns the same output as `matrix` (subject to change).
        """
        return self._values

    @values.setter
    def values(
        self,
        value,
    ):
        if len(self) != len(value):
            raise MismatchingSizeError(self, value)

        if not is_square(value):
            raise ValueError(f"{value} is not a square object")

        self._values = np.array(value, dtype=float)

    @property
    def fiducials(self):
        """
        Returns the fiducial values of the Fisher object as a numpy array.

        Raises
        ------
        MismatchingSizeError
            if the new values do not have the same size as the original ones

        TypeError
            if the new values do not have the same type as the original ones
        """
        return self._fiducials

    @fiducials.setter
    def fiducials(
        self,
        value: Collection[float],
    ):
        if len(value) != len(self):
            raise MismatchingSizeError(value, self)
        try:
            self._fiducials = np.array([float(_) for _ in value])
        except (TypeError, ValueError) as err:
            raise TypeError(err) from err

    @property
    def names(self):
        """
        Returns the parameter names of the Fisher object.

        Raises
        ------
        MismatchingSizeError
            if the new values do not have the same size as the original ones
        """
        return self._names

    @names.setter
    def names(
        self,
        value: Collection[str],
    ):
        if len(set(value)) != len(self):
            raise MismatchingSizeError(set(value), self)
        self._names = np.array(value, dtype=object)

    @property
    def latex_names(self):
        """
        Returns the LaTeX names of the parameters of the Fisher object.

        Raises
        ------
        MismatchingSizeError
            if the new values do not have the same size as the original ones
        """
        return self._latex_names

    @latex_names.setter
    def latex_names(
        self,
        value: Collection[str],
    ):
        if len(set(value)) != len(self):
            raise MismatchingSizeError(set(value), self)
        self._latex_names = np.array(value, dtype=object)

    def rename(
        self,
        names: Mapping[str, Union[str, Mapping]],
        ignore_errors: bool = False,
    ) -> FisherMatrix:
        """
        Returns a Fisher object with new names. This is primarily useful for
        renaming individual parameters.

        Parameters
        ----------
        names : dict-like
            a mapping (dictionary-like object) between the old names and the
            new ones. The values it maps to can either be a string (the new
            name), or a dict with keys `name`, `latex_name`, and `fiducial`
            (only `name` is mandatory).

        ignore_errors : bool, optional
            if set to True, will not raise an error if a parameter doesn't
            exist (default: False)

        Returns
        -------
        FisherMatrix
            the Fisher object with new names

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))
        >>> m.rename({'p1' : 'a', 'p2' : dict(name='b', latex_name='$b$', fiducial=2)})
        FisherMatrix(
            array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]]),
            names=array(['a', 'b', 'p3'], dtype=object),
            latex_names=array(['a', '$b$', 'p3'], dtype=object),
            fiducials=array([0., 2., 0.]))
        """
        if not ignore_errors:
            for name in names:
                if name not in self.names:
                    raise ParameterNotFoundError(name, self.names)

        names_new = copy.deepcopy(self.names)
        latex_names_new = copy.deepcopy(self.latex_names)
        fiducial_new = copy.deepcopy(self.fiducials)

        for name, value in names.items():
            index = np.where(names_new == name)
            # it's a mapping
            if isinstance(value, Mapping):
                value = _process_fisher_mapping(value)
                latex_names_new[index] = value["latex_name"]
                fiducial_new[index] = value["fiducial"]
                names_new[index] = value["name"]
            # otherwise, it's a string
            else:
                names_new[index] = value
                latex_names_new[index] = value

        return self.__class__(
            self.values,
            names=names_new,
            latex_names=latex_names_new,
            fiducials=fiducial_new,
        )

    def _repr_html_(self):
        """
        Representation of the Fisher object suitable for viewing in Jupyter
        notebook environments.
        """
        header_matrix = (
            "<thead><tr><th></th>"
            + (
                "".join(
                    [
                        f"""<th title="name = '{name}', fiducial = {fiducial}">{latex_name}</th>"""
                        for name, fiducial, latex_name in zip(
                            self.names,
                            self.fiducials,
                            self.latex_names,
                        )
                    ]
                )
            )
            + "</tr></thead>"
        )

        body_matrix = "<tbody>"
        for name, fiducial, (index, latex_name) in zip(
            self.names,
            self.fiducials,
            enumerate(self.latex_names),
        ):
            body_matrix += (
                f"""<tr><th title="name = '{name}', fiducial = {fiducial}">{latex_name}</th>"""
                + ("<td>{:.3f}</td>" * len(self)).format(*(self.values[:, index]))
                + "</tr>"
            )
        body_matrix += "</tbody>"

        html_matrix = f"<table>{header_matrix}{body_matrix}</table>"

        return html_matrix

    def __repr__(self):
        """
        Representation of the Fisher object for non-Jupyter interfaces.
        """
        return (
            "FisherMatrix(\n"
            f"    {repr(self.values)},\n"
            f"    names={repr(self.names)},\n"
            f"    latex_names={repr(self.latex_names)},\n"
            f"    fiducials={repr(self.fiducials)})"
        )

    def __str__(self):
        """
        String representation of the Fisher object.
        """
        return (
            "FisherMatrix(\n"
            f"    {self.values},\n"
            f"    names={self.names},\n"
            f"    latex_names={self.latex_names},\n"
            f"    fiducials={self.fiducials})"
        )

    def __iter__(self):
        raise NotImplementedError(
            f"Object of type `{self.__class__.__name__}` is not iterable; "
            "use `<object>[<key>]` to get or set elements of the object instead"
        )

    def __getitem__(
        self,
        keys: Union[tuple[str, ...], slice],
    ):
        """
        Implements access to elements in the Fisher object.
        Has support for slicing.
        """
        # the object can be sliced
        if isinstance(keys, slice):
            start, stop, step = keys.indices(len(self))
            indices = (slice(start, stop, step),) * self.ndim
            sl = slice(start, stop, step)
            values = self.values[indices]
            names = self.names[sl]
            latex_names = self.latex_names[sl]
            fiducials = self.fiducials[sl]

            return self.__class__(
                values,
                names=names,
                latex_names=latex_names,
                fiducials=fiducials,
            )

        try:
            _ = iter(keys)
        except TypeError as err:
            raise TypeError(err) from err

        # the keys can be a tuple
        if isinstance(keys, tuple):
            if len(keys) != self.ndim:
                raise ValueError(f"Expected {self.ndim} arguments, got {len(keys)}")

            # error checking
            for key in keys:
                if key not in self.names:
                    raise ParameterNotFoundError(key, self.names)

            indices = tuple(np.where(self.names == key) for key in keys)

        # otherwise, it's some generic object
        else:
            if keys not in self.names:
                raise ParameterNotFoundError(keys, self.names)

            indices = (np.where(self.names == keys),)

        return self._values[indices][0, 0]

    def __setitem__(
        self,
        keys: tuple[str, ...],
        value: float,
    ):
        """
        Implements setting of elements in the Fisher object.
        Does not support slicing.

        Raises
        ------
        TypeError
            if keys are not a tuple, or if `value` is not a number

        MismatchingSizeError
            if the size of keys are not equal to the dimensionality of the
            Fisher object

        ParameterNotFoundError
            if any of the keys is not in the parameter names
        """
        try:
            _ = iter(keys)
        except TypeError as err:
            raise TypeError(err) from err

        # the above fails for strings (since they are iterable), so we do an
        # explicit check here
        if isinstance(keys, str):
            raise MismatchingSizeError(keys)

        if len(keys) != self.ndim:
            raise MismatchingSizeError(keys)

        if not isinstance(value, Number):
            raise TypeError(f"{value} is not a number")

        for key in keys:
            if key not in self.names:
                raise ParameterNotFoundError(key, self.names)

        # automatically raises a value error
        indices = tuple(np.where(self.names == key) for key in keys)

        temp_data = copy.deepcopy(self._values)

        if not all(index == indices[0] for index in indices):
            for permutation in list(permutations(indices)):
                # update all symmetric parts
                temp_data[permutation] = value
        else:
            temp_data[indices] = value

        self._values = copy.deepcopy(temp_data)

    def fiducial(self, name: str) -> float:
        r"""
        Returns the value of the fiducial associated to the parameter `name`.

        Raises
        ------
        ParameterNotFoundError
            if the name is not in the Fisher object

        Examples
        --------
        Create a Fisher matrix:
        >>> fm = FisherMatrix(np.diag([1, 2]),
        ... names=['a', 'b'], fiducials=[2, 3])

        Get the value of the fiducial associated to parameter `b`:
        >>> fm.fiducial('b')
        3.0
        """
        if name not in self.names:
            raise ParameterNotFoundError(name, self.names)

        return self.fiducials[np.where(self.names == name)][0]

    def set_fiducial(self, name: str, value: float):
        """
        Sets the fiducial of parameter `name`.

        Raises
        ------
        ParameterNotFoundError
            if the name is not in the Fisher object

        Examples
        --------
        Create a Fisher matrix:
        >>> fm = FisherMatrix(np.diag([1, 2]),
        ... names=['a', 'b'], fiducials=[2, 3])

        Set the value of the fiducial associated to parameter `b`:
        >>> fm.set_fiducial('b', 4)
        >>> fm
        FisherMatrix(
            array([[1., 0.],
               [0., 2.]]),
            names=array(['a', 'b'], dtype=object),
            latex_names=array(['a', 'b'], dtype=object),
            fiducials=array([2., 4.]))
        """
        if name not in self.names:
            raise ParameterNotFoundError(name, self.names)

        self.fiducials[np.where(self.names == name)[0]] = value

    def latex_name(self, name: str) -> str:
        r"""
        Returns the LaTeX name associated to the parameter `name`.

        Raises
        ------
        ParameterNotFoundError
            if the name is not in the Fisher object

        Examples
        --------
        Create a Fisher matrix:
        >>> fm = FisherMatrix(np.diag([1, 2]),
        ... names=['a', 'b'], latex_names=[r'$\mathbf{A}$', r'$\mathbf{B}$'])

        Get the LaTeX name associated to parameter `b`:
        >>> fm.latex_name('b')
        '$\\mathbf{B}$'
        """
        if name not in self.names:
            raise ParameterNotFoundError(name, self.names)

        return self.latex_names[np.where(self.names == name)][0]

    def set_latex_name(self, name: str, value: str):
        r"""
        Sets the LaTeX name of parameter `name`.

        Raises
        ------
        ParameterNotFoundError
            if the name is not in the Fisher object

        Examples
        --------
        Create a Fisher matrix:
        >>> fm = FisherMatrix(np.diag([1, 2]),
        ... names=['a', 'b'], fiducials=[2, 3])

        Set the value of the fiducial associated to parameter `b`:
        >>> fm.set_latex_name('b', r'$\mathcal{B}$')
        >>> fm
        FisherMatrix(
            array([[1., 0.],
               [0., 2.]]),
            names=array(['a', 'b'], dtype=object),
            latex_names=array(['a', '$\\mathcal{B}$'], dtype=object),
            fiducials=array([2., 3.]))
        """
        if name not in self.names:
            raise ParameterNotFoundError(name, self.names)

        self.latex_names[np.where(self.names == name)[0]] = value

    def is_valid(self):
        """
        Checks whether the values make a valid Fisher object.
        A (square) matrix is a Fisher matrix if it satisifies the following two
        criteria:

        * it is symmetric
        * it is positive semi-definite

        Returns
        -------
        bool
            Whether or nor the Fisher object is valid.

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))
        >>> m.is_valid()
        True
        >>> FisherMatrix(np.diag([-1, 3])).is_valid()
        False
        """
        return is_symmetric(self.values) and is_positive_semidefinite(self.values)

    def sort(
        self,
        **kwargs,
    ) -> FisherMatrix:
        """
        Sorts the Fisher object by name according to some criterion.

        Parameters
        ----------
        **kwargs
            all of the other keyword arguments for the Python builtin <a
            href="https://docs.python.org/3/library/functions.html#sorted"
            target="_blank" rel="noopener noreferrer">`sorted`</a>.
            If none are specified, will sort according to the names of the parameters.
            In the special case that the value of the keyword `key` is set to
            either `'fiducials'` or `'latex_names'`, it will sort according to those.
            In the second special case that the value of the keyword `key` is
            set to an array of integers of equal size as the Fisher object, sorts them
            according to those instead.
            In the third (and final) special case that the value of the keyword
            `key` is set to an array strings matching those of parameter names
            of the FisherMatrix, the items are sorted according to those
            values.

        Returns
        -------
        FisherMatrix
            the Fisher object with sorted names

        Raises
        ------
        ValueError
            if `key` does not match the names of the Fisher object

        TypeError
            if `key` is not a callable

        Examples
        --------
        Define a Fisher matrix:
        >>> m = FisherMatrix(np.diag([3, 1, 2]), names=list('sdf'),
        ... latex_names=['hjkl', 'qwe', 'll'], fiducials=[8, 7, 3])

        Sort according to values of the fiducials:
        >>> m.sort(key='fiducials')
        FisherMatrix(
            array([[2., 0., 0.],
               [0., 1., 0.],
               [0., 0., 3.]]),
            names=array(['f', 'd', 's'], dtype=object),
            latex_names=array(['ll', 'qwe', 'hjkl'], dtype=object),
            fiducials=array([3., 7., 8.]))

        Sort according to the LaTeX names (alphabetically):
        >>> m.sort(key='latex_names')
        FisherMatrix(
            array([[3., 0., 0.],
               [0., 2., 0.],
               [0., 0., 1.]]),
            names=array(['s', 'f', 'd'], dtype=object),
            latex_names=array(['hjkl', 'll', 'qwe'], dtype=object),
            fiducials=array([8., 3., 7.]))
        """
        allowed_keys = ("fiducials", "latex_names")

        # something that can be passed to `sorted`
        if ("key" not in kwargs) or ("key" in kwargs and callable(kwargs["key"])):
            names = sorted(self.names, **kwargs)
            index = get_index_of_other_array(names, self.names)

        # an integer index
        if (
            "key" in kwargs
            and is_iterable(kwargs["key"])
            and all(hasattr(_, "__index__") for _ in kwargs["key"])
        ):
            index = np.array(kwargs["key"], dtype=int)
            names = self.names[index]

        # either 'fiducials' or 'latex_names'
        elif (
            "key" in kwargs
            and isinstance(kwargs["key"], str)
            and kwargs["key"] in allowed_keys
        ):
            index = np.argsort(getattr(self, kwargs["key"]))
            if "reversed" in kwargs and kwargs["reversed"] is True:
                index = np.flip(index)
            names = self.names[index]

        # the names themselves, in any order
        elif "key" in kwargs and is_iterable(kwargs["key"]):
            if set(kwargs["key"]) != set(self.names):
                raise ValueError(
                    f"The object '{kwargs['key']}' appears to be an iterable, "
                    f"but does not contain all of the names in the Fisher object ({self.names}); "
                    "if you meant to sort by the names, make sure they are all in the passed array!"
                )
            index = get_index_of_other_array(kwargs["key"], self.names)
            names = self.names[index]

        # something else should throw an error
        elif "key" in kwargs:
            raise TypeError(
                f"The object {kwargs['key']} is not a callable, a list of names, "
                f"or one of {allowed_keys}"
            )

        latex_names = self.latex_names[index]
        fiducials = self.fiducials[index]
        values = reindex_array(self.values, index)

        return self.__class__(
            values,
            names=names,
            latex_names=latex_names,
            fiducials=fiducials,
        )

    def __eq__(self, other):
        """
        The equality operator.
        Returns `True` if the operands have the following properties:

        * are instances of FisherMatrix
        * have same names (potentially shuffled)
        * have same dimensionality
        * have same fiducials (potentially shuffled)
        * have same values (potentially shuffled)
        """
        if set(self.names) != set(other.names):
            return False

        # index for re-shuffling parameters
        index = get_index_of_other_array(self.names, other.names)

        return (
            isinstance(other, self.__class__)
            and self.ndim == other.ndim
            and len(self) == len(other)
            and set(self.names) == set(other.names)
            and np.allclose(
                self.fiducials,
                other.fiducials[index],
            )
            and np.allclose(
                self.values,
                reindex_array(other.values, index),
            )
        )

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the Fisher object (for now always 2).
        """
        return np.ndim(self._values)

    def __len__(self):
        """
        Returns the number of parameters in the Fisher object.
        """
        return self._size

    def is_diagonal(self):
        """
        Checks whether the Fisher matrix is diagonal.

        Returns
        -------
        bool
        """
        return np.all(self.values == np.diag(np.diagonal(self.values)))

    def diagonal(self, **kwargs):
        """
        Returns the diagonal elements of the Fisher object as a numpy array.

        Returns
        -------
        array_like : float
            The diagonal elements as a numpy array.
        """
        return np.diag(self.values, **kwargs)

    def drop(
        self,
        *names: str,
        invert: bool = False,
        ignore_errors: bool = False,
    ) -> FisherMatrix:
        """
        Removes parameters from the Fisher object.

        Parameters
        ----------
        names : str
            the names of the parameters to drop.
            If passing a list or a tuple, make sure to unpack it using the
            asterisk (*).

        invert : bool, optional
            whether to drop all the parameters NOT in names (the complement)
            (default: False)

        ignore_errors : bool, optional
            should non-existing parameters be ignored (default: False)

        Returns
        -------
        FisherMatrix
            the Fisher object with removed names

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))

        Drop `p1` and `p3`:
        >>> m.drop('p1', 'p3')
        FisherMatrix(
            array([[2.]]),
            names=array(['p2'], dtype=object),
            latex_names=array(['p2'], dtype=object),
            fiducials=array([0.]))

        Same result, but note the asterisk (`*`):
        >>> m.drop(*['p1', 'p3'])
        FisherMatrix(
            array([[2.]]),
            names=array(['p2'], dtype=object),
            latex_names=array(['p2'], dtype=object),
            fiducials=array([0.]))

        Drop everything *except* `p1` and `p3`:
        >>> m.drop('p1', 'p3', invert=True)
        FisherMatrix(
            array([[1., 0.],
               [0., 3.]]),
            names=array(['p1', 'p3'], dtype=object),
            latex_names=array(['p1', 'p3'], dtype=object),
            fiducials=array([0., 0.]))
        """
        if not ignore_errors and not set(names).issubset(set(self.names)):
            raise ValueError(
                f"The names ({list(names)}) are not a strict subset "
                f"of the parameter names in the Fisher object ({self.names}); "
                "you can pass `ignore_errors=True` to ignore this error"
            )

        if ignore_errors:
            names = tuple(np.array([name for name in names if name in self.names]))

        if invert is True:
            names = tuple(set(names) ^ set(self.names))

        if set(names) == set(self.names):
            raise ValueError("Unable to remove all parameters")

        index = [np.array(np.where(self.names == name), dtype=int) for name in names]

        values = self.values
        for dim in range(self.ndim):
            values = np.delete(
                values,
                index,
                axis=dim,
            )

        fiducials = np.delete(self.fiducials, index)
        latex_names = np.delete(self.latex_names, index)
        names = tuple(np.delete(self.names, index))

        return self.__class__(
            values,
            names=names,
            latex_names=latex_names,
            fiducials=fiducials,
        )

    def trace(
        self,
        **kwargs,
    ):
        """
        Returns the trace of the Fisher object as a numpy array.
        """
        return np.trace(self.values, **kwargs)

    def eigenvalues(self):
        """
        Returns the eigenvalues of the Fisher object as a numpy array.
        """
        return np.linalg.eigvalsh(self.values)

    def eigenvectors(self):
        """
        Returns the right eigenvectors of the Fisher object as a numpy array.
        """
        return np.linalg.eigh(self.values)[-1]

    def condition_number(self):
        r"""
        Returns the condition number of the matrix with respect to the $L^2$ norm.

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2, 3]))
        >>> fm.condition_number()
        3.0
        """
        values = np.abs(self.eigenvalues())
        return np.max(values) / np.min(values)

    def inverse(self):
        """
        Returns the inverse of the Fisher matrix.

        Returns
        -------
        array_like : float
            the covariance matrix as a numpy array

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2, 5]))
        >>> fm.inverse()
        array([[1. , 0. , 0. ],
               [0. , 0.5, 0. ],
               [0. , 0. , 0.2]])
        """
        return np.linalg.inv(self.values)

    def determinant(self):
        """
        Returns the determinant of the matrix.

        Returns
        -------
        float
        """
        return np.linalg.det(self.values)

    def constraints(
        self,
        name: Optional[str] = None,
        marginalized: bool = True,
        sigma: Optional[float] = None,
        p: Optional[float] = None,
    ):
        r"""
        Returns the constraints on a parameter as a float, or on all of them
        as a numpy array if `name` is not specified.

        Parameters
        ----------
        name : str, optional
            the name of the parameter for which we we want the constraints
            (default: None)

        marginalized : bool, optional
            whether we want the marginalized or the unmarginalized
            constraints (default: True).

        sigma : float, optional
            how many sigmas away (default: 1).

        p : float, optional
            the confidence interval (p-value) (default: None).

        Returns
        -------
        float or array_like of float

        Notes
        -----
        The user should specify either `sigma` or `p`, but not both
        simultaneously.
        If neither are specified, defaults to `sigma=1`.

        The relationship between `p` and `sigma` is defined via:
        $$
            p(\sigma) = \int\limits_{\mu - \sigma}^{\mu + \sigma}
                        f(x, \mu, 1)\, \mathrm{d}x
                      = \mathrm{Erf}(\sigma / \sqrt{2})
        $$
        and therefore the inverse is simply:
        $$
            \sigma(p) = \sqrt{2}\, \mathrm{Erf}^{-1}(p)
        $$
        The values of `p` corresponding to 1, 2, 3 `sigma` are roughly
        0.683, 0.954, and 0.997, respectively.

        Examples
        --------
        Get (marginalized by default) constraints for all parameters:
        >>> m = FisherMatrix([[3, -2], [-2, 5]])
        >>> m.constraints()
        array([0.67419986, 0.52223297])

        Get the unmarginalized constraints instead:
        >>> m.constraints(marginalized=False)
        array([0.57735027, 0.4472136 ])

        Get the unmarginalized constraints for a single parameter:
        >>> m.constraints('p1', marginalized=False)
        array([0.57735027])

        Get the marginalized constraints for a single parameter:
        >>> m.constraints('p1', p=0.682689) # p-value roughly equal to 1 sigma
        array([0.67419918])
        """
        if sigma is not None and p is not None:
            raise ValueError(
                "Cannot specify both `p` and `sigma` simultaneously; "
                "please specify at most one of those"
            )

        if p is not None:
            if not 0 < p < 1:
                raise ValueError(
                    f"The value of `p` {p} is outside of the allowed range (0, 1)"
                )
            sigma = np.sqrt(2) * erfinv(p)
        elif sigma is None:
            sigma = 1

        if sigma <= 0:
            raise ValueError(
                f"The value of `sigma` {sigma} is outside of the allowed range (0, infinify)"
            )

        if marginalized:
            inv = self.__class__(
                self.inverse(),
                names=self.names,
                latex_names=self.latex_names,
                fiducials=self.fiducials,
            )
            result = np.sqrt(np.diag(inv.values)) * sigma
        else:
            result = 1.0 / np.sqrt(np.diag(self.values)) * sigma

        if name is not None:
            if name in self.names:
                return result[np.where(self.names == name)]
            raise ParameterNotFoundError(name, self.names)

        return result

    def __add__(
        self,
        other: FisherMatrix,
    ) -> FisherMatrix:
        """
        Returns the result of adding two Fisher objects.

        Notes
        -----
        We can add or subtract Fisher objects which have different parameter
        names.

        Due to the peculiarities of Python's `set` built-in, the ordering of
        names in the result is not guaranteed to be preserved, even for
        identical inputs. This is not a problem though, since `__eq__` can
        handle parameter shuffling, so if `m1` and `m2` are instances of
        `FisherMatrix`, `m1 + m2 == m1 + m2` will always yield `True`.
        """
        if isinstance(other, self.__class__):
            values = np.zeros(
                [len(self) + len(other) - len(set(self.names) & set(other.names))] * 2,
                dtype=float,
            )

            names = list(set(self.names) | set(other.names))
            fiducials = []
            latex_names = []
            for name in names:
                fiducial1 = (
                    self.fiducials[np.where(self.names == name)][0]
                    if name in self.names
                    else None
                )
                fiducial2 = (
                    other.fiducials[np.where(other.names == name)][0]
                    if name in other.names
                    else None
                )
                latex_name1 = (
                    self.latex_names[np.where(self.names == name)][0]
                    if name in self.names
                    else None
                )
                latex_name2 = (
                    other.latex_names[np.where(other.names == name)][0]
                    if name in other.names
                    else None
                )

                if (
                    fiducial1 is not None
                    and fiducial2 is not None
                    and not np.allclose(fiducial1, fiducial2)
                ):
                    raise MismatchingValuesError(
                        "fiducial values", fiducial1, fiducial2
                    )

                fiducials.append(fiducial1 if fiducial1 is not None else fiducial2)
                latex_names.append(
                    latex_name1 if latex_name1 is not None else latex_name2
                )

            # TODO make this work for arbitrary number of dimensions
            for (index1, name1), (index2, name2) in product(enumerate(names), repeat=2):
                # case 1: the names are in both
                if (
                    name1 in self.names
                    and name2 in self.names
                    and name1 in other.names
                    and name2 in other.names
                ):
                    values[index1, index2] = self[name1, name2] + other[name1, name2]
                # case 2a: the names are in the left one only
                elif name1 in self.names and name2 in self.names:
                    values[index1, index2] = self[name1, name2]
                # case 2b: the names are in the right one only
                elif name1 in other.names and name2 in other.names:
                    values[index1, index2] = other[name1, name2]
                # case 3: one name is in one, but not the other
                else:
                    values[index1, index2] = 0
        else:
            values = self.values + other
            names = self.names
            fiducials = self.fiducials
            latex_names = self.latex_names

        return self.__class__(
            values,
            names=names,
            latex_names=latex_names,
            fiducials=fiducials,
        )

    def __radd__(
        self,
        other: FisherMatrix,
    ) -> FisherMatrix:
        """
        Addition when the Fisher object is the right operand.
        """
        return self.__add__(other)

    def __neg__(self) -> FisherMatrix:
        """
        Returns the negation of the Fisher object.
        """
        return self.__class__(
            -self.values,
            names=self.names,
            latex_names=self.latex_names,
            fiducials=self.fiducials,
        )

    def __sub__(
        self,
        other: FisherMatrix,
    ) -> FisherMatrix:
        """
        Returns the result of subtracting two Fisher objects.
        """
        return self.__add__(-other)

    def __rsub__(
        self,
        other: FisherMatrix,
    ) -> FisherMatrix:
        """
        Subtraction when the Fisher object is the right operand.
        """
        # this will never be called if we subtract two FisherMatrix instances,
        # so we just need to handle floats
        return -self.__add__(-other)

    def __array_ufunc__(
        self,
        ufunc,
        method,
        *inputs,
        **kwargs,
    ):
        """
        Handles numpy's universal functions.
        For a complete list and explanation, see:
        https://numpy.org/doc/stable/reference/ufuncs.html
        """
        if method == "__call__":
            scalars = []
            for i in inputs:
                if isinstance(i, Number):
                    scalars.append(i)
                elif isinstance(i, self.__class__):
                    scalars.append(i.values)
                else:
                    return NotImplemented

            names = [_.names for _ in inputs if isinstance(_, self.__class__)]

            if not np.all([set(names[0]) == set(_) for _ in names]):
                raise ValueError("Mismatching names")

            # make sure the names have the same _ordering_
            for index, _ in enumerate(inputs):
                if isinstance(_, self.__class__):
                    scalars[index] = _.sort(
                        key=get_index_of_other_array(
                            names[0],
                            _.names,
                        )
                    ).values

            return self.__class__(
                ufunc(*scalars, **kwargs),
                names=self.names,
                latex_names=self.latex_names,
                fiducials=self.fiducials,
            )

        return NotImplemented

    def __pow__(
        self,
        other: Union[float, int],
    ) -> FisherMatrix:
        """
        Raises the Fisher object to some power.
        """
        return self.__class__(
            np.power(self.values, other),
            names=self.names,
            latex_names=self.latex_names,
            fiducials=self.fiducials,
        )

    def __matmul__(
        self,
        other: FisherMatrix,
    ):
        """
        Matrix multiplies two Fisher objects.
        """
        # make sure they have the right parameters
        if set(other.names) != set(self.names):
            raise MismatchingValuesError("parameter name", other.names, self.names)

        index = get_index_of_other_array(self.names, other.names)

        # make sure the fiducials match
        fiducials = other.fiducials[index]

        if not np.allclose(fiducials, self.fiducials):
            raise MismatchingValuesError("fiducial value", fiducials, self.fiducials)

        values = self.values @ reindex_array(other.values, index)

        return self.__class__(
            values,
            names=self.names,
            latex_names=self.latex_names,
            fiducials=self.fiducials,
        )

    def __truediv__(
        self,
        other: Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of dividing a Fisher object by a number, or another
        Fisher object (element-wise).
        """
        # we can only divide two objects if they have the same dimensions and sizes
        if isinstance(other, self.__class__):
            # maybe it's a FisherMatrix
            # make sure they have the right parameters
            if set(other.names) != set(self.names):
                raise MismatchingValuesError("parameter name", other.names, self.names)

            index = get_index_of_other_array(self.names, other.names)

            # make sure the fiducials match
            fiducials = other.fiducials[index]

            if not np.allclose(fiducials, self.fiducials):
                raise MismatchingValuesError(
                    "fiducial value", fiducials, self.fiducials
                )

            if other.ndim == self.ndim:
                values = self.values / reindex_array(other.values, index)
        else:
            values = self.values / other

        return self.__class__(
            values,
            names=self.names,
            latex_names=self.latex_names,
            fiducials=self.fiducials,
        )

    def __mul__(
        self,
        other: Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of multiplying a Fisher object by a number, or
        another Fisher object (element-wise).
        """
        # we can only multiply two objects if they have the same dimensions and sizes
        if isinstance(other, self.__class__):
            # maybe it's a FisherMatrix
            # make sure they have the right parameters
            if set(other.names) != set(self.names):
                raise MismatchingValuesError("parameter name", other.names, self.names)

            index = get_index_of_other_array(self.names, other.names)

            # make sure the fiducials match
            fiducials = other.fiducials[index]

            if not np.allclose(fiducials, self.fiducials):
                raise MismatchingValuesError(
                    "fiducial value", fiducials, self.fiducials
                )

            if other.ndim == self.ndim:
                values = self.values * reindex_array(other.values, index)
        else:
            values = self.values * other

        return self.__class__(
            values,
            names=self.names,
            latex_names=self.latex_names,
            fiducials=self.fiducials,
        )

    def __rmul__(
        self,
        other: Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of multiplying a number by a Fisher object, or
        another Fisher object (element-wise).
        """
        return self.__mul__(other)

    def reparametrize(
        self,
        jacobian,
        names: Optional[Collection[str]] = None,
        latex_names: Optional[Collection[str]] = None,
        fiducials: Optional[Collection[float]] = None,
    ) -> FisherMatrix:
        """
        Returns a new Fisher object with parameters `names`, which are
        related to the old ones via the transformation `jacobian`.
        See the <a
        href="https://en.wikipedia.org/w/index.php?title=Fisher_information&oldid=1063384000#Reparametrization"
        target="_blank" rel="noopener noreferrer">Wikipedia article</a> for
        more information.

        Parameters
        ----------
        transformation : array_like of float
            the Jacobian of the transformation

        names : array_like of str, optional
            list of new names for the Fisher object (default: None). If None,
            uses the old names.

        latex_names: array_like of str, optional
            list of new LaTeX names for the Fisher object (default: None). If
            None, and `names` is set, uses those instead, otherwise uses the
            old LaTeX names.

        fiducials : array_like of float, optional
            the new values of the fiducials (default: None). If None, defaults
            to old values.

        Returns
        -------
        FisherMatrix
            The new Fisher object with the specified names, LaTeX names, and
            fiducials.

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2]))
        >>> fm
        FisherMatrix(
            array([[1., 0.],
               [0., 2.]]),
            names=array(['p1', 'p2'], dtype=object),
            latex_names=array(['p1', 'p2'], dtype=object),
            fiducials=array([0., 0.]))

        Define a Jacobian:
        >>> jac = [[1, 4], [3, 2]]

        Reparametrize according to that Jacobian:
        >>> fm.reparametrize(jac, names=['a', 'b'])
        FisherMatrix(
            array([[19., 16.],
               [16., 24.]]),
            names=array(['a', 'b'], dtype=object),
            latex_names=array(['a', 'b'], dtype=object),
            fiducials=array([0., 0.]))
        """
        values = np.transpose(jacobian) @ self.values @ jacobian

        if names is not None:
            if len(set(names)) != np.shape(jacobian)[-1]:
                raise MismatchingSizeError(names)
            if latex_names is not None:
                if len(set(latex_names)) != np.shape(jacobian)[-1]:
                    raise MismatchingSizeError(latex_names)
            else:
                latex_names = copy.deepcopy(names)
        else:
            # we don't transform the names
            names = copy.deepcopy(self.names)
            latex_names = copy.deepcopy(self.latex_names)

        if fiducials is not None:
            if len(fiducials) != np.shape(jacobian)[-1]:
                raise MismatchingSizeError(fiducials)
        else:
            fiducials = copy.deepcopy(self.fiducials)

        return self.__class__(
            values,
            names=names,
            latex_names=latex_names,
            fiducials=fiducials,
        )

    def to_dict(self) -> dict:
        """
        Returns the Fisher object as a dictionary.
        """
        return dict(
            values=self.values,
            names=self.names,
            latex_names=self.latex_names,
            fiducials=self.fiducials,
        )

    def to_file(
        self, path: Union[str, Path], metadata: Optional[Mapping[str, Any]] = None
    ) -> dict[str, Any]:
        r"""
        Saves the Fisher object to a file (UTF-8 encoded).
        The format used is a simple JSON file, containing at least the values of the
        Fisher object, the names of the parameters, the LaTeX names, and the
        fiducial values.

        Parameters
        ----------
        path : str or Path
            the path to save the data to.

        metadata : dict, optional
            any metadata that should be associated to the object saved, in the
            form of a dictionary-like object

        Returns
        -------
        dict
            the dictionary that was saved

        Examples
        --------
        >>> from pprint import pprint
        >>> fm = FisherMatrix(np.diag([1, 2]),
        ... names=['a', 'b'], latex_names=[r'$\mathbf{A}$', r'$\mathbf{B}$'])
        >>> pprint(fm.to_file('example_matrix.json'), sort_dicts=False) # doctest: +SKIP
        {'values': array([[1., 0.],
               [0., 2.]]),
         'names': array(['a', 'b'], dtype=object),
         'latex_names': array(['$\\mathbf{A}$', '$\\mathbf{B}$'], dtype=object),
         'fiducials': array([0., 0.])}

        There is also a convenience function for reading it:
        >>> fm_read = FisherMatrix.from_file('example_matrix.json') # doctest: +SKIP

        Verify it's the same object:
        >>> fm == fm_read # doctest: +SKIP
        True
        """
        data = self.to_dict()

        if metadata is not None:
            data = {
                **data,
                **{"metadata": metadata},
            }

        with open(path, "w", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(data, indent=4, cls=FisherEncoder))

        return data

    def marginalize_over(
        self,
        *names: str,
        invert: bool = False,
        ignore_errors: bool = False,
    ) -> FisherMatrix:
        """
        Perform marginalization over some parameters.

        Parameters
        ----------
        names : str
            the names of the parameters to marginalize over

        invert : bool, optional
            whether to marginalize over all the parameters NOT in names (the complement) (default: False)

        ignore_errors : bool, optional
            should non-existing parameters be ignored (default: False)

        Returns
        -------
        FisherMatrix
            The Fisher object with the specified parameters marginalized over.

        Examples
        --------
        Generate a Fisher object using a random orthogonal matrix:
        >>> rm = np.array([[-0.0942819 ,  0.56164959,  0.7638867 ,  0.18138749, -0.24338518],
        ... [ 0.22058622,  0.74746891, -0.33297073, -0.47219083,  0.24248352],
        ... [-0.23923712,  0.18017836, -0.44325216,  0.05300493, -0.84321964],
        ... [-0.25595188,  0.30540004, -0.32672895,  0.77839888,  0.35855659],
        ... [ 0.90537665,  0.0103218 , -0.04881925,  0.36799464, -0.20587183]])
        >>> fm = FisherMatrix(rm.T @ np.diag([1, 2, 3, 7, 6]) @ rm)

        Marginalize over some parameters:
        >>> fm.marginalize_over('p1', 'p2')
        FisherMatrix(
            array([[ 1.67715593, -1.01556087,  0.30020774],
               [-1.01556087,  4.92788973,  0.91219829],
               [ 0.30020774,  0.91219829,  3.1796454 ]]),
            names=array(['p3', 'p4', 'p5'], dtype=object),
            latex_names=array(['p3', 'p4', 'p5'], dtype=object),
            fiducials=array([0., 0., 0.]))

        Marginalize over all parameters which are NOT `p1` or `p2`:
        >>> fm.marginalize_over('p1', 'p2', invert=True)
        FisherMatrix(
            array([[ 5.04480061, -0.04490451],
               [-0.04490451,  1.61599083]]),
            names=array(['p1', 'p2'], dtype=object),
            latex_names=array(['p1', 'p2'], dtype=object),
            fiducials=array([0., 0.]))
        """
        inv = self.__class__(
            self.inverse(),
            names=self.names,
            latex_names=self.latex_names,
            fiducials=self.fiducials,
        )
        if invert is True:
            names = tuple(set(names) ^ set(self.names))
        fisher = inv.drop(*names, ignore_errors=ignore_errors)
        return self.__class__(
            fisher.inverse(),
            names=fisher.names,
            latex_names=fisher.latex_names,
            fiducials=fisher.fiducials,
        )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
    ):
        """
        Reads a Fisher object from a file.

        Parameters
        ----------
        path : str or Path
            the path to the file

        Returns
        -------
        FisherMatrix

        Notes
        -----
        Any metadata is ignored when reading the file.
        If you want to read the metadata as well, use something like the following:

        >>> import json
        >>> with open(<filename>, 'r') as f: # doctest: +SKIP
        ...     data = json.loads(f.read()) # doctest: +SKIP

        Then `data` will contain a dictionary with all of the data from the
        file, which can be easily parsed.
        """
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.loads(file_handle.read())

        return cls.from_dict(data)

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ):
        """
        Reads a Fisher object from a dictionary.

        Parameters
        ----------
        data
            the dictionary

        Raises
        ------
        KeyError
            if the dictionary contains malformed data

        Returns
        -------
        FisherMatrix
        """
        return cls(
            data["values"],
            names=data["names"],
            latex_names=data["latex_names"],
            fiducials=data["fiducials"],
        )

    def correlation_matrix(self):
        r"""
        Computes the correlation matrix from the Fisher matrix, whose entries
        are given by:
        $$
            \mathsf{P}_{ij} \equiv \frac{\mathsf{C}_{ij}}{\sqrt{\mathsf{C}_{ii} \mathsf{C}_{jj}}}
        $$
        where $\mathsf{C}_{ij}$ is the $(i, j)$ element of the covariance
        matrix (the inverse of the Fisher matrix).

        Returns
        -------
        array_like : float
            the correlation matrix as a numpy array
        """
        inv = self.inverse()
        diag = np.diag(inv)

        return inv / np.sqrt(np.outer(diag, diag))

    def correlation_coefficient(
        self,
        name1: str,
        name2: str,
    ) -> float:
        r"""
        Returns the correlation coefficient between two parameters.

        Parameters
        ----------
        name1 : str
            the name of the first parameter

        name2 : str
            the name of the second parameter

        Returns
        -------
        float
        """
        return self.__class__(self.correlation_matrix(), names=self.names)[name1, name2]

    def figure_of_merit(self) -> float:
        r"""
        Computes the figure of merit (FoM), defined here as:
        $$
            \mathrm{FoM} \equiv \frac{1}{2} \log \left[ \det \mathsf{F} \right]
        $$
        where $\mathsf{F}$ denotes the Fisher matrix.

        Returns
        -------
        float

        Notes
        -----
        Occasionally, the FoM (also sometimes called the signal-to-noise,
        $\mathrm{SNR}$), is defined as:
        $$
            \mathrm{SNR} = \sqrt{\det \mathsf{F}}
        $$
        The two are related by a simple transformation:
        $$
            \mathrm{SNR} = \exp{\left( \mathrm{FoM} \right)}
        $$
        """
        return np.linalg.slogdet(self.values)[-1] / 2

    def figure_of_correlation(self) -> float:
        r"""
        Computes the figure of correlation (FoC), defined here as:
        $$
            \mathrm{FoC} \equiv - \frac{1}{2} \log \left[ \det \mathsf{P} \right]
        $$
        where $\mathsf{P}$ denotes the correlation matrix (see
        `correlation_matrix` for definition of $\mathsf{P}$).

        Returns
        -------
        float
        """
        return -np.linalg.slogdet(self.correlation_matrix())[-1] / 2
