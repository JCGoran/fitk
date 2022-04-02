"""
Package for performing operations on Fisher objects.
See here for documentation of `FisherMatrix`.
"""

# needed for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from collections import abc
import copy
from itertools import \
    permutations, \
    product
import json
from numbers import Number
import os
from typing import \
    Mapping, \
    Collection, \
    Optional, \
    Union, \
    Tuple

# third party imports
import numpy as np
from scipy.special import erfinv # pylint: disable=no-name-in-module
import matplotlib.pyplot as plt
from matplotlib.patheffects import Stroke, Normal
from matplotlib import colors

# first party imports
from fitk.fisher_utils import \
    float_to_latex, \
    ParameterNotFoundError, \
    MismatchingSizeError, \
    HTMLWrapper, \
    make_html_table, \
    make_default_names, \
    is_square, \
    is_symmetric, \
    is_positive_semidefinite, \
    get_index_of_other_array, \
    reindex_array, \
    get_default_rcparams, \
    jsonify



def _process_fisher_mapping(value : abc.Mapping):
    """
    Processes a mapping/dict and returns the sanitized output.
    """
    if 'name' not in value:
        raise ValueError(
            'The mapping/dict must contain at least the key `name`'
        )

    name = value['name']
    latex_name = value.get('latex_name', name)
    fiducial = value.get('fiducial', 0)

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
    >>> fm = FisherMatrix(np.diag([5, 4]))

    The object has a friendly representation in the interactive session:
    >>> fm
    FisherMatrix(array([[5., 0.],
           [0., 4.]]), names=array(['p1', 'p2'], dtype=object), latex_names=array(['p1', 'p2'], dtype=object), fiducial=array([0., 0.]))

    The string representation is probably more readable:
    >>> print(fm)
    FisherMatrix([[5. 0.]
     [0. 4.]], names=['p1' 'p2'], latex_names=['p1' 'p2'], fiducial=[0. 0.])

    List the names:
    >>> fm.names
    array(['p1', 'p2'], dtype=object)

    List the values of the Fisher object:
    >>> fm.values
    array([[5., 0.],
           [0., 4.]])

    List the values of the fiducials:
    >>> fm.fiducial
    array([0., 0.])

    Names can be changed in bulk (ditto for fiducial and values; dimension must of course match the original):
    >>> fm = FisherMatrix(np.diag([5, 4]))
    >>> fm.names = ['x', 'y']
    >>> fm.latex_names = [r'$\mathbf{X}$', r'$\mathbf{Y}$']
    >>> fm
    FisherMatrix(array([[5., 0.],
           [0., 4.]]), names=array(['x', 'y'], dtype=object), latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object), fiducial=array([0., 0.]))

    We can get and set individual elements of the matrix using dict-like notation:
    >>> fm = FisherMatrix(np.diag([5, 4]), names=['x', 'y'])
    >>> fm['x', 'x']
    5.0

    The off-diagonal elements are automatically updated when using the setter:
    >>> fm = FisherMatrix(np.diag([5, 4]), names=['x', 'y'])
    >>> fm['x', 'y'] = -2
    >>> fm
    FisherMatrix(array([[ 5., -2.],
           [-2.,  4.]]), names=array(['x', 'y'], dtype=object), latex_names=array(['x', 'y'], dtype=object), fiducial=array([0., 0.]))

    We can select submatrices by index:
    >>> fm = FisherMatrix(np.diag([1, 2, 3]), names=['x', 'y', 'z'])
    >>> fm[1:]
    FisherMatrix(array([[2., 0.],
           [0., 3.]]), names=array(['y', 'z'], dtype=object), latex_names=array(['y', 'z'], dtype=object), fiducial=array([0., 0.]))

    Fisher object with parameter names:
    >>> fm = FisherMatrix(np.diag([5, 4]), names=['x', 'y'], latex_names=['$\\mathbf{X}$', '$\\mathbf{Y}$'])
    >>> fm_with_names = FisherMatrix(np.diag([1, 2]), names=['x', 'y'])

    We can add Fisher objects (ordering of names is taken care of):
    >>> fm + fm_with_names # doctest: +SKIP
    FisherMatrix(array([[6., 0.],
           [0., 6.]]), names=array(['x', 'y'], dtype=object), latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object), fiducial=array([0., 0.]))

    We can also do element-wise multiplication or division:
    >>> fm * fm_with_names
    FisherMatrix(array([[5., 0.],
           [0., 8.]]), names=array(['x', 'y'], dtype=object), latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object), fiducial=array([0., 0.]))

    Furthermore, we can do matrix multiplication:
    >>> fm @ fm_with_names
    FisherMatrix(array([[5., 0.],
           [0., 8.]]), names=array(['x', 'y'], dtype=object), latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object), fiducial=array([0., 0.]))

    We can also perform standard matrix operations like the trace, eigenvalues, determinant:
    >>> fm.trace()
    9.0
    >>> fm.eigenvalues()
    array([4., 5.])
    >>> fm.determinant()
    19.999999999999996

    We can also take the matrix inverse:
    >>> fm.inverse()
    FisherMatrix(array([[0.2 , 0.  ],
           [0.  , 0.25]]), names=array(['x', 'y'], dtype=object), latex_names=array(['$\\mathbf{X}$', '$\\mathbf{Y}$'], dtype=object), fiducial=array([0., 0.]))

    We can drop parameters from the object:
    >>> fm.drop('x')
    FisherMatrix(array([[4.]]), names=array(['y'], dtype=object), latex_names=array(['$\\mathbf{Y}$'], dtype=object), fiducial=array([0.]))

    We can save it to a file (the returned value is the dictionary that was saved):
    >>> fm.to_file('example_matrix.json', overwrite=True)
    {'values': [[5.0, 0.0], [0.0, 4.0]], 'names': ['x', 'y'], 'latex_names': ['$\\mathbf{X}$', '$\\mathbf{Y}$'], 'fiducial': [0.0, 0.0]}

    Loading is performed by a class method `from_file`:
    >>> fm_new = FisherMatrix.from_file('example_matrix.json')
    """

    def __init__(
        self,
        values : Collection,
        names : Optional[Collection[str]] = None,
        latex_names : Optional[str] = None,
        fiducial : Optional[Collection[float]] = None,
    ):
        r"""
        Constructor for Fisher object.

        Parameters
        ----------
        values : array-like
            The values of the Fisher object.

        names : array-like iterable of `str`, default = None
            The names of the parameters.
            If not specified, defaults to `p1, ..., pn`.

        latex_names : array-like iterable of `str`, default = None
            The LaTeX names of the parameters.
            If not specified, defaults to `names`.

        fiducial : array-like iterable of `float`, default = None
            The fiducial values of the parameters. If not specified, default to
            0 for all parameters.

        metadata : Optional[dict], default = None
            any metadata associated with the Fisher object

        Raises
        ------
        * `ValueError` if the input array has the wrong dimensionality (not 2)
        * `ValueError` if the object is not square-like
        * `MismatchingSizeError` if the sizes of the array of names, values, LaTeX names and fiducials do not match

        Examples
        --------
        Specify a Fisher object without names:
        >>> FisherMatrix(np.diag([1, 2, 3]))
        FisherMatrix(array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]]), names=array(['p1', 'p2', 'p3'], dtype=object), latex_names=array(['p1', 'p2', 'p3'], dtype=object), fiducial=array([0., 0., 0.]))

        Specify a Fisher object with names, LaTeX names, and fiducials:
        >>> FisherMatrix(np.diag([1, 2]), names=['alpha', 'beta'],
        ... latex_names=[r'$\alpha$', r'$\beta$'], fiducial=[-3, 2])
        FisherMatrix(array([[1., 0.],
               [0., 2.]]), names=array(['alpha', 'beta'], dtype=object), latex_names=array(['$\\alpha$', '$\\beta$'], dtype=object), fiducial=array([-3.,  2.]))
        """

        if np.ndim(values) != 2:
            raise ValueError(
                f'The object {values} is not 2-dimensional'
            )

        if not is_square(values):
            raise ValueError(
                f'The object {values} is not square-like'
            )

        # try to treat it as an array-like object
        self._values = np.array(values, dtype=float)

        self._size = np.shape(self._values)[0]
        self._ndim = np.ndim(self._values)

        # setting the fiducial
        if fiducial is None:
            self._fiducial = np.zeros(self._size, dtype=float)
        else:
            self._fiducial = np.array(fiducial, dtype=float)

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
            len(_) == self._size \
            for _ in (self._names, self._fiducial, self._latex_names)
        ):
            raise MismatchingSizeError(
                self._values[0],
                self._names,
                self._fiducial,
                self._latex_names,
            )


    def rename(
        self,
        names : Mapping[str, Union[str, abc.Mapping]],
        ignore_errors : bool = False,
    ) -> FisherMatrix:
        """
        Returns a Fisher object with new names.

        Parameters
        ----------
        names : Mapping[str, Union[str, abc.Mapping]]
            a mapping (dictionary-like object) between the old names and the
            new ones. The values it maps to can either be a string (the new name), or a dict
            with keys `name`, `latex_name`, and `fiducial` (only `name` is mandatory).

        ignore_errors : bool, default = False
            if set to True, will not raise an error if a parameter doesn't exist

        Returns
        -------
        Instance of `FisherMatrix`.

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))
        >>> m.rename({'p1' : 'a', 'p2' : dict(name='b', latex_name='$b$', fiducial=2)})
        FisherMatrix(array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]]), names=array(['a', 'b', 'p3'], dtype=object), latex_names=array(['a', '$b$', 'p3'], dtype=object), fiducial=array([0., 2., 0.]))
        """
        # check uniqueness and size
        if len(set(names)) != len(names):
            raise MismatchingSizeError(set(names), names)

        if not ignore_errors:
            for name in names:
                if name not in self.names:
                    raise ParameterNotFoundError(name, self.names)

        names_new = copy.deepcopy(self.names)
        latex_names_new = copy.deepcopy(self.latex_names)
        fiducial_new = copy.deepcopy(self.fiducial)

        for name, value in names.items():
            index = np.where(names_new == name)
            # it's a mapping
            if isinstance(value, abc.Mapping):
                value = _process_fisher_mapping(value)
                latex_names_new[index] = value['latex_name']
                fiducial_new[index] = value['fiducial']
                names_new[index] = value['name']
            # otherwise, it's a string
            else:
                names_new[index] = value
                latex_names_new[index] = value

        return self.__class__(
            self.values,
            names=names_new,
            latex_names=latex_names_new,
            fiducial=fiducial_new,
        )


    def _repr_html_(self):
        """
        Representation of the Fisher object suitable for viewing in Jupyter
        notebook environments.
        """
        header_matrix = '<thead><tr><th></th>' + (
            '<th>{}</th>' * len(self)
        ).format(*self.latex_names) + '</tr></thead>'

        body_matrix = '<tbody>'
        for index, name in enumerate(self.latex_names):
            body_matrix += f'<tr><th>{name}</th>' + (
                '<td>{:.3f}</td>' * len(self)
            ).format(*(self.values[:, index])) + '</tr>'
        body_matrix += '</tbody>'

        html_matrix = f'<table>{header_matrix}{body_matrix}</table>'

        return html_matrix


    def __repr__(self):
        """
        Representation of the Fisher object for non-Jupyter interfaces.
        """
        return 'FisherMatrix(' \
            f'{repr(self.values)}, ' \
            f'names={repr(self.names)}, ' \
            f'latex_names={repr(self.latex_names)}, ' \
            f'fiducial={repr(self.fiducial)})'


    def __str__(self):
        """
        String representation of the Fisher object.
        """
        return 'FisherMatrix(' \
            f'{self.values}, ' \
            f'names={self.names}, ' \
            f'latex_names={self.latex_names}, ' \
            f'fiducial={self.fiducial})'



    def __getitem__(
        self,
        keys : Union[Tuple[str], slice],
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
            fiducial = self.fiducial[sl]

            return self.__class__(
                values,
                names=names,
                latex_names=latex_names,
                fiducial=fiducial,
            )

        try:
            _ = iter(keys)
        except TypeError as err:
            raise TypeError(err) from err

        # the keys can be a tuple
        if isinstance(keys, tuple):
            if len(keys) != self.ndim:
                raise ValueError(
                    f'Expected {self.ndim} arguments, got {len(keys)}'
                )

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
        keys : Tuple[str],
        value : float,
    ):
        """
        Implements setting of elements in the Fisher object.
        Does not support slicing.
        """
        try:
            _ = iter(keys)
        except TypeError as err:
            raise err

        if len(keys) != self.ndim:
            raise ValueError(f'Got length {len(keys)}')

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


    def is_valid(self):
        """
        Checks whether the values make a valid Fisher object.
        A (square) matrix is a Fisher matrix if it satisifies the following two
        criteria:

        * it is symmetric
        * it is positive semi-definite

        Returns
        -------
        `True` or `False`

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))
        >>> m.is_valid()
        True
        >>> FisherMatrix(np.diag([-1, 3])).is_valid()
        False
        """
        return \
            is_symmetric(self.values) and \
            is_positive_semidefinite(self.values)


    def imshow(
        self,
        colorbar : bool = False,
        show_values : bool = False,
        normalized : bool = False,
        colorbar_space : float = 0.02,
        colorbar_width : float = 0.05,
        colorbar_orientation : str = 'vertical',
        rc : dict = get_default_rcparams(),
        **kwargs,
    ):
        """
        Returns the image of the Fisher object.
        """
        # TODO should only work with 2D data.
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(len(self), len(self)))
            img = ax.imshow(
                self.values,
                interpolation='none',
                norm=colors.CenteredNorm(),
                **kwargs,
            )

            ax.set_xticks(np.arange(len(self)))
            ax.set_yticks(np.arange(len(self)))
            ax.set_xticklabels(self.latex_names)
            ax.set_yticklabels(self.latex_names)

            if colorbar:
                allowed_orientations = ('vertical', 'horizontal')
                if colorbar_orientation not in allowed_orientations:
                    raise ValueError(
                        f'\'{colorbar_orientation}\' is not one of: {allowed_orientations}'
                    )

                if colorbar_orientation == 'vertical':
                    cax = fig.add_axes(
                        [ax.get_position().x1 + colorbar_space,
                        ax.get_position().y0 * 0.997,
                        colorbar_width,
                        ax.get_position().height]
                    )
                    fig.colorbar(img, cax=cax, orientation=colorbar_orientation)
                    cax.set_yticklabels(
                        [f'${float_to_latex(_)}$' for _ in cax.get_yticks()]
                    )
                else:
                    cax = fig.add_axes(
                        [ax.get_position().x0,
                        ax.get_position().y1 + colorbar_space,
                        ax.get_position().width,
                        colorbar_width]
                    )
                    fig.colorbar(img, cax=cax, orientation=colorbar_orientation)
                    cax.xaxis.set_ticks_position('top')
                    cax.set_xticklabels(
                        [f'${float_to_latex(_)}$' for _ in cax.get_xticks()]
                    )

            # whether or not we want to display the actual values inside the
            # matrix
            if show_values:
                mid_coords = np.arange(len(self))
                for index1, index2 in product(range(len(self)), range(len(self))):
                    x = mid_coords[index1]
                    y = mid_coords[index2]
                    value = self.values[index1, index2] / \
                        np.sqrt(
                            self.values[index1, index1] \
                           *self.values[index2, index2]
                        ) if normalized \
                        else self.values[index1, index2]
                    text = ax.text(
                            x, y, f'${float_to_latex(value)}$',
                            ha='center', va='center', color='white',
                    )
                    text.set_path_effects(
                        [Stroke(linewidth=1, foreground="black"), Normal()]
                    )

        return fig


    def sort(
        self,
        **kwargs,
    ) -> FisherMatrix:
        """
        Sorts the Fisher object by name according to some criterion.

        Parameters
        ----------
        **kwargs
            all of the other keyword arguments for the Python builtin `sorted`.
            If none are specified, will sort according to the names of the parameters.
            In the special case that the value of the keyword `key` is set to
            either 'fiducial' or 'latex_names', it will sort according to those.
            In the second special case that the value of the keyword `key` is
            set to an array of integers of equal size as the Fisher object, sorts them
            according to those instead.

        Returns
        -------
        Instance of `FisherMatrix`.

        Examples
        --------
        >>> m = FisherMatrix(np.diag([3, 1, 2]), names=list('sdf'),
        ... latex_names=['hjkl', 'qwe', 'll'], fiducial=[8, 7, 3])
        >>> m.sort(key='fiducial')
        FisherMatrix(array([[2., 0., 0.],
               [0., 1., 0.],
               [0., 0., 3.]]), names=array(['f', 'd', 's'], dtype=object), latex_names=array(['ll', 'qwe', 'hjkl'], dtype=object), fiducial=array([3., 7., 8.]))
        >>> m.sort(key='latex_names')
        FisherMatrix(array([[3., 0., 0.],
               [0., 2., 0.],
               [0., 0., 1.]]), names=array(['s', 'f', 'd'], dtype=object), latex_names=array(['hjkl', 'll', 'qwe'], dtype=object), fiducial=array([8., 3., 7.]))
        """
        allowed_keys = ('fiducial', 'latex_names')
        # an integer index
        if 'key' in kwargs and all(hasattr(_, '__index__') for _ in kwargs['key']):
            index = np.array(kwargs['key'], dtype=int)
            names = self.names[index]
        # either 'fiducial' or 'latex_names'
        elif 'key' in kwargs and kwargs['key'] in allowed_keys:
            index = np.argsort(getattr(self, kwargs['key']))
            if 'reversed' in kwargs and kwargs['reversed'] is True:
                index = np.flip(index)
            names = self.names[index]
        # something that can be passed to `sorted`
        else:
            names = sorted(self.names, **kwargs)
            index = get_index_of_other_array(self.names, names)

        latex_names = self.latex_names[index]
        fiducial = self.fiducial[index]
        values = reindex_array(self.values, index)

        return self.__class__(
            values,
            names=names,
            latex_names=latex_names,
            fiducial=fiducial,
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

        return isinstance(other, self.__class__) \
        and self.ndim == other.ndim \
        and len(self) == len(other) \
        and set(self.names) == set(other.names) \
        and np.allclose(
            self.fiducial,
            other.fiducial[index],
        ) \
        and np.allclose(
            self.values,
            reindex_array(other.values, index),
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


    @property
    def values(self):
        """
        Returns the values in the Fisher object as a numpy array.
        """
        return self._values


    @values.setter
    def values(
        self,
        value,
    ):
        """
        Setter for the values of the Fisher object.
        """
        if self.ndim != np.ndim(value):
            raise ValueError(
                f'The dimensionality of the matrices do not match: {self.ndim} and {np.ndim(value)}'
            )
        if len(self) != len(value):
            raise MismatchingSizeError(self, value)

        if not is_square(value):
            raise ValueError(
                f'{value} is not a square object'
            )

        self._values = np.array(value, dtype=float)


    def is_diagonal(self):
        """
        Checks whether the Fisher matrix is diagonal.

        Returns
        -------
        `True` or `False`
        """
        return np.all(self.values == np.diag(np.diagonal(self.values)))


    def diagonal(self, **kwargs):
        """
        Returns the diagonal elements of the Fisher object as a numpy array.

        Returns
        -------
        array-like of floats
        """
        return np.diag(self.values, **kwargs)


    def drop(
        self,
        *names : str,
        invert : bool = False,
        ignore_errors : bool = False,
    ) -> FisherMatrix:
        """
        Removes parameters from the Fisher object.

        Parameters
        ----------
        names : string-like
            the names of the parameters to drop.
            If passing a list or a tuple, make sure to unpack it using the
            asterisk (*).

        invert : bool, default = False
            whether to drop all the parameters NOT in names (the complement)

        ignore_errors : bool, default = False
            should non-existing parameters be ignored

        Returns
        -------
        Instance of `FisherMatrix`

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))
        >>> m.drop('p1', 'p3')
        FisherMatrix(array([[2.]]), names=array(['p2'], dtype=object), latex_names=array(['p2'], dtype=object), fiducial=array([0.]))
        """
        if not ignore_errors and not set(names).issubset(set(self.names)):
            raise ValueError(
                f'The names ({list(names)}) are not a strict subset ' \
                f'of the parameter names in the Fisher object ({self.names}); ' \
                'you can pass `ignore_errors=True` to ignore this error'
            )

        if ignore_errors:
            names = np.array([name for name in names if name in self.names])

        if invert is True:
            names = set(names) ^ set(self.names)

        # TODO should we remove this?
        if set(names) == set(self.names):
            raise ValueError('Unable to remove all parameters')

        index = [np.array(np.where(self.names == name), dtype=int) for name in names]

        values = self.values
        for dim in range(self.ndim):
            values = np.delete(
                values,
                index,
                axis=dim,
            )

        fiducial = np.delete(self.fiducial, index)
        latex_names = np.delete(self.latex_names, index)
        names = np.delete(self.names, index)

        return self.__class__(
            values,
            names=names,
            latex_names=latex_names,
            fiducial=fiducial,
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
        Returns the condition number of the matrix with respect to the \(L^2\) norm.

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2, 3]))
        >>> fm.condition_number()
        3.0
        """
        values = np.abs(self.eigenvalues())
        return np.max(values) / np.min(values)


    def inverse(self) -> FisherMatrix:
        """
        Returns the inverse of the Fisher matrix.

        Returns
        -------
        Instance of `FisherMatrix`

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2, 5]))
        >>> fm.inverse()
        FisherMatrix(array([[1. , 0. , 0. ],
               [0. , 0.5, 0. ],
               [0. , 0. , 0.2]]), names=array(['p1', 'p2', 'p3'], dtype=object), latex_names=array(['p1', 'p2', 'p3'], dtype=object), fiducial=array([0., 0., 0.]))
        """
        # inverse satisfies properties of Fisher matrix, see:
        # https://math.stackexchange.com/a/26200
        return self.__class__(
            np.linalg.inv(self.values),
            names=self.names,
            latex_names=self.latex_names,
            fiducial=self.fiducial,
        )


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
        name : Optional[str] = None,
        marginalized : bool = True,
        sigma : Optional[float] = None,
        p : Optional[float] = None,
    ):
        r"""
        Returns the constraints on a parameter as a float, or on all of them
        as a numpy array if `name` is not specified.

        Parameters
        ----------
        name : Optional[str] = None
            the name of the parameter for which we we want the constraints

        marginalized : bool, default = True
            whether we want the marginalized or the unmarginalized
            constraints.

        sigma : Optional[float], default = None
            how many sigmas away.

        p : Optional[float], default = None
            the confidence interval (p-value).
            The relationship between `p` and `sigma` is defined via:
            \[
                p(\sigma) = \int\limits_{\mu - \sigma}^{\mu + \sigma}
                            f(x, \mu, 1)\, \mathrm{d}x
                          = \mathrm{Erf}(\sigma / \sqrt{2})
            \]
            and therefore the inverse is simply:
            \[
                \sigma(p) = \sqrt{2}\, \mathrm{Erf}^{-1}(p)
            \]
            The values of `p` corresponding to 1, 2, 3 `sigma` are roughly
            0.683, 0.954, and 0.997, respectively.

        Returns
        -------
        array-like of floats or single float

        Notes
        -----
        The user should specify either `sigma` or `p`, but not both
        simultaneously.
        If neither are specified, defaults to `sigma=1`.

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
                'Cannot specify both `p` and `sigma` simultaneously'
            )

        if p is not None:
            if not 0 < p < 1:
                raise ValueError(
                    f'The value of `p` {p} is outside of the allowed range (0, 1)'
                )
            sigma = np.sqrt(2) * erfinv(p)
        elif sigma is None:
            sigma = 1

        if sigma <= 0:
            raise ValueError(
                f'The value of `sigma` {sigma} is outside of the allowed range (0, infinify)'
            )

        if marginalized:
            inv = self.inverse()
            result = np.sqrt(np.diag(inv.values)) * sigma
        else:
            result = 1. / np.sqrt(np.diag(self.values)) * sigma

        if name is not None:
            if name in self.names:
                return result[np.where(self.names == name)]
            raise ParameterNotFoundError(name, self.names)

        return result


    @property
    def fiducial(self):
        """
        Returns the fiducial values of the Fisher object as a numpy array.
        """
        return self._fiducial


    @fiducial.setter
    def fiducial(
        self,
        value,
    ):
        """
        The setter for the fiducial values of the Fisher object.
        """
        if len(value) != len(self):
            raise MismatchingSizeError(value, self)
        try:
            self._fiducial = np.array([float(_) for _ in value])
        except TypeError as err:
            raise TypeError(err)


    @property
    def names(self):
        """
        Returns the parameter names of the Fisher object.
        """
        return self._names


    @names.setter
    def names(
        self,
        value,
    ):
        """
        The bulk setter for the names.
        The length of the names must be the same as the one of the original
        object.
        """
        if len(set(value)) != len(self):
            raise MismatchingSizeError(set(value), self)
        self._names = np.array(value, dtype=object)


    @property
    def latex_names(self):
        """
        Returns the LaTeX names of the parameters of the Fisher object.
        """
        return self._latex_names


    @latex_names.setter
    def latex_names(
        self,
        value,
    ):
        """
        The bulk setter for the LaTeX names.
        The length of the names must be the same as the one of the original
        object.
        """
        if len(set(value)) != len(self):
            raise MismatchingSizeError(set(value), self)
        self._latex_names = np.array(value, dtype=object)


    def __add__(
        self,
        other : FisherMatrix,
    ) -> FisherMatrix:
        """
        Returns the result of adding two Fisher objects.
        """
        try:
            other = float(other)
        except TypeError as err:
            # make sure the dimensions match
            if other.ndim != self.ndim:
                raise ValueError(
                    f'The dimensions of the objects do not match: {other.ndim} and {self.ndim}'
                )

            # make sure they have the right parameters
            if set(other.names) != set(self.names):
                raise ValueError(
                    f'Incompatible parameter names: {other.names} and {self.names}'
                )

            index = get_index_of_other_array(self.names, other.names)

            # make sure the fiducials match
            fiducial = other.fiducial[index]

            if not np.allclose(fiducial, self.fiducial):
                raise ValueError(
                    f'Incompatible fiducial values: {fiducial} and {self.fiducial}'
                )

            values = self.values + reindex_array(other.values, index)

            return self.__class__(
                values,
                names=self.names,
                latex_names=self.latex_names,
                fiducial=self.fiducial,
            )

        return self.__class__(
            self.values + other,
            names=self.names,
            latex_names=self.latex_names,
            fiducial=self.fiducial,
        )


    def __radd__(
        self,
        other : Union[FisherMatrix, float],
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
            fiducial=self.fiducial,
        )


    def __sub__(
        self,
        other : Union[FisherMatrix, float],
    ) -> FisherMatrix:
        """
        Returns the result of subtracting two Fisher objects.
        """
        return self.__add__(-other)


    def __rsub__(
        self,
        other : Union[FisherMatrix, float],
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
        if method == '__call__':
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
                raise ValueError(
                    'Mismatching names'
                )

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
                fiducial=self.fiducial,
            )

        return NotImplemented


    def __pow__(
        self,
        other : float,
    ) -> FisherMatrix:
        """
        Raises the Fisher object to some power.
        """
        return self.__class__(
            np.power(self.values, other),
            names=self.names,
            latex_names=self.latex_names,
            fiducial=self.fiducial,
        )


    def __matmul__(
        self,
        other : FisherMatrix,
    ):
        """
        Matrix multiplies two Fisher objects.
        """
        # make sure the dimensions match
        if other.ndim != self.ndim:
            raise ValueError(
                f'The dimensions of the objects do not match: {other.ndim} and {self.ndim}'
            )

        # make sure they have the right parameters
        if set(other.names) != set(self.names):
            raise ValueError(
                f'Incompatible parameter names: {other.names} and {self.names}'
            )

        index = get_index_of_other_array(self.names, other.names)

        # make sure the fiducials match
        fiducial = other.fiducial[index]

        if not np.allclose(fiducial, self.fiducial):
            raise ValueError(
                f'Incompatible fiducial values: {fiducial} and {self.fiducial}'
            )

        values = self.values @ reindex_array(other.values, index)

        return self.__class__(
            values,
            names=self.names,
            latex_names=self.latex_names,
            fiducial=self.fiducial,
        )


    def __truediv__(
        self,
        other : Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of dividing a Fisher object by a number, or another
        Fisher object (element-wise).
        """
        # we can only divide two objects if they have the same dimensions and sizes
        try:
            other = float(other)
        except TypeError as err:
            # maybe it's a FisherMatrix
            # make sure they have the right parameters
            if set(other.names) != set(self.names):
                raise ValueError

            index = get_index_of_other_array(self.names, other.names)

            # make sure the fiducials match
            fiducial = other.fiducial[index]

            if not np.allclose(fiducial, self.fiducial):
                raise ValueError

            if other.ndim == self.ndim:
                values = self.values / reindex_array(other.values, index)
            else:
                raise TypeError(err) from err
        else:
            values = self.values / other

        return self.__class__(
            values,
            names=self.names,
            latex_names=self.latex_names,
            fiducial=self.fiducial,
        )


    def __mul__(
        self,
        other : Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of multiplying a Fisher object by a number, or
        another Fisher object (element-wise).
        """
        # we can only multiply two objects if they have the same dimensions and sizes
        try:
            other = float(other)
        except TypeError as err:
            # maybe it's a FisherMatrix
            # make sure they have the right parameters
            if set(other.names) != set(self.names):
                raise ValueError

            index = get_index_of_other_array(self.names, other.names)

            # make sure the fiducials match
            fiducial = other.fiducial[index]

            if not np.allclose(fiducial, self.fiducial):
                raise ValueError

            if other.ndim == self.ndim:
                values = self.values * reindex_array(other.values, index)
            else:
                raise TypeError(err) from err
        else:
            values = self.values * other

        return self.__class__(
            values,
            names=self.names,
            latex_names=self.latex_names,
            fiducial=self.fiducial,
        )


    def __rmul__(
        self,
        other : Union[float, int],
    ) -> FisherMatrix:
        """
        Returns the result of multiplying a number by a Fisher object, or
        another Fisher object (element-wise).
        """
        return self.__mul__(other)


    def reparametrize(
        self,
        jacobian : Collection,
        names : Optional[Collection[str]] = None,
        latex_names : Optional[Collection[str]] = None,
        fiducial : Optional[Collection[float]] = None,
    ) -> FisherMatrix:
        """
        Returns a new Fisher object with parameters `names`, which are
        related to the old ones via the transformation `jacobian`.
        See the [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Fisher_information&oldid=1063384000#Reparametrization) for more information.

        Parameters
        ----------
        transformation : array-like
            the Jacobian of the transformation

        names : array-like, default = None
            list of new names for the Fisher object. If None, uses the old
            names.

        latex_names: array-like, default = None
            list of new LaTeX names for the Fisher object. If None, and
            `names` is set, uses those instead, otherwise uses the old LaTeX names.

        fiducial : array-like, default = None
            the new values of the fiducial. If not set, defaults to old values.

        Returns
        -------
        Instance of `FisherMatrix`.

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2]))
        >>> jac = [[1, 4], [3, 2]]
        >>> fm.reparametrize(jac, names=['a', 'b'])
        FisherMatrix(array([[19., 16.],
               [16., 24.]]), names=array(['a', 'b'], dtype=object), latex_names=array(['a', 'b'], dtype=object), fiducial=array([0., 0.]))
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

        if fiducial is not None:
            if len(fiducial) != np.shape(jacobian)[-1]:
                raise MismatchingSizeError(fiducial)
        else:
            fiducial = copy.deepcopy(self.fiducial)

        return self.__class__(
            values,
            names=names,
            latex_names=latex_names,
            fiducial=fiducial,
        )


    def pprint_eigenvalues(
        self,
        orientation : str = 'horizontal',
        **kwargs,
    ):
        """
        Shotcut for pretty printing eigenvalues.
        """
        fmt_values = kwargs.pop('fmt_values', '{:.3f}')
        return HTMLWrapper(
            make_html_table(
                self.eigenvalues(),
                fmt_values=fmt_values,
            )
        )


    def pprint_constraints(
        self,
        orientation : str = 'horizontal',
        **kwargs,
    ):
        """
        Shortcut for pretty printing constraints.
        """
        fmt_values = kwargs.pop('fmt_values', '{:.3f}')
        return HTMLWrapper(
            make_html_table(
                self.constraints(**kwargs),
                names=self.latex_names,
                fmt_values=fmt_values,
            )
        )


    def pprint_fiducial(
        self,
        orientation : str = 'horizontal',
        **kwargs,
    ):
        """
        Shortcut for pretty printing the fiducial values.
        """
        fmt_values = kwargs.pop('fmt_values', '{:.3f}')
        return HTMLWrapper(
            make_html_table(
                self.fiducial,
                names=self.latex_names,
                fmt_values=fmt_values,
            )
        )


    def to_file(
        self,
        path : str,
        *args : str,
        metadata : dict = {},
        overwrite : bool = False,
    ):
        r"""
        Saves the Fisher object to a file.
        The format used is a simple JSON file, containing at least the values of the
        Fisher object, the names of the parameters, the LaTeX names, and the
        fiducial values.

        Parameters
        ----------
        path : str
            the path to save the data to.

        args : str
            whatever other information about the object needs to be saved.
            Needs to be one of the following: `is_valid`, `eigenvalues`,
            `eigenvectors`, `trace`, `determinant`, or `constraints` (by
            default, the 1$\sigma$ marginalized constraints).

        metadata : dict, default = {}
            any metadata that should be associated to the object saved

        overwrite : bool, default = False
            whether to overwrite the file if it exists

        Returns
        -------
        The dictionary that was saved.

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2]), names=['a', 'b'], latex_names=[r'$\mathbf{A}$', r'$\mathbf{B}$'])
        >>> fm.to_file('example_matrix.json', overwrite=True)
        {'values': [[1.0, 0.0], [0.0, 2.0]], 'names': ['a', 'b'], 'latex_names': ['$\\mathbf{A}$', '$\\mathbf{B}$'], 'fiducial': [0.0, 0.0]}
        >>> fm_read = FisherMatrix.from_file('example_matrix.json') # convenience function for reading it
        >>> fm == fm_read # verify it's the same object
        True
        """
        data = {
            'values' : self.values.tolist(),
            'names' : self.names.tolist(),
            'latex_names' : self.latex_names.tolist(),
            'fiducial' : self.fiducial.tolist(),
        }

        allowed_metadata = {
            'is_valid' : bool,
            'eigenvalues' : np.array,
            'eigenvectors' : np.array,
            'trace' : float,
            'determinant' : float,
            'constraints' : np.array,
        }

        for arg in args:
            if arg not in allowed_metadata:
                raise ValueError(
                    f'name {arg} is not one of {list(allowed_metadata.keys())}'
                )

        for arg in metadata:
            if arg in data:
                raise ValueError(
                    f'name {arg} cannot be one of {list(data.keys())}'
                )

        data = {
            **data,
            **{arg : jsonify(getattr(self, arg)()) for arg in args},
            **metadata,
        }

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f'The file {path} already exists, please pass ' \
                '`overwrite=True` if you wish to explicitly overwrite '
                'the file'
            )

        with open(path, 'w', encoding='utf-8') as file_handle:
            file_handle.write(json.dumps(data, indent=4))

        return data


    def marginalize_over(
        self,
        *names : str,
        invert : bool = False,
        ignore_errors : bool = False,
    ) -> FisherMatrix:
        """
        Perform marginalization over some parameters.

        Parameters
        ----------
        names : str
            the names of the parameters to marginalize over

        invert : bool, default = False
            whether to marginalize over all the parameters NOT in names (the complement)

        ignore_errors : bool, default = False
            should non-existing parameters be ignored

        Returns
        -------
        Instance of `FisherMatrix`.

        Examples
        --------
        Generate a Fisher object using a random orthogonal matrix:
        >>> from scipy.stats import ortho_group
        >>> rm = ortho_group.rvs(5, random_state=12345)
        >>> fm = FisherMatrix(rm.T @ np.diag([1, 2, 3, 7, 6]) @ rm)

        Marginalize over some parameters:
        >>> fm.marginalize_over('p1', 'p2')
        FisherMatrix(array([[ 1.67715591, -1.01556085,  0.30020773],
               [-1.01556085,  4.92788976,  0.91219831],
               [ 0.30020773,  0.91219831,  3.1796454 ]]), names=array(['p3', 'p4', 'p5'], dtype=object), latex_names=array(['p3', 'p4', 'p5'], dtype=object), fiducial=array([0., 0., 0.]))

        Marginalize over all parameters which are NOT `p1` or `p2`:
        >>> fm.marginalize_over('p1', 'p2', invert=True)
        FisherMatrix(array([[ 5.04480062, -0.04490453],
               [-0.04490453,  1.61599083]]), names=array(['p1', 'p2'], dtype=object), latex_names=array(['p1', 'p2'], dtype=object), fiducial=array([0., 0.]))
        """
        inv = self.inverse()
        if invert is True:
            names = set(names) ^ set(self.names)
        fm = inv.drop(*names, ignore_errors=ignore_errors)
        return fm.inverse()


    @classmethod
    def from_file(
        cls,
        path : str,
    ):
        """
        Reads a Fisher object from a file.

        Parameters
        ----------
        path : str
            the path to the file

        Returns
        -------
        Instance of `FisherMatrix`
        """
        with open(path, 'r', encoding='utf-8') as file_handle:
            data = json.loads(file_handle.read())

        return cls(
            data['values'],
            names=data['names'],
            latex_names=data['latex_names'],
            fiducial=data['fiducial'],
        )
