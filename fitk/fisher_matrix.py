"""
Package for performing operations on Fisher objects.
See here for documentation of `FisherMatrix`, `FisherParameter`, and `from_file`.
"""

# needed for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
import copy
from itertools import \
    permutations, \
    product
import json
import os
from typing import \
    AnyStr, \
    NamedTuple, \
    Sequence, \
    Mapping, \
    Iterable, \
    Optional, \
    Callable, \
    Union, \
    List, \
    Any, \
    Dict, \
    Set, \
    SupportsFloat, \
    Tuple

# third party imports
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from matplotlib.patheffects import Stroke, Normal
from matplotlib import colors

# first party imports
from .fisher_utils import \
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
    jsonify



class FisherParameter:
    """
    Simple container for specifying the name, the LaTeX name, and the fiducial
    of a parameter.
    """
    def __init__(
        self,
        name : AnyStr,
        name_latex : Optional[AnyStr] = None,
        fiducial : float = 0,
    ):
        """
        Constructor.

        Parameters
        ----------
        name : AnyStr
            the name of the parameter

        name_latex : Optional[AnyStr], default = None
            the LaTeX name of the parameter. If not specified, defaults to `name`.

        fiducial : float, default = 0
            the fiducial value of the parameter
        """
        self.name = name
        self.name_latex = name_latex if name_latex is not None else name
        self.fiducial = fiducial


    def __repr__(self):
        """
        Representation of the FisherParameter
        """
        return \
            f"FisherParameter(name='{self.name}', " \
            f"name_latex='{self.name_latex}', " \
            f"fiducial={self.fiducial})"


    def __str__(self):
        return self.__repr__()



class FisherMatrix:
    r"""
    Class for handling Fisher objects.

    Examples
    --------
    >>> fm = FisherMatrix(np.diag([5, 4])) # no parameter names specified
    >>> fm # has a friendly representation in an interactive session
    FisherMatrix([[5 0]
     [0 4]], names=['p1' 'p2'], names_latex = ['p1' 'p2'], fiducial=[0. 0.])
    >>> fm.names # getting the names
    array(['p1', 'p2'], dtype=object)
    >>> fm.values # getting the underlying numpy array of values
    array([[5, 0],
           [0, 4]])
    >>> fm.fiducial # getting the values of the fiducials
    array([0., 0.])
    >>> fm.names = ['x', 'y'] # names can be changed (ditto for fiducial and values; dimension must of course match the original)
    >>> fm.names_latex = [r'$\mathbf{X}$', r'$\mathbf{Y}$'] # LaTeX names as well
    >>> fm_names = FisherMatrix(np.diag([1, 2]), names=['x', 'y']) # with parameter names
    >>> fm + fm_names # we can perform arithmetic operations on objects which have the same names (not necessarily in order) and fiducials; the LaTeX names do not need to match, but the resulting LaTeX names are inherited from the left-most object
    FisherMatrix([[6 0]
     [0 6]], names=['x' 'y'], names_latex = ['p1' 'p2'], fiducial=[0. 0.])
    >>> fm * fm_names # we can also do element-wise multiplication (or division with `/`)
    FisherMatrix([[5 0]
     [0 8]], names=['x' 'y'], names_latex = ['p1' 'p2'], fiducial=[0. 0.])
    >>> fm @ fm_names # likewise, we can do matrix multiplication
    FisherMatrix([[5 0]
     [0 8]], names=['x' 'y'], names_latex = ['p1' 'p2'], fiducial=[0. 0.])
    >>> fm.trace() # other linear algebra methods include: `eigenvalues`, `eigenvectors`, `determinant`
    9
    >>> fm.inverse() # can also take the matrix inverse
    FisherMatrix([[0.2  0.  ]
     [0.   0.25]], names=['x' 'y'], names_latex = ['p1' 'p2'], fiducial=[0. 0.])
    >>> fm.drop('x') # we can drop parameters from the Fisher matrix
    FisherMatrix([[4]], names=['y'], names_latex = ['p2'], fiducial=[0.])
    >>> fm.to_file('example_matrix.json') # we can save it to a file
    >>> fm_new = from_file('example_matrix.json') # we can of course load it as well
    """

    def __init__(
        self,
        values : np.array,
        names : Optional[Iterable[AnyStr]] = None,
        names_latex : Optional[AnyStr] = None,
        fiducial : Optional[Iterable[float]] = None,
    ):
        """
        Constructor for Fisher object.

        Parameters
        ----------
        values : array-like
            The values of the Fisher object.

        names : array-like iterable of `str`, default = None
            The names of the parameters.
            If not specified, defaults to `p1, ..., pn`.

        names_latex : array-like iterable of `str`, default = None
            The LaTeX names of the parameters.
            If not specified, defaults to `names`.

        fiducial : array-like iterable of `float`, default = None
            The fiducial values of the parameters. If not specified, default to
            0 for all parameters.

        Examples
        --------
        >>> FisherMatrix(np.diag([1, 2, 3])) # no names specified
        FisherMatrix([[1 0 0]
         [0 2 0]
         [0 0 3]], names=['p1' 'p2' 'p3'], names_latex = ['p1' 'p2' 'p3'], fiducial=[0. 0. 0.])
        >>> FisherMatrix(np.diag([1, 2]), names=['alpha', 'beta'], names_latex=[r'$\alpha$', r'$\beta$'], fiducial=[-3, 2]) # with names, LaTeX names, and fiducial
        FisherMatrix([[1 0]
         [0 2]], names=['alpha' 'beta'], names_latex = ['$\\alpha$' '$\\beta$'], fiducial=[-3.  2.])
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
        self._values = np.array(values)

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
        if names_latex is None:
            self._names_latex = copy.deepcopy(self._names)
        else:
            self._names_latex = np.array(names_latex, dtype=object)

        # check sizes of inputs
        if not all(
            len(_) == self._size \
            for _ in (self._names, self._fiducial, self._names_latex)
        ):
            raise MismatchingSizeError(
                self._values[0],
                self._names,
                self._fiducial,
                self._names_latex,
            )


    def rename(
        self,
        names : Mapping[AnyStr, Union[AnyStr, FisherParameter]],
        ignore_errors : bool = False,
    ) -> FisherMatrix:
        """
        Returns a Fisher object with new names.

        Parameters
        ----------
        names : Mapping[AnyStr, Union[AnyStr, FisherParameter]]
            a mapping (dictionary-like object) between the old names and the
            new ones. The values it maps to can either be a string (the new name), or an
            instance of `FisherParameter`, which takes a name, a LaTeX name, and a fiducial as
            its arguments.

        ignore_errors : bool, default = False
            if set to True, will not raise an error if a parameter doesn't exist

        Returns
        -------
        Instance of `FisherMatrix`.

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))
        >>> m.rename({'p1' : 'a', 'p2' : FisherParameter('b', name_latex='$b$', fiducial=2)})
        FisherMatrix([[1 0 0]
         [0 2 0]
         [0 0 3]], names=['a' 'b' 'p3'], names_latex=['a' '$b$' 'p3'], fiducial=[0. 2. 0.])
        """
        # check uniqueness and size
        if len(set(names)) != len(names):
            raise MismatchingSizeError(set(names), names)

        if not ignore_errors:
            for name in names:
                if name not in self.names:
                    raise ParameterNotFoundError(name, self.names)

        names_new = copy.deepcopy(self.names)
        names_latex_new = copy.deepcopy(self.names_latex)
        fiducial_new = copy.deepcopy(self.fiducial)

        for name, value in names.items():
            index = np.where(names_new == name)
            # it's a mapping to a FisherParameter
            if isinstance(value, FisherParameter):
                names_latex_new[index] = value.name_latex
                fiducial_new[index] = value.fiducial
                names_new[index] = value.name
            else:
                names_new[index] = value
                names_latex_new[index] = value

        return FisherMatrix(
            self.values,
            names=names_new,
            names_latex=names_latex_new,
            fiducial=fiducial_new,
        )


    def _repr_html_(self):
        """
        Representation of the Fisher object suitable for viewing in Jupyter
        notebook environments.
        """
        header_matrix = '<thead><tr><th></th>' + (
            '<th>{}</th>' * len(self)
        ).format(*self.names_latex) + '</tr></thead>'

        body_matrix = '<tbody>'
        for index, name in enumerate(self.names_latex):
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
        return f'FisherMatrix({self.values}, names={self.names}, names_latex = {self.names_latex}, fiducial={self.fiducial})'


    def __str__(self):
        """
        String representation of the Fisher object.
        """
        return self.__repr__()


    def __getitem__(
        self,
        keys : Union[Tuple[AnyStr], slice],
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
            names_latex = self.names_latex[sl]
            fiducial = self.fiducial[sl]

            return FisherMatrix(
                values,
                names=names,
                names_latex=names_latex,
                fiducial=fiducial,
            )

        try:
            _ = iter(keys)
        except TypeError as err:
            raise TypeError from err

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

        return self._values[indices]


    def __setitem__(
        self,
        keys : Tuple[AnyStr],
        value : SupportsFloat,
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
        rc : dict = {},
        **kwargs,
    ):
        """
        Returns the image of the Fisher object.
        """
        # TODO should only work with 2D data.
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(self.size, self.size))
            img = ax.imshow(
                self.values,
                interpolation='none',
                norm=colors.CenteredNorm(),
                **kwargs,
            )

            ax.set_xticks(np.arange(self.size))
            ax.set_yticks(np.arange(self.size))
            ax.set_xticklabels(self.names_latex)
            ax.set_yticklabels(self.names_latex)

            if colorbar:
                allowed_orientations = ('vertical', 'horizontal')
                if colorbar_orientation not in allowed_orientations:
                    raise ValueError(
                        f'\'{colorbar_orientation}\' is not one of: {allowed_orientations}'
                    )

                if colorbar_orientation == 'vertical':
                    cax = fig.add_axes(
                        [ax.get_position().x1 + colorbar_space,
                        ax.get_position().y0,
                        colorbar_width,
                        ax.get_position().height]
                    )
                    fig.colorbar(img, cax=cax)
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
                    fig.colorbar(img, cax=cax, orientation='horizontal')
                    cax.xaxis.set_ticks_position('top')
                    cax.set_xticklabels(
                        [f'${float_to_latex(_)}$' for _ in cax.get_xticks()]
                    )
                fig.colorbar(img)

            # whether or not we want to display the actual values inside the
            # matrix
            if show_values:
                mid_coords = np.arange(self.size)
                for index1, index2 in product(range(self.size), range(self.size)):
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
            either 'fiducial' or 'names_latex', it will sort according to those.
            In the second special case that the value of the keyword `key` is
            set to an array of integers of equal size as the Fisher object, sorts them
            according to those instead.

        Returns
        -------
        Instance of `FisherMatrix`.

        Examples
        --------
        >>> m = FisherMatrix(np.diag([3, 1, 2]), names=list('sdf'), names_latex=['hjkl', 'qwe', 'll'], fiducial=[8, 7, 3])
        >>> m.sort(key='fiducial')
        FisherMatrix([[2 0 0]
         [0 1 0]
         [0 0 3]], names=['f' 'd' 's'], names_latex = ['ll' 'qwe' 'hjkl'], fiducial=[3. 7. 8.])
        >>> m.sort(key='names_latex')
        FisherMatrix([[1 0 0]
         [0 2 0]
         [0 0 3]], names=['d' 'f' 's'], names_latex = ['qwe' 'll' 'hjkl'], fiducial=[7. 3. 8.])
        """
        allowed_keys = ('fiducial', 'names_latex')
        # an integer index
        if 'key' in kwargs and all(hasattr(_, '__index__') for _ in kwargs['key']):
            index = np.array(kwargs['key'], dtype=int)
            names = self.names[index]
        # either 'fiducial' or 'names_latex'
        elif 'key' in kwargs and kwargs['key'] in allowed_keys:
            index = np.argsort(getattr(self, kwargs['key']))
            if 'reversed' in kwargs and kwargs['reversed'] is True:
                index = np.flip(index)
            names = self.names[index]
        # something that can be passed to `sorted`
        else:
            names = sorted(self.names, **kwargs)
            index = get_index_of_other_array(self.names, names)

        names_latex = self.names_latex[index]
        fiducial = self.fiducial[index]
        values = reindex_array(self.values, index)

        return FisherMatrix(
            values,
            names=names,
            names_latex=names_latex,
            fiducial=fiducial,
        )


    def __eq__(self, other):
        """
        The equality operator.
        Returns True if the operands have the following properties:
            - are instances of FisherMatrix
            - have same names (potentially shuffled)
            - have same dimensionality
            - have same fiducials (potentially shuffled)
            - have same values (potentially shuffled)
        """
        if set(self.names) != set(other.names):
            return False

        # index for re-shuffling parameters
        index = get_index_of_other_array(self.names, other.names)

        return isinstance(other, FisherMatrix) \
        and self.ndim == other.ndim \
        and len(self) == len(other) \
        and set(self.names) == set(other.names) \
        and np.allclose(
            self.fiducial[index],
            other.fiducial
        ) \
        and np.allclose(
            reindex_array(self.values, index),
            other.values
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
        if self.ndim != np.dim(value):
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


    @property
    def size(self):
        """
        Returns the number of parameters in the Fisher object (same as
        `len(object)`).
        """
        return self._size


    def drop(
        self,
        *names : AnyStr,
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

        ignore_errors : bool, default = False
            should non-existing parameters be ignored

        Returns
        -------
        Instance of `FisherMatrix`

        Examples
        --------
        >>> m = FisherMatrix(np.diag([1, 2, 3]))
        >>> m.drop('p1', 'p3')
        FisherMatrix([[2]], names=['p2'], names_latex = ['p2'], fiducial=[0.])
        """
        if not ignore_errors and not set(names).issubset(set(self.names)):
            raise ValueError(
                f'The names ({list(names)}) are not a strict subset ' \
                f'of the parameter names in the Fisher object ({self.names}); ' \
                'you can pass `ignore_errors=True` to ignore this error'
            )

        if ignore_errors:
            names = np.array([name for name in names if name in self.names])

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
        names_latex = np.delete(self.names_latex, index)
        names = np.delete(self.names, index)

        return FisherMatrix(
            values,
            names=names,
            names_latex=names_latex,
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


    def inverse(self) -> FisherMatrix:
        """
        Returns the inverse of the Fisher matrix.

        Returns
        -------
        Instance of `FisherMatrix`
        """
        # inverse satisfies properties of Fisher matrix, see:
        # https://math.stackexchange.com/a/26200
        return FisherMatrix(
            np.linalg.inv(self.values),
            names=self.names,
            names_latex=self.names_latex,
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
        name : Optional[AnyStr] = None,
        marginalized : bool = True,
        sigma : Optional[SupportsFloat] = None,
        p : Optional[SupportsFloat] = None,
    ):
        r"""
        Returns the constraints on a parameter as a float, or on all of them
        as a numpy array if `name` is not specified.

        Parameters
        ----------
        name : Optional[AnyStr] = None
            the name of the parameter for which we we want the constraints

        marginalized : bool, default = True
            whether we want the marginalized or the unmarginalized
            constraints.

        sigma : Optional[SupportsFloat], default = None
            how many sigmas away.

        p : Optional[SupportsFloat], default = None
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

        Notes
        -----
        The user should specify either `sigma` or `p`, but not both
        simultaneously.
        If neither are specified, defaults to `sigma=1`.

        Examples
        --------
        >>> m = FisherMatrix([[3, -2], [-2, 5]])
        >>> m.constraints() # constraints for all parameters (marginalized)
        array([0.67419986, 0.52223297])
        >>> m.constraints(marginalized=False) # constraints for all parameters (unmarginalized)
        array([0.57735027, 0.4472136 ])
        >>> m.constraints('p1', marginalized=False) # one parameter only
        array([0.57735027])
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
    def names_latex(self):
        """
        Returns the LaTeX names of the parameters of the Fisher object.
        """
        return self._names_latex


    @names_latex.setter
    def names_latex(
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
        self._names_latex = np.array(value, dtype=object)


    def __add__(
        self,
        other : FisherMatrix,
    ) -> FisherMatrix:
        """
        Returns the result of adding two Fisher objects.
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

        values = self.values + reindex_array(other.values, index)

        return FisherMatrix(
            values,
            names=self.names,
            names_latex=self.names_latex,
            fiducial=self.fiducial,
        )


    def __sub__(
        self,
        other : FisherMatrix,
    ) -> FisherMatrix:
        """
        Returns the result of subtracting two Fisher objects.
        """
        temp = copy.deepcopy(other)
        temp.values = -temp.values
        return self.__add__(temp)


    def __pow__(
        self,
        other : SupportsFloat,
    ):
        """
        Raises the Fisher object to some power.
        """
        return FisherMatrix(
            np.power(self.values, other),
            names=self.names,
            names_latex=self.names_latex,
            fiducial=self.fiducial,
        )


    def __matmul__(
        self,
        other : FisherMatrix,
    ):
        """
        Multiplies two Fisher objects.
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

        return FisherMatrix(
            values,
            names=self.names,
            names_latex=self.names_latex,
            fiducial=self.fiducial,
        )


    def __truediv__(
        self,
        other : Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of dividing a Fisher object by a number, or another
        Fisher object.
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
                raise TypeError from err
        else:
            values = self.values / other

        return FisherMatrix(
            values,
            names=self.names,
            names_latex=self.names_latex,
            fiducial=self.fiducial,
        )


    def __mul__(
        self,
        other : Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of multiplying a Fisher object by a number, or
        another Fisher object.
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

        return FisherMatrix(
            values,
            names=self.names,
            names_latex=self.names_latex,
            fiducial=self.fiducial,
        )


    def __rmul__(
        self,
        other : Union[float, int],
    ) -> FisherMatrix:
        """
        Returns the result of multiplying a number by a Fisher object, or
        another Fisher object.
        """
        return self.__mul__(other)


    def reparametrize(
        self,
        jacobian : Iterable[Any],
        names : Optional[Iterable[AnyStr]] = None,
        names_latex : Optional[Iterable[AnyStr]] = None,
        fiducial : Optional[Iterable[float]] = None,
    ) -> FisherMatrix:
        """
        Returns a new Fisher object with parameters `names`, which are
        related to the old ones via the transformation `jacobian`.
        Currently limited to rank 26 tensors (we run out of single letters in
        the English alphabet otherwise).
        Does not differentiate between covariant/contravariant indices.
        See the [Wikipedia article](https://en.wikipedia.org/w/index.php?title=Fisher_information&oldid=1063384000#Reparametrization) for more information.

        Parameters
        ----------
        transformation : array-like
            the Jacobian of the transformation

        names : array-like, default = None
            list of new names for the Fisher object. If None, uses the old
            names.

        names_latex: array-like, default = None
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
        FisherMatrix([[33 19]
         [19 17]], names=['a' 'b'], names_latex = ['a' 'b'], fiducial=[0. 0.])
        """
        if self.ndim > 26:
            raise ValueError(
                'The dimensionality of the Fisher object is > 26, which is not supported'
            )

        char_first = 'A'
        char_second = 'a'

        # makes the string 'aA,bB,cC,dD,...'
        index_transformation = ','.join(
            [
                f'{chr(ord(char_first) + i)}{chr(ord(char_second) + i)}' \
                for i in range(self.ndim)
            ]
        )
        # the tensor to be transformed has just dummy indices 'ABCD...'
        index_dummy = ''.join([chr(ord(char_second) + i) for i in range(self.ndim)])
        # the output tensor has indices 'abcd...'
        index_result = ''.join([chr(ord(char_first) + i) for i in range(self.ndim)])

        values = np.einsum(
            f'{index_transformation},{index_dummy}->{index_result}',
            *([jacobian] * self.ndim), self.values
        )

        if names is not None:
            if len(set(names)) != len(self.names):
                raise MismatchingSizeError(names, self.names)
            if names_latex is not None:
                if len(set(names_latex)) != len(self.names_latex):
                    raise MismatchingSizeError(names_latex, self.names_latex)
            else:
                names_latex = copy.deepcopy(names)
        else:
            # we don't transform the names
            names = copy.deepcopy(self.names)
            names_latex = copy.deepcopy(self.names_latex)

        if fiducial is not None:
            if len(fiducial) != len(self.fiducial):
                raise MismatchingSizeError(fiducial, self.fiducial)
        else:
            fiducial = copy.deepcopy(self.fiducial)

        return FisherMatrix(
            values,
            names=names,
            names_latex=names_latex,
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
                names=self.names_latex,
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
                names=self.names_latex,
                fmt_values=fmt_values,
            )
        )


    def to_file(
        self,
        path : AnyStr,
        *args : AnyStr,
        metadata : dict = {},
        overwrite : bool = False,
    ):
        r"""
        Saves the Fisher object to a file.
        The format is a simple JSON file, containing at least the values of the
        Fisher object, the names of the parameters, the LaTeX names, and the
        fiducial values.

        Parameters
        ----------
        path : AnyStr
            the path to save the data to.

        args : AnyStr
            whatever other information about the object needs to be saved.
            Needs to be a name of one of the methods of the class.

        metadata : dict, default = {}
            any metadata that should be associated to the object saved

        overwrite : bool, default = False
            whether to overwrite the file if it exists

        Returns
        -------
        None

        Examples
        --------
        >>> fm = FisherMatrix(np.diag([1, 2]), names=['a', 'b'], names_latex=[r'$\mathbf{A}$', r'$\mathbf{B}$'])
        >>> fm.to_file('example_matrix.json') # assuming it doesn't exist
        >>> with open('example_matrix.json', 'r') as f:
        ...     f.read() # see the raw contents
        '{\n    "values": [\n        [\n            1,\n            0\n        ],\n        [\n            0,\n            2\n        ]\n    ],\n    "names": [\n        "a",\n        "b"\n    ],\n    "names_latex": [\n        "$\\\\mathbf{A}$",\n        "$\\\\mathbf{B}$"\n    ],\n    "fiducial": [\n        0.0,\n        0.0\n    ]\n}'
        >>> fm_read = from_file('example_matrix.json') # convenience function for reading it
        >>> fm == fm_read # verify it's the same object
        True
        """
        data = {
            'values' : self.values.tolist(),
            'names' : self.names.tolist(),
            'names_latex' : self.names_latex.tolist(),
            'fiducial' : self.fiducial.tolist(),
        }

        allowed_metadata = {
            'is_valid' : bool,
            'eigenvalues' : np.array,
            'eigenvectors' : np.array,
            'trace' : float,
            'determinant' : float,
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

        with open(path, 'w') as f:
            f.write(json.dumps(data, indent=4))


    def marginalize_over(
        self,
        *names : AnyStr,
        invert : bool = False,
        ignore_errors : bool = False,
    ) -> FisherMatrix:
        """
        Perform marginalization over some parameters.

        Parameters
        ----------
        names : AnyStr
            the names of the parameters to marginalize over

        invert : bool, default = False
            whether to marginalize over all the parameters NOT in names

        ignore_errors : bool, default = False
            should non-existing parameters be ignored

        Returns
        -------
        Instance of `FisherMatrix`.
        """
        inv = self.inverse()
        if invert is True:
            names = set(names) ^ set(self.names)
        fm = inv.drop(*names, ignore_errors=ignore_errors)
        return fm.inverse()



def from_file(
    path : AnyStr,
):
    """
    Reads a Fisher object from a file.

    Parameters
    ----------
    path : AnyStr
        the path to the file

    Returns
    -------
    Instance of `FisherMatrix`
    """
    with open(path, 'r') as f:
        data = json.loads(f.read())

    return FisherMatrix(
        data['values'],
        names=data['names'],
        names_latex=data['names_latex'],
        fiducial=data['fiducial'],
    )
