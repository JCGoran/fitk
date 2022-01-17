"""
Package for performing operations on Fisher objects.
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
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patheffects import Stroke, Normal
from matplotlib import colors

# first party imports
from fisher_utils import \
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



class FisherMatrix:
    """
    Class for handling Fisher objects.
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
            self._fiducial = np.zeros(self._size)
        else:
            self._fiducial = np.array(fiducial)

        # setting the names
        if names is None:
            self._names = make_default_names(self._size)
        else:
            # check they're unique
            if len(set(names)) != len(names):
                raise MismatchingSizeError(set(names), names)

            self._names = np.array(names)

        # setting the pretty names (LaTeX)
        if names_latex is None:
            self._names_latex = copy.deepcopy(self._names)
        else:
            self._names_latex = np.array(names_latex)

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
        Checks whether the values make a valid Fisher matrix.
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
    ):
        """
        Sorts the Fisher object by name according to some criterion.

        Parameters
        ----------
        **kwargs
            all of the other keyword arguments for the Python builtin `sorted`
        """
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

        self._values = value


    def is_diagonal(self):
        """
        Checks whether the Fisher matrix is diagonal.
        """
        return np.all(self.values == np.diag(np.diagonal(self.values)))


    def diagonal(self, **kwargs):
        """
        Returns the diagonal elements of the Fisher object as a numpy array.
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

        Examples
        --------
        > m = FisherMatrix(np.diag(1, 2, 3))
        > assert m.drop('p1', 'p3') == FisherMatrix(np.diag(2), names=['p2'])
        True
        """
        if not ignore_errors and not set(names).issubset(set(self.names)):
            raise ValueError(
                f'The names ({list(names)}) are not a strict subset ' \
                f'of the parameter names in the Fisher object ({self.names}); ' \
                'you can pass `ignore_errors=True` to ignore this error'
            )
        elif ignore_errors:
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


    def inverse(self):
        """
        Returns the inverse of the Fisher matrix.
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
        """
        return np.linalg.det(self.values)


    def constraints(
        self,
        name : Optional[AnyStr] = None,
        marginalized : bool = True,
        sigma : Optional[SupportsFloat] = None,
        p : Optional[SupportsFloat] = None,
    ):
        """
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
            the confidence interval (p-value)

        Notes
        -----
        The user should specify either `sigma` or `p`, but not both
        simultaneously.
        If unspecified, defaults to `sigma=1`.
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
            sigma = norm.ppf(p)
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
        self._names = value


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
        self._names_latex = value


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


    def __floordiv__(
        self,
        other : FisherMatrix,
    ) -> FisherMatrix:
        """
        Returns the result of dividing two Fisher objects.
        """
        if type(other) != type(self):
            raise TypeError(
                f'Incompatible types for division: {type(other)} and {type(self)}'
            )
        # TODO implement this
        return NotImplemented


    def __truediv__(
        self,
        other : Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of dividing a Fisher object by a number, or another Fisher object.
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
        Returns the result of multiplying a Fisher object by a number, or another Fisher object.
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
        Returns the result of multiplying a number with a Fisher object.
        """
        return self.__mul__(other)


    def reparametrize(
        self,
        jacobian : Iterable[Any],
        names : Optional[Iterable[AnyStr]] = None,
        names_latex : Optional[Iterable[AnyStr]] = None,
        fiducial : Optional[Iterable[float]] = None,
    ):
        """
        Returns a new Fisher object with parameters `names`, which are
        related to the old ones via the transformation `jacobian`.
        Currently limited to rank 26 tensors (we run out of single letters in
        the English alphabet otherwise).
        Does not differentiate between covariant/contravariant indices.

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
            if len(set(names)) != self.names:
                raise MismatchingSizeError(names, self.names)
            if names_latex is not None:
                if len(set(names_latex)) != self.names_latex:
                    raise MismatchingSizeError(names_latex, self.names_latex)
            else:
                names_latex = names
        else:
            # we don't transform the names
            names = self.names
            names_latex = self.names_latex

        if fiducial is not None:
            if len(fiducial) != len(self.fiducial):
                raise MismatchingSizeError(fiducial, self.fiducial)
        else:
            fiducial = self.fiducial

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
        overwrite : bool = False,
    ):
        """
        Saves the Fisher object to a file.

        Parameters
        ----------
        path : AnyStr
            the path to save the data to.

        args : AnyStr
            whatever other metadata about the object needs to be saved.
            Needs to be a name of one of the methods of the class.

        overwrite : bool, default = False
            whether to overwrite the file if it exists
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

        for arg in args:
            data.update(
                {arg : jsonify(getattr(self, arg)())},
            )

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f'The file {path} already exists, please pass ' \
                '`overwrite=True` if you wish to explicitly overwrite '
                'the file'
            )

        with open(path, 'w') as f:
            f.write(json.dumps(data, indent=4))



def from_file(
    path : AnyStr,
):
    """
    Reads a Fisher object from a file.

    Parameters
    ----------
    path : AnyStr
        the path to the file
    """
    with open(path, 'r') as f:
        data = json.loads(f.read())

    return FisherMatrix(
        data['values'],
        names=data['names'],
        names_latex=data['names_latex'],
        fiducial=data['fiducial'],
    )
