"""
Library for performing operations on Fisher-like objects (vectors, matrices, tensors).
"""

from __future__ import annotations

from itertools import permutations
import copy
from typing import NamedTuple, Sequence, Mapping, Iterable, Optional, Callable, \
    Union, List, Any, Dict, Set, Tuple

import numpy as np

from utils import CustomSet

# TODO implement custom errors
# TODO implement plotting of contours, Fisher vectors and Fisher matrices
# TODO implement plotting of bananas (a la DALI)
# TODO implement an index method which can access elements by index
# TODO implement saving of objects to disk. Decide how this should be done (in same file, in another file, etc.)
# TODO implement arbitrary metadata for the FisherTensor

class _HTML_Wrapper:
    """
    A wrapper class for pretty printing objects in a Jupyter notebook.
    """
    def __init__(self, string):
        self._string = string
    def _repr_html_(self):
        return self._string


def _make_html_table(
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


def _default_names(
    size : int,
    character : str = 'p',
):
    """
    Returns the array with default names `p1, ..., pn`
    """
    if size < 0:
        raise ValueError
    return [f'{character}{_ + 1}' for _ in range(size)]


def _is_square(data):
    """
    Checks whether a numpy array-able object is square.
    """

    shape = np.shape(data)
    return all(shape[0] == _ for _ in shape)


def _is_symmetric(data):
    """
    Checks whether a numpy array-able object is symmetric.
    """
    return np.allclose(np.transpose(data), np.array(data))


def _is_positive_semidefinite(data):
    """
    Checks whether a numpy array-able object is positive semi-definite.
    """
    return np.all(np.linalg.eigvalsh(data) >= 0)


def _has_positive_diagonal(data):
    """
    Checks whether all of the values on the diagonal are positive.
    """
    return np.all(np.diag(data) >= 0)


# TODO make custom errors so we don't just raise a ValueError all the time
def _check_fisher_tensor(
    data,
    ndim : int,
):
    """
    Checks if an object is a valid Fisher tensor.

    Parameters
    ----------
    data : array-like
        the data we wish to validate

    ndim : int
        the expected number of dimensions of the data
    """
    if (ndim == 1 and np.ndim(data) != 1) \
    or (ndim == 2 and np.ndim(data) != 2):
        raise ValueError(
            f'Expected {ndim} dimension(s), got {np.ndim(data)}.'
        )

    # check that it's square
    if not _is_square(data):
        raise ValueError(
            f'Expected a square tensor, got {np.shape(data)}'
        )

    # -||- with positive diagonal elements in even dimensions
    if np.ndim(data) % 2 == 0 and not _has_positive_diagonal(data):
        raise ValueError(
            f'The array {data} has negative elements on the diagonal'
        )

    if ndim == 2:
        # -||- that is also symmetric
        if not _is_symmetric(data):
            raise ValueError(
                f'The matrix {data} must be symmetric'
            )
        # check that it also has non-negative eigenvalues
        # see https://stats.stackexchange.com/a/49961
        if not _is_positive_semidefinite(data):
            raise ValueError(
                f'The matrix {data} must be positive semi-definite'
            )


class FisherTensor:
    """
    Class for handling arbitrary tensor data with names.
    """

    # global safety flag
    _safe_global = True

    def __init__(
        self,
        data,
        names = None,
        fiducial = None,
        ndim : int = None,
        character : str = 'p',
        safe : bool = True,
        **kwargs,
    ):
        """
        Constructor for Fisher tensor.

        Parameters
        ----------
        data : array-like or string (path to file) or dict or FisherTensor
            The data of the Fisher tensor.
            If it's a string, accepts the same keyword arguments as numpy.loadtxt.
            If it's a dict, it creates a diagonal tensor, and the `names`
            argument is ignored.
            If it's array-like, it attempts to convert it to a numpy array.

        names : array-like, default = None
            The names of the parameters.
            If not specified (a None-like object), default to `p1, ..., pn`.
            Must be a hashable type that's not an int

        fiducial : array-like, default = None
            The fiducial values of the parameters. If not specified (a
            None-like object), default to 0 for all parameters.

        ndim : int, default = None
            The expected dimensionality of the Fisher tensor.

        character : str, default = 'p'
            The name of the default parameters.

        safe : bool, default = True
            Whether an error should be raised if the data format is invalid.
        """

        # short circuiting if we're just typecasting
        if isinstance(data, FisherTensor):
            other = FisherTensor(
                data.data,
                names=data.names,
                fiducial=data.fiducial,
                safe=data.safe,
                ndim=data.ndim,
            )
            self._data = copy.deepcopy(other.data)
            self._size = copy.deepcopy(other.size)
            self._names = copy.deepcopy(other.names)
            self._fiducial = copy.deepcopy(other.fiducial)
            self._safe = copy.deepcopy(other.safe)
            self._ndim = copy.deepcopy(other.ndim)
            return

        self._safe = safe

        _isdict = False

        if isinstance(data, str):
            self._data = np.loadtxt(data, **kwargs)
        # NOTE if we pass a dict, the Fisher object will always be diagonal,
        # and the `names` parameter passed is not necessary
        elif isinstance(data, dict):
            if ndim == 2:
                self._data = np.diag(list(data.values()))
            else:
                self._data = np.array(list(data.values()))
            _isdict = True
        else:
            # try to treat it as an array-like object
            self._data = np.array(data)

        # special case: we pass a flat array to a FisherMatrix constructor
        if np.ndim(self._data) == 1 and ndim == 2:
            self._data = np.diag(np.array(self._data))

        # check that the data is sane
        if ndim and self._safe and FisherTensor._safe_global:
            _check_fisher_tensor(self._data, ndim)

        # we can access the element if the above check succeeded
        self._size = np.shape(self._data)[0]

        # setting the names
        if _isdict:
            _names = list(data.keys())
            # validation that they're not integers or floats
            for name in _names:
                if hasattr(name, '__abs__'):
                    raise ValueError
            self._names = CustomSet(_names)
        else:
            if names is None:
                self._names = CustomSet(_default_names(self._size, character))
            else:
                try:
                    _ = iter(names)
                except TypeError as err:
                    raise TypeError from err
                # special case: we can't pass a string-like object
                if hasattr(names, 'split'):
                    raise TypeError(f'Invalid type {type(names)} for argument `names`.')
                if len(names) != self._size:
                    raise ValueError
                # this will catch anything that isn't hashable
                if len(set(names)) != len(names):
                    raise ValueError('The parameter names must be unique.')
                for name in names:
                    if hasattr(name, '__abs__'):
                        raise ValueError
                self._names = CustomSet(names)

        # setting the fiducial
        if fiducial is None:
            self._fiducial = np.zeros(self._size)
        else:
            try:
                _ = iter(fiducial)
            except TypeError as err:
                raise TypeError('The fiducial must be iterable') from err
            if len(fiducial) != self._size:
                raise ValueError('Fiducial length different from size of input array')
            # fiducials must be castable to numbers
            try:
                self._fiducial = np.array([float(_) for _ in fiducial])
            except TypeError as err:
                raise TypeError('Unable to convert fiducials to float type') from err

    def __getitem__(
        self,
        keys,
    ):
        """
        Implements access to elements in the Fisher tensor.
        Has support for slicing.
        """
        # the object can be sliced
        if isinstance(keys, slice):
            start, stop, step = keys.indices(len(self))
            indices = (slice(start, stop, step),) * self.ndim
            data_new = self.data[indices]
            names_new = self.names[start:stop:step]
            fiducial_new = self.fiducial[start:stop:step]
            return FisherTensor(
                data_new,
                names=names_new,
                fiducial=fiducial_new,
                ndim=self.ndim,
                safe=self.safe,
            )

        try:
            _ = iter(keys)
        except TypeError as err:
            raise TypeError from err

        # the keys can be a tuple
        if isinstance(keys, tuple):
            if len(keys) != self.ndim:
                raise ValueError(f'Got length {len(keys)}')

            for key in keys:
                if key not in self._names:
                    raise ValueError(
                        f'Parameter {key} not found'
                    )

            indices = tuple(self._names.index(key) for key in keys)

        # otherwise, it's some generic object
        else:
            if keys not in self._names:
                raise ValueError(f'Parameter {keys} not found')
            indices = (self._names.index(keys),)
        return self._data[indices]

    def __setitem__(
        self,
        keys,
        value,
    ):
        """
        Implements setting of elements in the Fisher tensor.
        Does not support slicing.
        """
        try:
            _ = iter(keys)
        except TypeError as err:
            raise err

        if len(keys) != self.ndim:
            raise ValueError(f'Got length {len(keys)}')

        # automatically raises a value error
        indices = tuple(self._names.index(key) for key in keys)

        temp_data = copy.deepcopy(self._data)

        if not all(index == indices[0] for index in indices):
            for permutation in list(permutations(indices)):
                # update all symmetric parts
                temp_data[permutation] = value
        else:
            temp_data[indices] = value

        if self.safe and FisherTensor.safe_global():
            _check_fisher_tensor(temp_data, np.ndim(temp_data))

        self._data = copy.deepcopy(temp_data)

    @property
    def safe(self):
        """
        Returns whether the safety flag of the object is turned on or off.
        """
        return self._safe

    def is_valid(self):
        """
        Checks whether the data in the object is a valid Fisher tensor.
        """
        try:
            _check_fisher_tensor(self.data, self.ndim)
            return True
        except ValueError:
            return False

    def sort(
        self,
        **kwargs
    ):
        """
        Sorts the Fisher tensor by name according to some criterion.
        Note that each element of names should be sortable, i.e. should have
        the comparison operators (==, <, <=, >, =>, !=) implemented.

        Parameters
        ----------
        **kwargs
            all of the other keyword arguments for the builtin `sorted`
        """
        data = self.data
        names = sorted(self.names, **kwargs)
        index = np.array([names.index(name) for name in self.names])
        fiducial = self.fiducial[index]

        for dim in range(self.ndim):
            data = np.swapaxes(
                np.swapaxes(data, 0, dim)[index],
                dim, 0
            )

        return FisherTensor(
            data,
            names=names,
            fiducial=fiducial,
            safe=self.safe,
            ndim=self.ndim,
        )

    @staticmethod
    def safe_global():
        """
        Returns the state of the global safety flag.
        """
        return FisherTensor._safe_global

    @staticmethod
    def set_unsafe_global():
        """
        Sets the `safe_global` parameter to False.
        """
        FisherTensor._safe_global = False

    @staticmethod
    def set_safe_global():
        """
        Sets the `safe_global` parameter to True.
        """
        FisherTensor._safe_global = True

    def set_unsafe(self):
        """
        Causes the object not to perform checks if we input invalid data
        (__setattr__, @property, etc.)
        """
        self._safe = False

    def set_safe(self):
        """
        The inverse of `set_safe`.
        """
        self._safe = True

    def __eq__(self, other):
        """
        The equality operator.
        Returns True if the operands have the following properties:
            - are instances of FisherTensor
            - have same names (potentially shuffled)
            - have same dimensionality
            - have same fiducials (potentially shuffled)
            - have same data (potentially shuffled)
        """
        if set(self.names) != set(other.names):
            return False
        # index for re-shuffling parameters
        index = np.array([other.names.index(name) for name in self.names])
        return isinstance(other, FisherTensor) \
        and self.ndim == other.ndim \
        and len(self) == len(other) \
        and set(self.names) == set(other.names) \
        and np.allclose(
            self.fiducial[index],
            other.fiducial
        ) \
        and np.allclose(
            self.data[index],
            other.data
        )

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the tensor.
        """
        return np.ndim(self._data)

    def __len__(self):
        """
        Returns the number of parameters in the Fisher tensor.
        """
        return self._size

    @property
    def data(self):
        """
        Returns the object as a numpy array.
        """
        return self._data

    @data.setter
    def data(
        self,
        value,
    ):
        """
        Setter for the values of the FisherTensor.
        """
        if self.safe and FisherTensor.safe_global():
            _check_fisher_tensor(value, self.ndim)
        if len(value) != len(self) \
        or self.ndim != np.dim(value) \
        or not _is_square(value):
            raise ValueError
        self._data = value

    def diagonal(self, **kwargs):
        """
        Returns the diagonal elements of the Fisher tensor as a numpy array.
        """
        return np.diag(self._data, **kwargs)

    @property
    def size(self):
        """
        Returns the number of parameters in the Fisher tensor.
        """
        return self._size

    def drop(
        self,
        *names : Any,
    ) -> FisherTensor:
        """
        Removes parameters from the Fisher object.

        Parameters
        ----------
        names : Any
            the parameters to drop.
            If passing a list or a tuple, make sure to unpack it using the
            asterisk (*).

        Examples
        --------
        > m = FisherTensor(np.diag(1, 2, 3))
        > assert m.drop('p1', 'p3') == FisherTensor(np.diag(2), names=['p2']) # returns True
        """
        if not set(names).issubset(set(self.names)):
            raise ValueError
        if set(names) == set(self.names):
            raise ValueError('Unable to remove all parameters')

        data = self.data
        for index in range(self.ndim):
            data = np.delete(
                data,
                [self.names.index(name) for name in names],
                axis=index,
            )
        fiducial_new = np.array(
            [
                self.fiducial[self.names.index(name)] for name in self.names \
                if name not in names
            ]
        )
        names_new = [
            self.names[self.names.index(name)] for name in self.names \
            if name not in names
        ]

        # TODO how do we cast this to whatever the subclass is?
        return FisherTensor(
            data,
            names=names_new,
            fiducial=fiducial_new,
            ndim=self.ndim,
            safe=self.safe,
        )

    def trace(
        self,
        **kwargs,
    ):
        """
        Returns the trace of the Fisher tensor as a numpy array.
        """
        return np.trace(self._data, **kwargs)

    @property
    def fiducial(self):
        """
        Returns the fiducial values of the Fisher tensor as a numpy array.
        """
        return self._fiducial

    @fiducial.setter
    def fiducial(
        self,
        value,
    ):
        """
        The setter for the fiducial values of the FisherTensor.
        """
        if isinstance(value, dict):
            # check we don't have any stray parameters
            if not all(key in self.names for key in value.keys()):
                raise ValueError
            # the values must be float-able
            try:
                [float(_) for _ in value.values()]
            except TypeError as err:
                raise TypeError from err
            for key in value.keys():
                self.fiducial[self.names.index(key)] = float(value[key])
        else:
            if len(value) != len(self):
                raise ValueError
            try:
                self._fiducial = np.array([float(_) for _ in value])
            except TypeError as err:
                raise TypeError from err

    @property
    def names(self):
        """
        Returns the parameter names of the Fisher matrix.
        """
        return self._names

    @names.setter
    def names(
        self,
        value,
    ):
        """
        The bulk setter for the names.
        Alternatively, one can use `<instance>[<name>] = <new name>` for
        setting individual names.
        The length of the names must be the same as the one of the original
        object.
        """
        if len(set(value)) != len(self):
            raise ValueError
        self._names = value

    @property
    def names_latex(self):
        # TODO should this be a generic function instead? I.e. there may be
        # names other than latex ones
        """
        Returns the LaTeX names of the parameters of the Fisher matrix.
        """
        return NotImplemented

    def __add__(
        self,
        value : FisherTensor,
    ) -> FisherTensor:
        """
        Returns the result of adding two Fisher tensors.
        """
        # maybe it's a FisherTensor
        if not isinstance(value, FisherTensor):
            raise TypeError

        # make sure the dimensions match
        if value.ndim != self.ndim:
            raise ValueError

        # make sure they have the right parameters
        if set(value.names) != set(self.names):
            raise ValueError

        # make sure the fiducials match
        fiducial = np.array(
            [value.fiducial[self.names.index(x)] for x in value.names]
        )
        if not np.allclose(fiducial, self.fiducial):
            raise ValueError

        # TODO implement addition when parameters are shuffled
        return FisherTensor(
            data=self.data + value.data,
            names=self.names,
            fiducial=self.fiducial,
            safe=self.safe,
            ndim=self.ndim,
        )

    def __truediv__(
        self,
        value : Union[FisherTensor, float, int],
    ) -> FisherTensor:
        """
        Returns the result of dividing two Fisher tensors, or a Fisher tensor
        by a number.
        """
        # we can only divide two objects if they have the same dimensions and sizes
        try:
            value = float(value)
        except TypeError as err:
            # maybe it's a FisherTensor
            if isinstance(value, FisherTensor):
                # make sure they have the right parameters
                if set(value.names) != set(self.names):
                    raise ValueError
                # make sure the fiducials match
                fiducial = np.array(
                    [value.fiducial[self.names.index(x)] for x in value.names]
                )
                if not np.allclose(fiducial, self.fiducial):
                    raise ValueError

                if value.ndim == self.ndim:
                    # TODO parameter ordering again...
                    data = self.data / value.data
                else:
                    raise TypeError from err
            else:
                raise TypeError from err
        else:
            data = self.data / value

        return FisherTensor(
            data,
            names=self.names,
            fiducial=self.fiducial,
            safe=self.safe,
            ndim=self.ndim,
        )

    def __mul__(
        self,
        value : Union[FisherTensor, float, int, np.array],
    ) -> Union[FisherTensor, float]:
        """
        Returns the result of multiplying two Fisher tensors, or an int or float
        """
        # first idea: it's a float-like object
        try:
            value = float(value)
        except TypeError as err:
            # maybe it's a FisherTensor
            if isinstance(value, FisherTensor):
                # make sure they have the right parameters
                if set(value.names) != set(self.names):
                    raise ValueError
                # make sure the fiducials match
                fiducial = np.array(
                    [value.fiducial[self.names.index(x)] for x in value.names]
                )
                if not np.allclose(fiducial, self.fiducial):
                    raise ValueError

                # vector x vector (special case since we return a float)
                if value.ndim == 1 and self.ndim == 1:
                    return np.dot(self.data, value.data)

                # matrix x vector or vector x matrix
                if (value.ndim == 1 and self.ndim == 2) \
                or (value.ndim == 2 and self.ndim == 1):
                    data = np.dot(
                        self.data,
                        np.array([value.data[self.names.index(x)] for x in value.names])
                    )
                    ndim = 1

                # matrix x matrix
                elif value.ndim == 2 and self.ndim == 2:
                    # TODO make sure the names aren't shuffled
                    data = np.dot(
                        self.data,
                        value,
                    )
                    ndim = 2

                else:
                    return NotImplemented
            else:
                raise TypeError from err
        else:
            data = self.data * value
            ndim = self.ndim

        return FisherTensor(
            data,
            names=self.names,
            fiducial=self.fiducial,
            safe=self.safe,
            ndim=ndim,
        )

    def reparametrize(
        self,
        jacobian : Iterable[Any],
        names : Iterable[Any] = None,
    ):
        """
        Returns a new Fisher tensor with parameters `names`, which are
        related to the old ones via the transformation `jacobian`.
        Currently limited to rank 26 tensors (we run out of single letters in
        the English alphabet otherwise).
        Does not differentiate between covariant/contravariant indices.

        Parameters
        ----------
        transformation : array-like
            the Jacobian of the transformation

        names : array-like, default = None
            list of new names for the Fisher object. If None, uses the old names.
        """
        if self.ndim > 26:
            raise ValueError
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

        data = np.einsum(
            f'{index_transformation},{index_dummy}->{index_result}',
            *([jacobian] * self.ndim), self.data
        )

        if names is not None:
            names_new = names
            if len(set(names_new)) != self.names:
                raise ValueError
        else:
            # we don't transform the names
            names_new = self.names

        return FisherTensor(
            data,
            names=names_new,
            fiducial=self.fiducial,
            safe=self.safe,
            ndim=self.ndim,
        )


class FisherVector(FisherTensor):
    """
    Class for handling vectors which can be multiplied with Fisher matrices.
    As an example, see e.g. https://arxiv.org/abs/2004.12981, eq. (3.9)
    """

    def __init__(
        self,
        data,
        names = None,
        fiducial = None,
        **kwargs,
    ):
        # short circuiting if we're just typecasting
        if isinstance(data, FisherTensor) and data.ndim == 1:
            other = FisherVector(
                data.data,
                names=data.names,
                fiducial=data.fiducial,
                safe=data.safe,
            )
            self._data = copy.deepcopy(other.data)
            self._size = copy.deepcopy(other.size)
            self._names = copy.deepcopy(other.names)
            self._fiducial = copy.deepcopy(other.fiducial)
            self._safe = copy.deepcopy(other.safe)
            return

        # TODO fix this in the FisherTensor constructor
        if isinstance(data, FisherTensor) and data.ndim >= 2:
            raise ValueError

        super().__init__(
            data=data,
            names=names,
            fiducial=fiducial,
            ndim=1,
            **kwargs
        )

    @staticmethod
    def join(*vectors : FisherVector):
        """
        Joins (concatenates) a bunch of Fisher vectors into a new one and
        returns the result.
        """
        return NotImplemented

    def __add__(
        self,
        value : FisherVector,
    ) -> FisherVector:
        """
        Returns the result of adding two Fisher vectors.
        """
        result = super().__add__(value)
        return FisherVector(result)

    def __truediv__(
        self,
        value : Union[FisherVector, float, int],
    ) -> FisherVector:
        """
        Returns the result of dividing two Fisher vectors, or a Fisher vector
        by a number.
        """
        return FisherVector(super().__truediv__(value))

    def __mul__(
        self,
        value : Union[float, int, FisherVector],
    ) -> Union[float, FisherVector]:
        """
        Returns the result of multiplying two Fisher vectors, or a Fisher
        vector by an int or float
        """
        result = super().__mul__(value)

        try:
            result = float(result)
        except TypeError as err:
            if isinstance(result, FisherTensor):
                if result.ndim == 1:
                    return FisherVector(result)
                if result.ndim == 2:
                    return FisherMatrix(result)
        else:
            return result

        return NotImplemented

    def _repr_html_(self):
        """
        HTML representation of the FisherVector.
        Useful when using a Jupyter notebook.
        """

        body = '<tbody>'
        for index, name in enumerate(self._names):
            body += f'<tr><th>{name}</th>' + '<td>{:.3f}</td>'.format(self.data[index]) + '</tr>'
        body += '</tbody>'

        return f'<table>{body}</table>'


class FisherMatrix(FisherTensor):
    """
    Class for handling Fisher matrices.
    """
    # TODO implement caching maybe?

    def __init__(
        self,
        data,
        names = None,
        fiducial = None,
        **kwargs,
    ):
        """
        Constructor for Fisher matrix.

        Parameters
        ----------
        data : array-like or string (path to file) or dict or FisherMatrix
            The data of the Fisher matrix.
            If it's a string, accepts the same keyword arguments as numpy.loadtxt.
            If it's a dict, it creates a diagonal matrix, and the `names`
            argument is ignored.
            If it's array-like, it attempts to convert it to a numpy array.

        names : array-like, default = None
            The names of the parameters.
            If not specified (a None-like object), default to `p1, ..., pn`.
            Must be a hashable type

        fiducial : array-like, default = None
            The fiducial values of the parameters. If not specified (a
            None-like object), default to 0 for all parameters.
        """

        # short circuiting if we're just typecasting
        if isinstance(data, FisherTensor) and data.ndim in [1, 2]:
            other = FisherMatrix(
                data.data,
                names=data.names,
                fiducial=data.fiducial,
                safe=data.safe,
            )
            self._data = copy.deepcopy(other.data)
            self._size = copy.deepcopy(other.size)
            self._names = copy.deepcopy(other.names)
            self._fiducial = copy.deepcopy(other.fiducial)
            self._safe = copy.deepcopy(other.safe)
            return

        # TODO fix this in the FisherTensor constructor
        if isinstance(data, FisherTensor) and data.ndim >= 3:
            raise ValueError

        # just run the constructor for a FisherTensor of dimension 2
        super().__init__(
            data,
            names=names,
            fiducial=fiducial,
            ndim=2,
            **kwargs
        )

    def diagonal(self):
        """
        Returns the diagonal elements of the Fisher matrix as a numpy array.
        """
        return super().diagonal()

    def trace(self):
        """
        Returns the trace of the Fisher matrix.
        """
        return super().trace()

    def eigenvalues(self):
        """
        Returns the eigenvalues of the Fisher matrix as a numpy array.
        """
        return np.linalg.eigvalsh(self._data)

    def eigenvectors(self):
        """
        Returns the right eigenvectors of the Fisher matrix as a numpy array.
        """
        return np.linalg.eigh(self._data)[-1]

    def determinant(self):
        """
        Returns the determinant of the matrix.
        """
        return np.linalg.det(self._data)

    def constraints(
        self,
        marginalized : bool = True,
        sigma : float = 1.0,
    ):
        """
        Returns the constraints on the parameters as a [TODO].

        Parameters
        ----------
        marginalized : bool, default = True
            whether we want the marginalized or the unmarginalized constraints.

        sigma : float, default = 1.0
            how many sigmas away.
        """
        if sigma <= 0:
            raise ValueError

        if marginalized:
            inv = self.inverse()
            return np.sqrt(np.diag(inv.data)) * sigma

        return 1. / np.sqrt(np.diag(self.data)) * sigma


    @staticmethod
    def blockdiagonal(*matrices):
        """
        Returns a block diagonal matrix of Fisher elements.
        The names of the parameters should be unique.
        """
        # TODO see if we can make them unique if they aren't already
        return NotImplemented

    def inverse(
        self,
        inplace : bool = False,
    ):
        """
        Returns the inverse of the Fisher matrix as another FisherMatrix.

        Parameters
        ----------
        inplace : bool, default = False
            if True, modifies the existing FisherMatrix and returns None
        """
        # inverse satisfies properties of Fisher matrix, see:
        # https://math.stackexchange.com/a/26200
        if not inplace:
            return FisherMatrix(
                np.linalg.inv(self.data),
                names=self.names,
                fiducial=self.fiducial,
                safe=self.safe,
            )
        self._data = np.linalg.inv(self.data)

    def __truediv__(
        self,
        value : Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of dividing two Fisher matrices, or a Fisher matrix
        by a number.
        """
        return FisherMatrix(super().__truediv__(value))

    def __add__(self, value):
        return FisherMatrix(super().__add__(value))

    def __mul__(
        self,
        value : Union[FisherMatrix, float, int, FisherVector, np.array],
    ) -> Union[FisherMatrix, float, FisherVector]:
        """
        Returns the result of multiplying two Fisher matrices, or by a
        FisherVector, an int, or float
        """
        result = super().__mul__(value)
        try:
            result = float(value)
        except TypeError as err:
            if isinstance(result, FisherTensor):
                if result.ndim == 1:
                    return FisherVector(result)
                if result.ndim == 2:
                    return FisherMatrix(result)
        else:
            return result

        return NotImplemented

    def is_diagonal(self):
        """
        Checks whether the Fisher matrix is diagonal.
        """
        return np.all(self.data == np.diag(np.diagonal(self.data)))

    def _repr_html_(self):
        """
        HTML representation of the FisherMatrix.
        Useful when using a Jupyter notebook.
        """
        header_matrix = '<thead><tr><th></th>' + (
            '<th>{}</th>' * len(self)
        ).format(*self._names) + '</tr></thead>'

        body_matrix = '<tbody>'
        for index, name in enumerate(self._names):
            body_matrix += f'<tr><th>{name}</th>' + (
                '<td>{:.3f}</td>' * len(self)
            ).format(*(self.data[:, index])) + '</tr>'
        body_matrix += '</tbody>'

        html_matrix = f'<table>{header_matrix}{body_matrix}</table>'

        return html_matrix

    def pprint_eigenvalues(
        self,
        **kwargs,
    ):
        """
        Shotcut for pretty printing eigenvalues.
        """
        fmt_values = kwargs.pop('fmt_values', '{:.3f}')
        return _HTML_Wrapper(
            _make_html_table(
                self.eigenvalues(),
                fmt_values=fmt_values,
            )
        )

    def pprint_constraints(
        self,
        **kwargs,
    ):
        """
        Shortcut for pretty printing constraints.
        """
        fmt_values = kwargs.pop('fmt_values', '{:.3f}')
        return _HTML_Wrapper(
            _make_html_table(
                self.constraints(**kwargs),
                names=self.names,
                fmt_values=fmt_values,
            )
        )

    def pprint_fiducial(
        self,
        **kwargs,
    ):
        """
        Shortcut for pretty printing the fiducial values.
        """
        fmt_values = kwargs.pop('fmt_values', '{:.3f}')
        return _HTML_Wrapper(
            _make_html_table(
                self.fiducial,
                names=self.names,
                fmt_values=fmt_values,
            )
        )
