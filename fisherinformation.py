from __future__ import annotations

from collections import OrderedDict
import copy
from typing import NamedTuple, Sequence, Mapping, Iterable, Optional, Callable, \
    Union, List, Any, Dict, Set, Tuple

import numpy as np

def _default_names(size : int):
    """Returns the numpy array with default names `p1, ..., pn`"""
    if size < 0:
        raise ValueError
    return np.array([f'p{_ + 1}' for _ in range(size)])


def _check_fisher_matrix(data):
    # check it's a matrix
    if np.ndim(data) != 2:
        raise ValueError(
            f'Expected 2 dimensions, got {np.ndim(data)}.'
        )

    # -||- that is square
    if np.shape(data)[0] != np.shape(data)[-1]:
        raise ValueError(
            f'Expected a square matrix, got {np.shape(data)}'
        )

    # -||- with positive diagonal elements
    for element in np.diag(data):
        if element <= 0:
            raise ValueError(
                f'Found non-positive diagonal element: {element}'
            )

    # -||- that is also symmetric
    if not np.allclose(data, np.transpose(data)):
        raise ValueError(
            f'The matrix {data} must be symmetric'
        )


def _check_fisher_vector(data):
    # check it has the right dimensions
    if np.ndim(data) != 1:
        raise ValueError(
            f'Expected 1 dimension, got {np.ndim(data)}.'
        )


class FisherVector:
    """
    Class for handling vectors which can be multiplied with Fisher matrices.
    As an example, see e.g. https://arxiv.org/abs/2004.12981, eq. (3.9)
    """

    def __init__(self):
        """
        Constructor for Fisher vector.

        Parameters
        ----------
        data : array-like or string (path to file) or dict or FisherVector
            The data of the Fisher vector. It tries to guess what the user
            meant, and raises an error if it's unable to unambiguously figure it out.

        names : array-like, default = None
            The names of the parameters.
            If not specified (a None-like object), default to `p1, ..., pn`.

        fiducial : array-like, default = None
            The fiducial values of the parameters. If not specified (a
            None-like object), default to 0 for all parameters.
        """
        pass

    @property
    def fiducial(self):
        """
        Returns the fiducial values of the Fisher vector as a numpy array(?)
        """
        pass

    @property
    def parameters(self):
        """
        Returns the parameter names of the Fisher vector.
        """
        pass

    def parameters_latex(self):
        # TODO should this be a generic function instead? I.e. there may be
        # names other than latex ones
        """
        Returns the LaTeX names of the parameters of the Fisher vector.
        """
        pass

    def __add__(
        self,
        value : FisherVector,
    ) -> FisherVector:
        """
        Returns the result of adding two Fisher vectors.
        """
        pass

    def __truediv__(
        self,
        value : Union[FisherVector, float, int],
    ) -> FisherVector:
        """
        Returns the result of dividing two Fisher vectors, or a Fisher vector
        by a number.
        """
        pass

    def __rmul__(
        self,
        value : Union[float, int, FisherVector],
    ) -> Union[float, FisherVector]:
        # see https://stackoverflow.com/questions/6892616/python-multiplication-override
        pass

    def __mul__(
        self,
        value : Union[float, int, FisherVector],
    ) -> Union[float, FisherVector]:
        """
        Returns the result of multiplying two Fisher vectors, or a Fisher
        vector by an int or float
        """
        pass


class FisherMatrix:
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
        if isinstance(data, FisherMatrix):
            self._data = copy.deepcopy(data._data)
            self._size = copy.deepcopy(data._size)
            self._names = copy.deepcopy(data._names)
            self._fiducial = copy.deepcopy(data._fiducial)
            return

        _isdict = False

        if isinstance(data, str):
            self._data = np.loadtxt(data, **kwargs)
        # NOTE if we pass a dict, the Fisher matrix will always be diagonal,
        # and the `names` parameter passed is not necessary
        elif isinstance(data, (dict, OrderedDict)):
            self._data = np.diag(list(data.values()))
            _isdict = True
        else:
            # try to treat it as an array-like object
            self._data = np.array(data)

        # check that the data is sane
        _check_fisher_matrix(self._data)

        # we can access the element if the above check succeeded
        self._size = np.shape(self._data)[0]

        # setting the names
        if _isdict:
            self._names = list(data.keys())
        else:
            if names is None:
                self._names = _default_names(self._size)
            else:
                try:
                    _ = iter(names)
                except:
                    raise TypeError
                if len(names) != self._size:
                    raise ValueError
                # this will catch anything that isn't hashable
                if len(set(names)) != len(names):
                    raise ValueError('The parameter names must be unique.')
                self._names = names

        # setting the fiducial
        if not fiducial:
            self._fiducial = np.zeros(self._size)
        else:
            try:
                _ = iter(fiducial)
            except:
                raise TypeError
            # fiducials must be numbers
            if not all(isinstance(_, (int, float)) for _ in fiducial):
                raise TypeError
            if len(fiducial) != self._size:
                raise ValueError
            self._fiducial = np.array(fiducial)


    def __getitem__(
        self,
        keys,
    ):
        """Implements dictionary-like access to items in the Fisher matrix."""
        try:
            _ = iter(keys)
        except TypeError as err:
            raise err

        if len(keys) != 2:
            raise ValueError(f'Got length {keys}')

        for key in keys:
            if key not in self._names.tolist():
                raise ValueError(
                    f'Parameter {key} not found'
                )

        return self._data[np.where(self._names == keys[0]), np.where(self._names == keys[1])]

    def __setitem__(
        self,
        keys,
        value,
    ):
        """Implements setting of elements in the Fisher matrix."""
        try:
            _ = iter(keys)
        except TypeError as err:
            raise err

        if len(keys) != 2:
            raise ValueError

        index1 = np.where(self._names == keys[0])
        index2 = np.where(self._names == keys[1])

        # for some reason, np.where returns a tuple
        if not index1[0].size or not index2[0].size:
            raise ValueError

        if index1 != index2:
            self._data[index1, index2] = value
            # also update the symmetric part
            self._data[index2, index1] = value
        else:
            # value on the diagonal must be positive
            if value <= 0:
                raise ValueError
            self._data[index1, index1] = value

    def rename(
        self,
        parameters : Union[Iterable, dict, OrderedDict, Callable],
        inplace : bool = False,
    ) -> Union[FisherMatrix, None]:
        """
        Renames existing parameters.

        Parameters
        ----------
        parameters : array-like, or dict, or function
            how the new parameters are mapped to the old ones.

        inplace : bool, default = False
            should the function be called in-place.
            Returns None if `inplace=True`.
        """

        if isinstance(parameters, (dict, OrderedDict)):
            for key in parameters.keys():
                if key not in self._names:
                    raise ValueError
            if len(set(parameters.values())) != len(self):
                raise ValueError
            if inplace:
                for key in parameters.keys():
                    self._names[self._names.index(key)] = parameters[key]
            else:
                return FisherMatrix(
                    self._data,
                    names=[self._names[self._names.index(key)] for key in parameters.keys()],
                    fiducial=self._fiducial,
                )

        elif callable(parameters):
            try:
                temp_names = [parameters(name) for name in self._names]
            except:
                raise ValueError
            if len(set(temp_names)) != len(self):
                raise ValueError
            if inplace:
                self._names = temp_names
            else:
                return FisherMatrix(
                    self._data,
                    names=temp_names,
                    fiducial=self._fiducial,
                )


        else:
            if len(set(parameters)) != len(self):
                raise ValueError
            if inplace:
                self._names = parameters
            else:
                return FisherMatrix(
                    self._data,
                    names=parameters,
                    fiducial=self._fiducial,
                )

    def __eq__(self, other):
        # TODO implement comparison when the parameters are shuffled
        pass

    def __len__(self):
        """
        Returns the number of parameters in the Fisher matrix, NOT the
        number of elements in the matrix (which is nparams^2).
        """
        return self._size

    @property
    def matrix(self):
        """
        Returns the Fisher matrix as a numpy array.
        """
        return self._data

    @property
    def diagonal(self):
        """
        Returns the diagonal elements of the Fisher matrix as a numpy array.
        """
        return np.diag(self._data)

    @property
    def size(self):
        """
        Returns the number of parameters in the Fisher matrix.
        """
        return self._size

    @property
    def eigenvalues(self):
        """
        Returns the eigenvalues of the Fisher matrix as a numpy array.
        """
        return np.linalg.eigvalsh(self._data)

    @property
    def eigenvectors(self):
        """
        Returns the right eigenvectors of the Fisher matrix as a numpy array.
        """
        return np.linalg.eigh(self._data)[-1]

    def drop_parameters(self, parameters : Iterable):
        """
        Removes parameters from the Fisher matrix.
        """
        return NotImplemented

    @property
    def fiducial(self):
        """
        Returns the fiducial values of the Fisher matrix as a numpy array(?)
        """
        return np.array(self._fiducial)

    @property
    def tr(self):
        """
        Returns the trace of the Fisher matrix as a numpy array.
        """
        return np.trace(self._data)

    @staticmethod
    def block(*matrices):
        """
        Returns a block matrix of Fisher elements.
        The names of the parameters should be unique.
        """
        # TODO see if we can make them unique
        return NotImplemented

    @fiducial.setter
    def fiducial(self, value):
        if len(value) != len(self):
            raise ValueError
        if not all(isinstance(element, (int, float)) for element in value):
            raise TypeError
        self._fiducial = value

    @property
    def inverse(self):
        """
        Returns the inverse of the Fisher matrix.
        """
        return np.linalg.inv(self._data)

    @property
    def parameters(self):
        """
        Returns the parameter names of the Fisher matrix.
        """
        return self._names

    def parameters_latex(self):
        # TODO should this be a generic function instead? I.e. there may be
        # names other than latex ones
        """
        Returns the LaTeX names of the parameters of the Fisher matrix.
        """
        pass

    def __add__(
        self,
        value : FisherMatrix,
    ) -> FisherMatrix:
        """
        Returns the result of adding two Fisher matrices.
        """
        pass

    def __truediv__(
        self,
        value : Union[FisherMatrix, float, int],
    ) -> FisherMatrix:
        """
        Returns the result of dividing two Fisher matrices, or a Fisher matrix
        by a number.
        """
        pass

    def __rmul__(
        self,
        value : Union[FisherMatrix, float, int, FisherVector],
    ) -> Union[FisherMatrix, float, FisherVector]:
        # see https://stackoverflow.com/questions/6892616/python-multiplication-override
        pass

    def __mul__(
        self,
        value : Union[FisherMatrix, float, int, FisherVector, np.array],
    ) -> Union[FisherMatrix, float, FisherVector]:
        """
        Returns the result of multiplying two Fisher matrices, or by a
        FisherVector, an int, or float
        """
        # NOTE add, truediv, and mul should all be able to handle their
        # respective operations when the other value is an int, a float, or a
        # FisherVector
        # of course, if we multiply a FisherMatrix with a FisherVector (or a
        # plain old list|tuple|dict|numpy array), the result should be another
        # FisherVector
        pass

    def change_parameters(
        self,
        parameters_new : Iterable[Any],
        jacobian : Iterable[Any],
    ):
        """
        Returns a new Fisher matrix with parameters `parameters_new`, which are
        related to the old ones via a transformation `jacobian`.
        """
        pass
