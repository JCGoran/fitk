import numpy as np

import pytest

from fisherinformation import FisherMatrix

class TestFisherMatrix:
    def test_init(self):
        # a file
        FisherMatrix('test_matrix.dat', comments='#')
        # a dictionary
        FisherMatrix({'a' : 1, 'b' : 2})
        # an array
        FisherMatrix(np.diag([1, 2, 3]))
        # a FisherMatrix
        FisherMatrix(FisherMatrix(np.diag([1, 2, 3])))

        # unhashable type for names
        with pytest.raises(TypeError):
            FisherMatrix(np.diag([1, 2, 3]), names=[['a','b'], 'c', 'd'])

        # non-positive diagonal
        with pytest.raises(ValueError):
            FisherMatrix(np.diag([-1, 2, 3]))

        # non-square matrix
        with pytest.raises(ValueError):
            FisherMatrix(np.array([[1, 2], [3, 4], [5, 6]]))

        # not a matrix
        with pytest.raises(ValueError):
            FisherMatrix(np.array([[[1], [2], [3]]]))

        # not a symmetric matrix
        with pytest.raises(ValueError):
            FisherMatrix(np.array([[1, 0], [1, 1]]))

    def test_getitem(self):
        data = FisherMatrix(np.diag([1, 2, 3]))
        assert data['p1', 'p1'] == 1
        assert data['p1', 'p2'] == 0
        assert data['p2', 'p2'] == 2
        assert data['p3', 'p3'] == 3
        assert all(x == y for x, y in zip(np.array(['p1', 'p2', 'p3']), data.parameters))

        # same type, keys don't exist
        with pytest.raises(ValueError):
            data['p0', 'p0']

        # wrong specifier
        with pytest.raises(ValueError):
            data['p0']

        # wrong type of key
        with pytest.raises(TypeError):
            data[1]

        # wrong type, keys don't exist
        with pytest.raises(ValueError):
            data[1, 1]

    def test_setitem(self):
        data = FisherMatrix(np.diag([1, 2, 3]))

        # assignment should succeed
        data['p1', 'p2'] = 5

        # matrix should remain symmetric
        assert np.allclose(np.transpose(data.matrix), data.matrix)

        # cannot have negative element on diagonal
        with pytest.raises(ValueError):
            data['p1', 'p1'] = -1

        # trying to set a nonexisting element
        with pytest.raises(ValueError):
            data['p0', 'p0'] = 1
