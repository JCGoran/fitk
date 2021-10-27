import numpy as np

import pytest

from fisherinformation import FisherVector, FisherMatrix, FisherTensor


class TestFisherVector:
    def test_init(self):
        # a file
        FisherVector('test_vector.dat', comments='#')
        # a dictionary
        FisherVector({'a' : 1, 'b' : 2})
        # a flat array
        FisherVector([1, 2, 3])
        # a 1D FisherTensor
        FisherVector(FisherTensor([1, 2, 3]))

        # cannot convert FisherMatrix to FisherVector
        with pytest.raises(ValueError):
            FisherVector(FisherMatrix([1, 2, 3]))

class TestFisherMatrix:
    def test_init(self):
        # a file
        FisherMatrix('test_matrix.dat', comments='#')
        # a dictionary
        FisherMatrix({'a' : 1, 'b' : 2})
        # a flat array
        FisherMatrix([1, 2, 3])
        # a 2D array
        FisherMatrix(np.diag([1, 2, 3]))
        # a FisherMatrix
        FisherMatrix(FisherMatrix(np.diag([1, 2, 3])))
        # a FisherVector
        FisherMatrix(FisherVector([1, 2, 3]))
        # a 2D FisherTensor
        FisherMatrix(FisherTensor([1, 2, 3]))

        # cannot use a string literal as names
        with pytest.raises(TypeError):
            FisherMatrix([1, 2, 3], names='123')

        # this will still work though, since we can iterate over the
        # fiducial and cast each element into a float
        FisherMatrix([1, 2, 3], names=list('123'), fiducial='123')

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
        assert ['p1', 'p2', 'p3'] == data.names

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
        # NOTE the matrix should still be positive semi-definite
        data['p1', 'p2'] = 1

        # matrix should remain symmetric
        assert np.allclose(np.transpose(data.data), data.data)

        # cannot have negative element on diagonal
        with pytest.raises(ValueError):
            data['p1', 'p1'] = -1

        # trying to set a nonexisting element
        with pytest.raises(ValueError):
            data['p0', 'p0'] = 1

    def test_constraints(self):
        data = FisherMatrix([[1, -1], [-1, 1.2]])
        assert np.all(data.constraints(marginalized=True) >= 0)
        assert np.all(data.constraints(marginalized=False) >= 0)
        assert np.allclose(
            FisherMatrix([1, 2, 3]).constraints(marginalized=True),
            FisherMatrix([1, 2, 3]).constraints(marginalized=False)
        )

        with pytest.raises(ValueError):
            data.constraints(sigma=-1)

    def test_sort(self):
        data = FisherMatrix([1, 2, 3], fiducial=[-1, 0, 1], names=['p3', 'p1', 'p2'])
        data_new = data.sort()
        assert data_new == FisherMatrix([3, 1, 2], fiducial=[1, -1, 0])

    def test_drop(self):
        data = FisherMatrix([1, 2, 3], fiducial=[-1, 0, 1])
        data_new = data.drop(['p1'])

        assert np.allclose(data_new.data, np.diag([2, 3]))
        assert data_new.names == ['p2', 'p3']
        assert np.allclose(data_new.fiducial, np.array([0, 1]))

        with pytest.raises(ValueError):
            data_new = data.drop(data.names)

    def test_float(self):
        data = FisherMatrix([3])
        with pytest.raises(TypeError):
            float(data)

    def test_add(self):
        m1 = FisherMatrix([1, 2, 3])
        m2 = FisherMatrix([6, 5, 4])
        assert m1 + m2 == FisherMatrix([7, 7, 7])

    def test_mul(self):
        m1 = FisherMatrix([[2, -1], [-1, 3]])
        v1 = FisherVector([1, -1])
        assert m1 * v1 == FisherVector([3, -4])
        # the result commutes for symmetric matrices
        assert v1 * m1 == FisherVector([3, -4])
        assert np.allclose(v1 * v1, 2)
        # it's not commutative though
        assert (m1 * v1) * v1 != m1 * (v1 * v1)
        # if the order is right, it should be fine
        assert (v1 * m1) * v1 == v1 * (m1 * v1)

    def test_truediv(self):
        m = FisherMatrix([[2, -1], [-1, 3]])
        v = FisherVector([3, 4])
        assert m / 2 == FisherMatrix([[1, -0.5], [-0.5, 3 / 2.]])
        # we only define matrix division for same-sized objects
        # also, due to potential divisions by zero and negative eigenvalues,
        # we turn off the safety flags
        FisherTensor.set_unsafe_global()
        assert m / m == FisherMatrix(np.ones((m.size, m.size)))
        assert v / v == FisherVector(np.ones(v.size))
        # the other stuff should raise errors
        with pytest.raises(TypeError):
            2 / m
        with pytest.raises(TypeError):
            v / m
        with pytest.raises(TypeError):
            m / v
        FisherTensor.set_safe_global()

    def test_reparametrize(self):
        FisherTensor.set_unsafe_global()
        data = FisherMatrix([[2, -1], [-1, 3]])
        jacobian = np.array([[3, 2], [6, 7]])
        data_new = data.reparametrize(jacobian)
        assert np.allclose(data_new.data, np.array([[18, 45], [45, 135]]))
        FisherTensor.set_safe_global()
