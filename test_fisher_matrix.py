"""
Various tests of the FisherTensor class.
"""

import numpy as np
from scipy.stats import ortho_group
import pytest


from fisher_utils import \
    ParameterNotFoundError, \
    MismatchingSizeError, \
    is_iterable, \
    float_to_latex, \
    make_default_names, \
    is_square, \
    is_symmetric, \
    is_positive_semidefinite, \
    get_index_of_other_array, \
    jsonify, \
    reindex_array

from fisher_matrix import from_file
from fisher_matrix import FisherMatrix as FisherTensor



class TestFisherUtils:
    """
    Tests of the various helper utilities.
    """
    def test_is_iterable(self):
        assert is_iterable('asdf')
        assert is_iterable([1, 2, 3])
        assert is_iterable((1, 2, 3))
        assert not is_iterable(1)


    def test_float_to_latex(self):
        assert float_to_latex(1e-3) == r'10^{-3}'
        assert float_to_latex(2e-3) == r'2 \times 10^{-3}'
        assert float_to_latex(1e-4) == r'10^{-4}'
        assert float_to_latex(1e-5) == r'10^{-5}'
        assert float_to_latex(100) == r'100'
        assert float_to_latex(345) == r'345'
        # the below has three significant digits
        assert float_to_latex(1234) == r'1.23 \times 10^{3}'


    def test_make_default_names(self):
        assert np.all(make_default_names(3) == np.array(['p1', 'p2', 'p3']))


    def test_is_square(self):
        assert is_square([[1, 0], [1, 1]])
        # 1D arrays pass the check as well
        assert is_square([1, 2])
        assert is_square(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ]
        )

        assert not is_square([[1, 2]])
        assert not is_square([[1, 2], [3, 4, 5]])
        assert not is_square([[1, 2], [8]])


    def test_is_symmetric(self):
        # 1D arrays are automatically symmetric
        assert is_symmetric([1, 2])
        assert is_symmetric([[1, 0], [0, 1]])
        assert is_symmetric([[3, -2], [-2, 1]])

        # generate a random orthogonal matrix with a fixed seed
        ort = ortho_group.rvs(3, random_state=12345)
        # by definition, the matrix O^T @ D @ O is symmetric, where D is a
        # diagonal matrix
        assert is_symmetric(ort.T @ np.diag([1, 2, 3]) @ ort)


    def test_is_positive_semidefinite(self):
        # generate a random orthogonal matrix with a fixed seed
        ort = ortho_group.rvs(3, random_state=12345)
        assert is_positive_semidefinite(ort.T @ np.diag([1, 2, 3]) @ ort)
        assert is_positive_semidefinite([[1, 0], [0, 1]])


    def test_get_index_of_other_array(self):
        A = list('asdf')

        B = list('fsda')
        assert np.allclose(
            [3, 1, 2, 0],
            get_index_of_other_array(A, B),
        )


    def test_reindex_array(self):
        A = list('asdf')
        B = list('fsda')
        C = list('fdas')

        assert np.all(
            reindex_array(B, get_index_of_other_array(A, B)) == A
        )

        assert np.all(
            reindex_array(A, get_index_of_other_array(B, A)) == B
        )

        assert np.all(
            reindex_array(C, get_index_of_other_array(A, C)) == A
        )

        assert np.all(
            reindex_array(A, get_index_of_other_array(C, A)) == C
        )



class TestFisherTensor:
    """
    Tests for the Fisher object.
    """
    def test_init(self):
        # a file
        FisherTensor(np.loadtxt('test_matrix.dat', comments='#'))
        # a 2D array
        FisherTensor(np.diag([1, 2, 3]))

        # cannot use a string literal as names
        with pytest.raises(TypeError):
            FisherTensor(np.diag([1, 2, 3]), names='123')

        # non-positive diagonal
        assert not FisherTensor(np.diag([-1, 2, 3])).is_valid()

        # non-square matrix
        with pytest.raises(ValueError):
            FisherTensor(np.array([[1, 2], [3, 4], [5, 6]]))

        # not a matrix
        with pytest.raises(ValueError):
            FisherTensor(np.array([[[1], [2], [3]]]))

        # not a symmetric matrix
        assert not FisherTensor(np.array([[1, 0], [1, 1]])).is_valid()

        # get the LaTeX names right
        assert np.all(
            FisherTensor(np.diag([1, 2, 3]), names=['a', 'b', 'c']).names_latex == ['a', 'b', 'c']
        )


    def test_from_file(self):
        fm = from_file('example_fisher.json')
        assert fm == FisherTensor(
            np.diag([2, 1, 3]),
            names=list('bac'),
            names_latex=[r'\mathcal{B}', r'\mathcal{A}', r'\mathcal{C}'],
            fiducial=[5, 4, 6],
        )


    def test_getitem(self):
        data = FisherTensor(np.diag([1, 2, 3]))
        assert data['p1', 'p1'] == 1
        assert data['p1', 'p2'] == 0
        assert data['p2', 'p2'] == 2
        assert data['p3', 'p3'] == 3
        assert np.all(np.array(['p1', 'p2', 'p3']) == data.names)

        # same type, keys don't exist
        with pytest.raises(ParameterNotFoundError):
            data['p0', 'p0']

        # wrong specifier
        with pytest.raises(ParameterNotFoundError):
            data['p0']

        # wrong type of key
        with pytest.raises(TypeError):
            data[1]

        # wrong type, keys don't exist
        with pytest.raises(ParameterNotFoundError):
            data[1, 1]


    def test_setitem(self):
        data = FisherTensor(np.diag([1, 2, 3]))

        # assignment should succeed
        # NOTE the matrix should still be positive semi-definite
        data['p1', 'p2'] = 1

        # matrix should remain symmetric
        assert np.allclose(np.transpose(data.values), data.values)

        data['p1', 'p1'] = -1
        assert not data.is_valid()

        # trying to set a nonexisting element
        with pytest.raises(ParameterNotFoundError):
            data['p0', 'p0'] = 1


    def test_constraints(self):
        data = FisherTensor([[1, -1], [-1, 1.2]])
        assert np.all(data.constraints(marginalized=True, sigma=2) >= 0)
        assert np.all(data.constraints(marginalized=False, sigma=2) >= 0)
        assert np.allclose(
            FisherTensor(np.diag([1, 2, 3])).constraints(marginalized=True),
            FisherTensor(np.diag([1, 2, 3])).constraints(marginalized=False)
        )

        with pytest.raises(ValueError):
            data.constraints(sigma=-1)

        with pytest.raises(ValueError):
            data.constraints(sigma=1, p=0.3)

        with pytest.raises(ValueError):
            data.constraints(p=-2)


    def test_sort(self):
        m = FisherTensor([
            [11, 12, 13],
            [21, 22, 23],
            [31, 32, 33]],
            fiducial=[-1, 0, 1],
            names=['p3', 'p1', 'p2'],
        )

        m_new = m.sort()

        assert m_new == FisherTensor([
            [33, 31, 32],
            [13, 11, 12],
            [23, 21, 22]],
            fiducial=[1, -1, 0],
            names=['p1', 'p2', 'p3']
        )


    def test_drop(self):
        data = FisherTensor(np.diag([1, 2, 3]), fiducial=[-1, 0, 1])
        data_new = data.drop('p1')

        assert np.allclose(data_new.values, np.diag([2, 3]))
        for old, new in zip(data_new.names, ['p2', 'p3']):
            assert old == new
        #assert np.all(data_new.names == np.array(['p2', 'p3']))
        assert np.allclose(data_new.fiducial, np.array([0, 1]))

        with pytest.raises(ValueError):
            data_new = data.drop(*data.names)


    def test_add(self):
        m1 = FisherTensor(np.diag([1, 2, 3]))
        m2 = FisherTensor(np.diag([6, 5, 4]))
        assert m1 + m2 == FisherTensor(np.diag([7, 7, 7]))


    def test_mul(self):
        """
        Test for multiplication by a number or FisherTensor (elementwise).
        """
        m1 = FisherTensor([[2, -1], [-1, 3]])

        assert m1 * 2 == FisherTensor(m1.values * 2)
        assert 2 * m1 == FisherTensor(m1.values * 2)

        r1 = FisherTensor(m1.values) * FisherTensor(1 / m1.values)
        r2 = FisherTensor(m1.values * (1 / m1.values))
        assert r1 == r2


    def test_matmul(self):
        """
        Test for matrix multiplication.
        """
        m1 = FisherTensor([[1, -2], [-2, 4]])
        m2 = FisherTensor([[2, 0], [0, 3]])

        assert m1 @ m2 == FisherTensor(m1.values @ m2.values)
        assert m2 @ m1 == FisherTensor(m2.values @ m1.values)


    def test_truediv(self):
        """
        Test for division by a number or FisherTensor (elementwise).
        """
        m = FisherTensor([[2, -1], [-1, 3]])
        assert m / 2 == FisherTensor(m.values / 2)
        assert m / m == FisherTensor(np.ones((m.size, m.size)))
        # the other stuff should raise errors
        with pytest.raises(TypeError):
            2 / m


    def test_reparametrize(self):
        m = FisherTensor([[2, -1], [-1, 3]])
        jacobian_m = np.array([[3, 2], [6, 7]])
        m_new = m.reparametrize(jacobian_m)
        assert np.allclose(m_new.values, np.array([[18, 45], [45, 135]]))