"""
Tests of the utilities of FITK
"""

from __future__ import annotations

# standard library imports
from itertools import product

# third-party imports
import numpy as np
import pytest
from scipy.stats import ortho_group

# first party imports
from fitk.fisher_utils import (
    MismatchingSizeError,
    MismatchingValuesError,
    ParameterNotFoundError,
    find_diff_weights,
    float_to_latex,
    get_index_of_other_array,
    is_iterable,
    is_positive_semidefinite,
    is_square,
    is_symmetric,
    make_default_names,
    math_mode,
    process_units,
    reindex_array,
)


class TestFisherUtils:
    """
    Tests of the various helper utilities.
    """

    def test_is_iterable(self):
        assert is_iterable("asdf")
        assert is_iterable([1, 2, 3])
        assert is_iterable((1, 2, 3))
        assert not is_iterable(1)

    def test_float_to_latex(self):
        assert float_to_latex(1e-3) == r"10^{-3}"
        assert float_to_latex(2e-3) == r"2 \times 10^{-3}"
        assert float_to_latex(1e-4) == r"10^{-4}"
        assert float_to_latex(1e-5) == r"10^{-5}"
        assert float_to_latex(100) == r"100"
        assert float_to_latex(345) == r"345"
        # the below has three significant digits
        assert float_to_latex(1234) == r"1.23 \times 10^{3}"

    def test_make_default_names(self):
        assert np.all(make_default_names(3) == np.array(["p1", "p2", "p3"]))
        assert np.all(make_default_names(2, "p") == ["p1", "p2"])

        with pytest.raises(ValueError):
            make_default_names(-1)

    def test_is_square(self):
        assert is_square([[1, 0], [1, 1]])

        # 1D arrays do not pass the check
        assert not is_square([1, 2])

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
        A = list("asdf")

        B = list("fsda")
        assert np.allclose(
            [3, 1, 2, 0],
            get_index_of_other_array(A, B),
        )

    def test_reindex_array(self):
        A = list("asdf")
        B = list("fsda")
        C = list("fdas")

        assert np.all(reindex_array(B, get_index_of_other_array(A, B)) == A)

        assert np.all(reindex_array(A, get_index_of_other_array(B, A)) == B)

        assert np.all(reindex_array(C, get_index_of_other_array(A, C)) == A)

        assert np.all(reindex_array(A, get_index_of_other_array(C, A)) == C)

    def test_process_units(self):
        units = ["MiB", "kb", "GB", "Mib"]
        benchmarks = [1 / 8388608, 1 / 1000, 1 / 8e09, 1 / 1048576]

        for unit, benchmark in zip(units, benchmarks):
            assert np.allclose(benchmark, process_units(unit))

        with pytest.raises(ValueError):
            process_units("asdf")

        with pytest.raises(ValueError):
            process_units("BiB")

    def test_find_diff_weights(self):
        stencil = [-3, -2, -1, 0, 1]
        benchmark = [1, -4, 6, -4, 1]
        assert np.allclose(
            benchmark,
            find_diff_weights(stencil, 4),
        )

        stencil_d1_a3_forward = [0, 1, 2, 3]
        benchmark = [-11 / 6, 3, -3 / 2, 1 / 3]

        assert np.allclose(benchmark, find_diff_weights(stencil_d1_a3_forward, 1))

        stencil_d1_a4_center = [-2, -1, 0, 1, 2]
        benchmark = np.array([1, -8, 0, 8, -1]) / 12

        assert np.allclose(benchmark, find_diff_weights(stencil_d1_a4_center, 1))

        stencil_d1_a3_backward = [-3, -2, -1, 0]
        benchmark = np.array([-1 / 3, 3 / 2, -3, 11 / 6])

        assert np.allclose(benchmark, find_diff_weights(stencil_d1_a3_backward))

        # number of stencil points smaller than the order of the derivative
        # requested
        with pytest.raises(ValueError):
            find_diff_weights([-1, 1], order=2)

    def test_math_mode(self):
        """
        Tests for the `math_mode` utility function
        """
        item = "a"
        assert math_mode(item) == f"${item}$"

        arr = ("a", "b", "c")
        result = [f"${_}$" for _ in arr]
        assert math_mode(arr) == result

        assert math_mode(list(arr)) == result

        assert math_mode(np.array(arr)) == result

        with pytest.raises(TypeError):
            math_mode(1)
