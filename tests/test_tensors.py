"""
Tests of the matrix submodule
"""

from __future__ import annotations

# standard library imports
import json
import os
from itertools import product

# third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cosmicfish_pylib.fisher_derived import fisher_derived as CFFisherDerived
from cosmicfish_pylib.fisher_matrix import fisher_matrix as CFFisherMatrix
from cosmicfish_pylib.fisher_operations import (
    eliminate_parameters,
    marginalise,
    marginalise_over,
    reshuffle,
)
from scipy.stats import ortho_group

# first party imports
from fitk.tensors import FisherMatrix, _process_fisher_mapping
from fitk.utilities import (
    MismatchingSizeError,
    MismatchingValuesError,
    ParameterNotFoundError,
    math_mode,
)

DATADIR_INPUT = os.path.join(os.path.dirname(__file__), "data_input")
DATADIR_OUTPUT = os.path.join(os.path.dirname(__file__), "data_output")


class TestFisherMatrix:
    """
    Tests for the Fisher object.
    """

    def test_process_fisher_mapping(self):
        value = dict(name="p", latex_name="$p$", fiducial=-1)
        _process_fisher_mapping(value) == value

        value = dict(latex_name="$p$", fiducial=-1)
        with pytest.raises(ValueError):
            _process_fisher_mapping(value)

    def test_init(self):
        # a file
        FisherMatrix(
            np.loadtxt(
                os.path.join(DATADIR_INPUT, "test_numpy_matrix.dat"), comments="#"
            )
        )
        # a 2D array
        FisherMatrix(np.diag([1, 2, 3]))

        # cannot use a string literal as names
        with pytest.raises(TypeError):
            FisherMatrix(np.diag([1, 2, 3]), names="123")

        # non-positive diagonal
        assert not FisherMatrix(np.diag([-1, 2, 3])).is_valid()

        # non-square matrix
        with pytest.raises(ValueError):
            FisherMatrix(np.array([[1, 2], [3, 4], [5, 6]]))

        # not a matrix
        with pytest.raises(ValueError):
            FisherMatrix(np.array([[[1], [2], [3]]]))

        # not a symmetric matrix
        assert not FisherMatrix(np.array([[1, 0], [1, 1]])).is_valid()

        # get the LaTeX names right
        assert np.all(
            FisherMatrix(np.diag([1, 2, 3]), names=["a", "b", "c"]).latex_names
            == ["a", "b", "c"]
        )

        # mismatching sizes
        with pytest.raises(MismatchingSizeError):
            FisherMatrix(np.diag([1, 2]), names=["a", "b", "c"])

    def test_iter(self):
        """
        Test iteration of the `FisherMatrix` class
        """
        fm = FisherMatrix(np.diag([1, 2]))

        with pytest.raises(NotImplementedError):
            iter(fm)

        with pytest.raises(NotImplementedError):
            list(fm)

    def test_setters(self):
        """
        Test setters for `names`, `values`, `fiducials`, and `latex_names`.
        """
        fm = FisherMatrix(np.diag([1, 2, 3]))

        with pytest.raises(MismatchingSizeError):
            fm.names = ["a", "b"]

        fm.names = ["a", "b", "c"]

        with pytest.raises(MismatchingSizeError):
            fm.latex_names = ["$a$", "$b$"]

        fm.latex_names = math_mode(fm.names)

        with pytest.raises(MismatchingSizeError):
            fm.fiducials = [1, 2]

        with pytest.raises(TypeError):
            fm.fiducials = [1, "a", 2]

        fm.fiducials = [1, 2, 3]

    def test_to_dict(self):
        """
        Check we get the correct representation of the Fisher object as a
        dictionary
        """
        fm = FisherMatrix(np.diag([1, 2]))
        assert fm.to_dict() == dict(
            values=fm.values,
            names=fm.names,
            latex_names=fm.latex_names,
            fiducials=fm.fiducials,
        )

    def test_to_file(self):
        names = "Omegam Omegab w0 wa h ns sigma8 aIA etaIA betaIA".split(" ")
        # taken from v2 of https://arxiv.org/abs/1910.09273, table 1, and page 23
        fiducials = [0.32, 0.05, -1, 0, 0.67, 0.96, 0.816, 1.72, -0.41, 2.17]
        latex_names = [
            r"$\Omega_\mathrm{m}$",
            r"$\Omega_\mathrm{b}$",
            "$w_0$",
            "$w_a$",
            "$h$",
            "$n_s$",
            r"$\sigma_8$",
            r"$\mathcal{A}_\mathrm{IA}$",
            r"$\eta_\mathrm{IA}$",
            r"$\beta_\mathrm{IA}$",
        ]
        fm = FisherMatrix(
            np.loadtxt(
                os.path.join(DATADIR_INPUT, "test_numpy_matrix.dat"), comments="#"
            ),
            names=names,
            latex_names=latex_names,
            fiducials=fiducials,
        )
        fm.to_file(
            os.path.join(DATADIR_OUTPUT, "test_numpy_matrix.json"),
            metadata={
                "comment": "Fisher matrix forecast from Euclid IST:F paper",
                "url_arxiv": "https://arxiv.org/abs/1910.09273",
                "url_github": "https://github.com/euclidist-forecasting/fisher_for_public/blob/94bdfd09b26e4bed3654c0b95f4a2fb1f0cb192e/All_Results/optimistic/flat/EuclidISTF_WL_w0wa_flat_optimistic.txt",
            },
        )
        fm_read = FisherMatrix.from_file(
            os.path.join(DATADIR_OUTPUT, "test_numpy_matrix.json")
        )
        assert fm == fm_read

    def test_from_file(self):
        fm = FisherMatrix.from_file(os.path.join(DATADIR_INPUT, "test_matrix.json"))
        assert fm == FisherMatrix(
            np.diag([2, 1, 3]),
            names=list("bac"),
            latex_names=[r"\mathcal{B}", r"\mathcal{A}", r"\mathcal{C}"],
            fiducials=[5, 4, 6],
        )

    def test_from_dict(self):
        with open(
            os.path.join(DATADIR_INPUT, "test_matrix.json"), "r", encoding="utf-8"
        ) as file_handle:
            data = json.loads(file_handle.read())

        fm = FisherMatrix.from_dict(data)
        assert fm == FisherMatrix(
            np.diag([2, 1, 3]),
            names=list("bac"),
            latex_names=[r"\mathcal{B}", r"\mathcal{A}", r"\mathcal{C}"],
            fiducials=[5, 4, 6],
        )

    def test_rename(self):
        m1 = FisherMatrix(np.diag([1, 2, 3]), names=list("abc"))
        m2 = m1.rename({"a": dict(name="x", latex_name=None, fiducial=1)})

        assert m2 == FisherMatrix(
            m1.values,
            names=list("xbc"),
            latex_names=list("xbc"),
            fiducials=[1, 0, 0],
        )

        # the new names are not unique
        with pytest.raises(MismatchingSizeError):
            m1.rename({"a": "q", "b": "q"})

        # duplicate parameter
        with pytest.raises(ValueError):
            m2 = m1.rename({"a": "b"})

        # parameter to rename doesn't exist
        with pytest.raises(ValueError):
            m2 = m1.rename({"x": "y"})

        # this should work since we explicitly turned off the checker
        m2 = m1.rename({"x": "y", "b": "b"}, ignore_errors=True)
        assert m2 == m1

        m2 = m1.rename({"x": "y", "b": "d"}, ignore_errors=True)
        assert m2 != m1
        assert not np.all(m2.names == m1.names)

        m = FisherMatrix(np.diag([1, 2, 3]))
        assert m.rename(
            {
                "p1": "a",
                "p2": dict(name="b", latex_name="$b$", fiducial=2),
            },
        ) == FisherMatrix(
            m1.values,
            names=["a", "b", "p3"],
            fiducials=[0, 2, 0],
            latex_names=["a", "$b$", "p3"],
        )

    def test_getitem(self):
        data = FisherMatrix(np.diag([1, 2, 3]))
        assert data["p1", "p1"] == 1
        assert data["p1", "p2"] == 0
        assert data["p2", "p2"] == 2
        assert data["p3", "p3"] == 3
        assert np.all(np.array(["p1", "p2", "p3"]) == data.names)

        # check slicing works
        assert data[:2] == data.drop(data.names[-1])
        assert data[:1] == data.drop(data.names[0], invert=True)

        # check we aren't passing too many values
        with pytest.raises(ValueError):
            data["p1", "p2", "p3"]

        # same type, keys don't exist
        with pytest.raises(ParameterNotFoundError):
            data["p0", "p0"]

        # wrong specifier
        with pytest.raises(ParameterNotFoundError):
            data["p0"]

        # wrong type of key
        with pytest.raises(TypeError):
            data[1]

        # wrong type, keys don't exist
        with pytest.raises(ParameterNotFoundError):
            data[1, 1]

    def test_setitem(self):
        data = FisherMatrix(np.diag([1, 2, 3]))

        # assignment should succeed
        # NOTE the matrix should still be positive semi-definite
        data["p1", "p2"] = 1

        # matrix should remain symmetric
        assert np.allclose(np.transpose(data.values), data.values)

        data["p1", "p1"] = -1
        assert not data.is_valid()

        # trying to set a nonexisting element
        with pytest.raises(ParameterNotFoundError):
            data["p0", "p0"] = 1

        # wrong type of value to assign
        with pytest.raises(TypeError):
            data["p1", "p1"] = np.array([1, 2])

        # wrong number of keys (specific for strings)
        with pytest.raises(MismatchingSizeError):
            data["p1"] = 1

        # wrong number of keys
        with pytest.raises(MismatchingSizeError):
            data["p1", "p2", "p3"] = 1

        # the key is not an iterable
        with pytest.raises(TypeError):
            data[1] = 0

    def test_matrix(self):
        data = FisherMatrix(np.diag([1, 2, 3]))
        assert np.allclose(data.matrix, data.values)

    def test_values(self):
        data = FisherMatrix(np.diag([1, 2, 3]))
        data.values = [[1, 0, -1], [0, 2, 0], [-1, 0, 4]]

        # wrong dimensions
        with pytest.raises(MismatchingSizeError):
            data.values = np.diag([1, 2])

        # not square matrix
        # TODO maybe this should raise some other error?
        with pytest.raises(ValueError):
            data.values = [1, 2, 3]

    def test_is_diagonal(self):
        assert not FisherMatrix([[1, -1], [-1, 5]]).is_diagonal()
        assert FisherMatrix(np.diag([1, 2])).is_diagonal()

    def test_diagonal(self):
        fisher = FisherMatrix(np.diag([1, 2, 3]))
        assert np.allclose([1, 2, 3], fisher.diagonal())

    def test_determinant(self):
        fisher = FisherMatrix(np.diag([1, 2, 3]))
        assert np.allclose(fisher.determinant(), 6)

    def test_constraints(self):
        data = FisherMatrix([[1, -1], [-1, 1.2]])
        assert np.all(data.constraints(marginalized=True, sigma=2) >= 0)
        assert np.all(data.constraints(marginalized=False, sigma=2) >= 0)
        assert np.allclose(
            FisherMatrix(np.diag([1, 2, 3])).constraints(marginalized=True),
            FisherMatrix(np.diag([1, 2, 3])).constraints(marginalized=False),
        )

        assert np.allclose(
            data.constraints(p=0.682689),
            data.constraints(sigma=1),
        )

        data_cf = CFFisherMatrix(data.values)
        assert np.allclose(
            data_cf.get_confidence_bounds(confidence_level=0.682689),
            data.constraints(p=0.682689),
        )

        assert np.allclose(
            data_cf.get_confidence_bounds(confidence_level=0.95),
            data.constraints(p=0.95),
        )

        with pytest.raises(ValueError):
            data.constraints(sigma=-1)

        with pytest.raises(ValueError):
            data.constraints(sigma=1, p=0.3)

        with pytest.raises(ValueError):
            data.constraints(p=-2)

        with pytest.raises(ParameterNotFoundError):
            data.constraints(name="asdf")

    def test_sort(self):
        m = FisherMatrix(
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            fiducials=[-1, 0, 1],
            names=["p3", "p1", "p2"],
        )

        assert m.sort() == FisherMatrix(
            [[22, 23, 21], [32, 33, 31], [12, 13, 11]],
            fiducials=[0, 1, -1],
            names=["p1", "p2", "p3"],
            latex_names=["p1", "p2", "p3"],
        )

        assert m.sort(key=[2, 1, 0]) == FisherMatrix(
            [[33, 32, 31], [23, 22, 21], [13, 12, 11]],
            fiducials=[1, 0, -1],
            names=["p2", "p1", "p3"],
        )

        m = FisherMatrix(
            np.diag([1, 3, 5]),
            fiducials=[2, 0, 1],
            names=["p3", "p1", "p2"],
        )

        assert np.allclose(
            m.sort(key=["p1", "p3", "p2"]).values,
            reshuffle(
                CFFisherMatrix(m.values, param_names=m.names), ["p1", "p3", "p2"]
            ).get_fisher_matrix(),
        )

        # not one of special options, nor a callable
        with pytest.raises(TypeError):
            m.sort(key="not callable")

        assert m.sort(key="fiducials") == FisherMatrix(
            np.diag([3, 5, 1]), names=["p1", "p2", "p3"], fiducials=[0, 1, 2]
        )

        assert m.sort(key="fiducials", reversed=True) == FisherMatrix(
            np.diag([1, 5, 3]), names=["p3", "p2", "p1"], fiducials=[2, 1, 0]
        )

        fm = FisherMatrix(np.diag([1, 2, 3, 4]), names=["b1", "b2", "f1", "f2"])
        assert np.all(
            fm.sort(key=["f1", "b1", "f2", "b2"]).names
            == np.array(["f1", "b1", "f2", "b2"])
        )

    def test_eq(self):
        assert FisherMatrix(np.diag([1, 2]), names=list("ba"),) == FisherMatrix(
            np.diag([2, 1]),
            names=list("ab"),
        )

        assert (
            FisherMatrix(
                np.diag([4, 1, 2, 3]),
                names=list("xabc"),
            )
            == FisherMatrix(
                np.diag([1, 2, 3, 4]),
                names=list("abcx"),
            )
            == FisherMatrix(
                np.diag([3, 2, 4, 1]),
                names=list("cbxa"),
            )
        )

        assert (
            FisherMatrix(
                np.diag([2, 3, 1]),
                names=list("bca"),
            )
            == FisherMatrix(
                np.diag([1, 2, 3]),
                names=list("abc"),
            )
            == FisherMatrix(
                np.diag([3, 2, 1]),
                names=list("cba"),
            )
        )

    def test_trace(self):
        data = FisherMatrix(np.diag([1, 2, 3]))

        assert np.allclose(data.trace(), np.trace(data.values))

    def test_condition_number(self):
        data = FisherMatrix(np.diag([1, 2, 3]))

        assert np.allclose(data.condition_number(), 3)

    def test_drop(self):
        data = FisherMatrix(np.diag([1, 2, 3]), fiducials=[-1, 0, 1])
        data_new = data.drop("p1")

        # names to drop don't exist
        with pytest.raises(ValueError):
            data.drop("p1", "a", ignore_errors=False)

        # ignoring errors, we get the correct thing
        assert data.drop("p1", ignore_errors=False) == data.drop(
            "p1", "a", ignore_errors=True
        )

        assert np.allclose(data_new.values, np.diag([2, 3]))
        for old, new in zip(data_new.names, ["p2", "p3"]):
            assert old == new
        # assert np.all(data_new.names == np.array(['p2', 'p3']))
        assert np.allclose(data_new.fiducials, np.array([0, 1]))

        fm_cf = CFFisherMatrix(data.values)
        assert np.allclose(
            eliminate_parameters(fm_cf, ["p1"]).get_fisher_matrix(), data_new.values
        )

        data_new = data.drop("p2", "p1", invert=True)
        assert data_new == FisherMatrix(
            np.diag([2, 1]),
            names=["p2", "p1"],
            fiducials=[0, -1],
        )
        fm_cf = CFFisherMatrix(data.values)
        assert np.allclose(
            eliminate_parameters(fm_cf, ["p3"]).get_fisher_matrix(), data_new.values
        )

        with pytest.raises(ValueError):
            data_new = data.drop(*data.names)

    def test_representations(self):
        """
        Tests `__repr__`, `__str__`, and `_repr_html_`
        """
        m = FisherMatrix(np.diag([1, 2]))
        str(m)
        repr(m)
        m._repr_html_()

    def test_add(self):
        m1 = FisherMatrix(np.diag([1, 2, 3]))
        m2 = FisherMatrix(np.diag([6, 5, 4]))
        assert m1 + m2 == FisherMatrix(m1.values + m2.values)
        assert m1 + 3 == FisherMatrix(m1.values + 3)
        assert 3 + m1 == FisherMatrix(3 + m1.values)

        # mismatching fiducials
        with pytest.raises(MismatchingValuesError):
            m1 + FisherMatrix(m1.values, fiducials=[1, 2, 3])

        fm1 = FisherMatrix([[1, -2], [-2, 5]], names=["a", "b"])
        fm2 = FisherMatrix(np.diag([3, 4, 6]), names=["a", "c", "d"])
        result = fm1 + fm2

        assert result == FisherMatrix(
            [
                [4, -2, 0, 0],
                [-2, 5, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 6],
            ],
            names=["a", "b", "c", "d"],
            latex_names=["p1", "p2", "p3", "p4"],
        )

    def test_sub(self):
        m1 = FisherMatrix(np.diag([1, 2, 3]))
        assert m1 - m1 == FisherMatrix(m1.values - m1.values)
        assert m1 - 1 == FisherMatrix(m1.values - 1)
        assert 1 - m1 == FisherMatrix(1 - m1.values)

    def test_mul(self):
        """
        Test for multiplication by a number or FisherMatrix (elementwise).
        """
        m1 = FisherMatrix([[2, -1], [-1, 3]])

        assert m1 * 2 == FisherMatrix(m1.values * 2)
        assert 2 * m1 == FisherMatrix(m1.values * 2)

        r1 = FisherMatrix(m1.values) * FisherMatrix(1 / m1.values)
        r2 = FisherMatrix(m1.values * (1 / m1.values))
        assert r1 == r2

        with pytest.raises(MismatchingValuesError):
            m1 * FisherMatrix(np.diag([1, 2]), names=["a", "b"])

        with pytest.raises(MismatchingValuesError):
            m1 * FisherMatrix(np.diag([1, 2]), fiducials=[1, 2])

    def test_matmul(self):
        """
        Test for matrix multiplication.
        """
        m1 = FisherMatrix([[1, -2], [-2, 4]])
        m2 = FisherMatrix([[2, 0], [0, 3]])

        assert m1 @ m2 == FisherMatrix(m1.values @ m2.values)
        assert m2 @ m1 == FisherMatrix(m2.values @ m1.values)

        with pytest.raises(MismatchingValuesError):
            m1 @ FisherMatrix(np.diag([1, 2]), names=["a", "b"])

        with pytest.raises(MismatchingValuesError):
            m1 @ FisherMatrix(np.diag([1, 2]), fiducials=[1, 2])

    def test_truediv(self):
        """
        Test for division by a number or FisherMatrix (elementwise).
        """
        m = FisherMatrix([[2, -1], [-1, 3]])
        assert m / 2 == FisherMatrix(m.values / 2)
        assert m / m == FisherMatrix(np.ones((len(m), len(m))))
        # the other stuff should raise errors
        with pytest.raises(TypeError):
            2 / m

        with pytest.raises(MismatchingValuesError):
            m / FisherMatrix(np.diag([1, 2]), names=["a", "b"])

        with pytest.raises(MismatchingValuesError):
            m / FisherMatrix(np.diag([1, 2]), fiducials=[1, 2])

    def test_reparametrize(self):
        m = FisherMatrix([[2, -1], [-1, 3]])
        jacobian_m = np.array([[3, 2], [6, 7]])
        m_new = m.reparametrize(jacobian_m)
        assert np.allclose(
            m_new.values, np.transpose(jacobian_m) @ m.values @ jacobian_m
        )
        m2 = FisherMatrix(np.diag([1, 2]))
        jac = [[1, 4], [3, 2]]
        m2_new = m2.reparametrize(jac, names=["a", "b"])
        assert np.allclose(m2_new.values, np.transpose(jac) @ m2.values @ jac)

        with pytest.raises(MismatchingSizeError):
            m.reparametrize(jac, names=["a"])

        with pytest.raises(MismatchingSizeError):
            m.reparametrize(jac, names=["a", "b"], latex_names=["a"])

        with pytest.raises(MismatchingSizeError):
            m.reparametrize(jac, fiducials=[1])

    @pytest.mark.xfail(reason="CosmicFish fails for some reason")
    def test_reparametrize_cf(self):
        m2 = FisherMatrix(np.diag([1, 2]))
        jac = [[1, 4], [3, 2]]
        m2_new = m2.reparametrize(jac, names=["a", "b"])

        m_cf = CFFisherMatrix(fisher_matrix=m2.values, param_names=m2.names)

        m_cf_derived = CFFisherDerived(
            jac, param_names=m2.names, derived_param_names=["a", "b"]
        )
        m_cf_new = m_cf_derived.add_derived(m_cf)

        assert np.allclose(
            m_cf_new.get_fisher_matrix(),
            np.transpose(jac) @ m_cf.get_fisher_matrix() @ jac,
        )

        assert np.allclose(m2_new.values, m_cf_new.get_fisher_matrix())

    def test_marginalize_over(self):
        size = 5
        size_marg = 3
        ort = ortho_group.rvs(size, random_state=12345)
        rng = np.random.RandomState(12345)  # pylint: disable=no-member
        values = ort.T @ np.diag(rng.rand(size)) @ ort
        m = FisherMatrix(values)
        names_to_marginalize_over = [f"p{_ + 1}" for _ in range(size_marg)]
        m_marg = m.marginalize_over(*names_to_marginalize_over)
        assert m_marg == FisherMatrix(
            np.linalg.inv(np.linalg.inv(values)[size_marg:, size_marg:]),
            names=m.names[size_marg:],
            latex_names=m.latex_names[size_marg:],
            fiducials=m.fiducials[size_marg:],
        )

        m_cf = CFFisherMatrix(values)

        assert np.allclose(m.inverse(), m_cf.get_fisher_inverse())

        m_marg1 = m.marginalize_over(*names_to_marginalize_over, invert=True)
        m_marg2 = m.marginalize_over(*names_to_marginalize_over)
        m_cf_marginalized1 = marginalise(m_cf, names_to_marginalize_over)
        m_cf_marginalized2 = marginalise_over(m_cf, names_to_marginalize_over)

        assert np.allclose(m_cf_marginalized1.get_fisher_matrix(), m_marg1.values)
        assert np.allclose(m_cf_marginalized2.get_fisher_matrix(), m_marg2.values)

    def test_ufunc(self):
        """
        Test of numpy universal functions.
        """
        m = FisherMatrix([[2, -1], [-1, 3]])
        assert np.exp(m) == FisherMatrix(np.exp(m.values))

        m1 = FisherMatrix([[1, -2], [-2, 4]])
        m2 = FisherMatrix([[2, 0], [0, 3]])

        assert np.multiply(m1, m2) == m1 * m2
        assert np.multiply(3, m2) == 3 * m2
        assert np.sin(m2) == FisherMatrix(np.sin(m2.values))

        m3 = FisherMatrix(np.diag([1]))

        # we can't add Fisher matrices with mismatching length of names
        with pytest.raises(ValueError):
            np.add(m, m3)

        m4 = FisherMatrix(np.diag([1, 2]), names=["p2", "p1"])

        assert np.add(m, m4) == FisherMatrix([[4, -1], [-1, 4]], names=["p1", "p2"])

        assert np.power(m, 3) == m**3

    def test_correlation_matrix(self):
        fisher1 = FisherMatrix(np.diag([1, 2]))
        corr1 = fisher1.correlation_matrix()

        assert np.allclose(corr1[0, 0], 1)
        assert np.allclose(corr1[0, 1], 0)
        assert np.allclose(corr1[1, 0], 0)
        assert np.allclose(corr1[1, 1], 1)

        fisher2 = FisherMatrix([[1, 1 / 2], [1 / 2, 1]])
        corr2 = fisher2.correlation_matrix()

        assert np.allclose(corr2[0, 0], 1)
        assert np.allclose(corr2[0, 1], -1 / 2)
        assert np.allclose(corr2[1, 0], -1 / 2)
        assert np.allclose(corr2[1, 1], 1)

    def test_correlation_coefficient(self):
        fisher1 = FisherMatrix([[1, 1 / 2], [1 / 2, 1]])

        assert np.allclose(
            fisher1.correlation_coefficient(fisher1.names[0], fisher1.names[0]), 1
        )

        assert np.allclose(
            fisher1.correlation_coefficient(fisher1.names[0], fisher1.names[1]), -1 / 2
        )

    def test_figure_of_merit(self):
        fisher1 = FisherMatrix([[1, 1 / 2], [1 / 2, 1]])

        assert np.allclose(
            fisher1.figure_of_merit(), np.log(np.sqrt(np.linalg.det(fisher1.values)))
        )

    def test_figure_of_correlation(self):
        fisher1 = FisherMatrix([[1, 1 / 2], [1 / 2, 1]])

        assert np.allclose(
            fisher1.figure_of_correlation(),
            -np.log(np.sqrt(np.linalg.det(fisher1.correlation_matrix()))),
        )
