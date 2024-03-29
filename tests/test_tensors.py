"""
Tests of the matrix submodule
"""

from __future__ import annotations

# standard library imports
import json
from pathlib import Path

# third-party imports
import numpy as np
import pytest
import sympy
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
from fitk.tensors import FisherMatrix, _jacobian, _process_fisher_mapping
from fitk.utilities import (
    MismatchingSizeError,
    MismatchingValuesError,
    P,
    ParameterNotFoundError,
    math_mode,
)

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"
DATADIR_OUTPUT = Path(__file__).resolve().parent / "data_output"


def test_transformation():
    r"""
    Test for the function `_jacobian`
    """
    names = ["omega_m", "omega_b", "h"]
    h = 0.67
    fiducials = [0.3 * h**2, 0.05 * h**2, h]
    transformation = {"omega_m": "Omega_m * h**2", "omega_b": "Omega_b * h **2"}
    result = _jacobian(dict(zip(names, fiducials)), transformation)


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
        FisherMatrix(np.loadtxt(DATADIR_INPUT / "test_numpy_matrix.dat", comments="#"))
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
            np.loadtxt(DATADIR_INPUT / "test_numpy_matrix.dat", comments="#"),
            names=names,
            latex_names=latex_names,
            fiducials=fiducials,
        )
        fm.to_file(
            DATADIR_OUTPUT / "test_numpy_matrix.json",
            metadata={
                "comment": "Fisher matrix forecast from Euclid IST:F paper",
                "url_arxiv": "https://arxiv.org/abs/1910.09273",
                "url_github": "https://github.com/euclidist-forecasting/fisher_for_public/blob/94bdfd09b26e4bed3654c0b95f4a2fb1f0cb192e/All_Results/optimistic/flat/EuclidISTF_WL_w0wa_flat_optimistic.txt",
            },
        )
        fm_read = FisherMatrix.from_file(DATADIR_OUTPUT / "test_numpy_matrix.json")
        assert fm == fm_read

    def test_from_file(self):
        fm = FisherMatrix.from_file(DATADIR_INPUT / "test_matrix.json")
        assert fm == FisherMatrix(
            np.diag([2, 1, 3]),
            names=list("bac"),
            latex_names=[r"\mathcal{B}", r"\mathcal{A}", r"\mathcal{C}"],
            fiducials=[5, 4, 6],
        )

    def test_from_dict(self):
        with open(
            DATADIR_INPUT / "test_matrix.json", "r", encoding="utf-8"
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
        m2_with_p = m1.rename({"a": P(name="x", latex_name=None, fiducial=1)})

        assert m2 == FisherMatrix(
            m1.values,
            names=list("xbc"),
            latex_names=list("xbc"),
            fiducials=[1, 0, 0],
        )

        assert m2 == m2_with_p

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
            m.sort(key=1)

        # special case: `is_iterable` should catch the `NotImplementedError`
        # thrown by `iter(FisherMatrix)`
        with pytest.raises(TypeError):
            m.sort(key=m)

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

        # is an iterable, but doesn't have the correct names
        with pytest.raises(ValueError):
            m.sort(key=["a", "b", "c"])

    def test_eq(self):
        assert FisherMatrix(
            np.diag([1, 2]),
            names=list("ba"),
        ) == FisherMatrix(
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

    def test_reparametrize_symbolic(self):
        """
        Check that the SymPy computation of the Jacobian works
        """
        m = FisherMatrix(
            np.diag([1, 2, 3, 10]),
            names=["omega_a", "omega_b", "c", "q"],
            fiducials=[0.21, 0.035, 0.7, 2],
        )

        new_names = ["Omega_a", "Omega_b", "c", "q"]

        new_fiducials = [
            m.fiducial("omega_a") / m.fiducial("c") ** 2,
            m.fiducial("omega_b") / m.fiducial("c") ** 2,
            m.fiducial("c"),
            m.fiducial("q"),
        ]

        jacobian = np.array(
            [
                [
                    m.fiducial("c") ** 2,
                    0,
                    2 * m.fiducial("omega_a") / m.fiducial("c"),
                    0,
                ],
                [
                    0,
                    m.fiducial("c") ** 2,
                    2 * m.fiducial("omega_b") / m.fiducial("c"),
                    0,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        benchmark = m.reparametrize(
            jacobian,
            names=new_names,
            fiducials=new_fiducials,
        )

        result = m.reparametrize_symbolic(
            {
                "omega_b": "Omega_b * c**2",
                "omega_a": "Omega_a * c**2",
                "c": "c",
            },
            latex_names={"Omega_b": r"$\Omega_b$"},
        ).sort(key=benchmark.names)

        assert np.all(benchmark.names == result.names)

        assert np.allclose(benchmark.fiducials, result.fiducials)

        assert result == benchmark

    def test_reparametrize_symbolic_exceptions(self):
        """
        Test that various exceptions are raised in invalid scenarios
        """
        fm = FisherMatrix(
            np.diag([1, 2, 3, 10]),
            names=["omega_a", "omega_b", "c", "q"],
            fiducials=[0.21, 0.035, 0.7, 2],
        )

        with pytest.raises(ParameterNotFoundError):
            fm.reparametrize_symbolic({"Omega_a": "asdf"})

        with pytest.raises(sympy.SympifyError):
            fm.reparametrize_symbolic({"omega_a": "3x /// z"})

        with pytest.raises(ValueError):
            fm.reparametrize_symbolic({"omega_a": "q * e", "c": "omega_a * b"})

        with pytest.raises(ValueError):
            fm.reparametrize_symbolic({"omega_b": "a * b"})

    def test_reparametrize_symbolic_nsolve(self):
        """
        Check that `sympy.nsolve` works when finding the new fiducials
        """
        fm = FisherMatrix(
            np.diag([1, 2, 3]),
            names=["omega_m", "b", "c"],
            fiducials=[0.3, 1.2, 0.1],
        )

        fm_new = fm.reparametrize_symbolic({"omega_m": "x * exp(x) * sin(x)"})

        assert np.allclose(fm_new.fiducial("x"), 0.445677)

        # same thing, but with `initial_guess` set
        fm_new = fm.reparametrize_symbolic(
            {"omega_m": "x * exp(x) * sin(x)"},
            initial_guess={"x": 0.4},
        )

        assert np.allclose(fm_new.fiducial("x"), 0.445677)

    def test_reparametrize_symbolic_invalid(self):
        """
        Check that we properly handle non-existing and complex solutions
        """
        fm = FisherMatrix(
            np.diag([1, 2, 3]),
            names=["omega_m", "b", "c"],
            fiducials=[0.3, 1.2, 0.1],
        )

        # no solution
        with pytest.raises(ValueError):
            fm.reparametrize_symbolic({"omega_m": "-sqrt(x)"})

        # complex solution
        # NOTE this raises a `UserWarning`, but later raises an exception, so
        # we're good
        with pytest.raises(TypeError), pytest.warns(UserWarning):
            fm.reparametrize_symbolic({"omega_m": "-x**2"})

    def test_reparametrize_symbolic_multiple_solutions(self):
        """
        Check that `solution_index` keyword argument works
        """
        fm = FisherMatrix(
            np.diag([1, 2, 3]),
            names=["omega_m", "b", "c"],
            fiducials=[0.3, 1.2, 0.1],
        )

        with pytest.warns(UserWarning):
            fm_new = fm.reparametrize_symbolic({"omega_m": "x**2"})

        # should not raise a warning if we specify an index
        fm_new = fm.reparametrize_symbolic({"omega_m": "x**2"}, solution_index=-1)

    def test_reparametrize_symbolic_identity(self):
        """
        Check that calling `reparametrize_symbolic` with the parameters being
        just renamed outputs the same result as calling `rename`
        """
        fm = FisherMatrix(
            np.diag([1, 2, 3]),
            names=["a", "b", "c"],
            fiducials=[4, 5, 6],
        )

        mapping = {"a": "x", "b": "y"}

        assert fm.rename(mapping) == fm.reparametrize_symbolic(mapping)

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

    def test_inverse(self):
        fm = FisherMatrix([[1, -1], [-1, 1]])

        # singular matrix
        with pytest.raises(np.linalg.LinAlgError):
            fm.inverse()

        # using the pseudoinverse, we do not encounter any errors
        fm.inverse(use_pinv=True)

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

    def test_fiducial(self):
        """
        Test for the `fiducial` method
        """
        fm = FisherMatrix(np.diag([1, 2]), names=["a", "b"], fiducials=[2, 3])

        assert np.allclose(fm.fiducial("b"), 3)

        with pytest.raises(ParameterNotFoundError):
            fm.fiducial("asdf")

        with pytest.raises(ParameterNotFoundError):
            fm.set_fiducial("q", 10)

        fm.set_fiducial("a", 6)

        assert np.allclose(fm.fiducials, [6, 3])

    def test_latex_name(self):
        """
        Test for the `latex_name` method
        """
        fm = FisherMatrix(
            np.diag([1, 2]),
            names=["a", "b"],
            latex_names=[r"$\mathbf{A}$", r"$\mathbf{B}$"],
        )

        assert fm.latex_name("b") == r"$\mathbf{B}$"

        with pytest.raises(ParameterNotFoundError):
            fm.latex_name("asdf")

        with pytest.raises(ParameterNotFoundError):
            fm.set_latex_name("asdf", "y")

        fm.set_latex_name("b", "q")

        assert np.all(fm.latex_names == np.array([r"$\mathbf{A}$", "q"]))

    def test_eigenvectors(self):
        fm = FisherMatrix(np.diag([1, 2, 3]))
        assert np.allclose(fm.eigenvectors(), np.linalg.eigh(fm.values)[-1])

    def test_from_parameters(self):
        fm = FisherMatrix.from_parameters(
            np.diag([1, 2, 3]),
            P("a"),
            P("b", latex_name=r"$\mathrm{B}$", fiducial=-3),
            P("c", fiducial=1),
        )

        assert all(fm.names == ["a", "b", "c"])
        assert np.allclose(fm.fiducials, [0, -3, 1])
        assert all(fm.latex_names == ["a", r"$\mathrm{B}$", "c"])

        with pytest.raises(MismatchingSizeError):
            fm = FisherMatrix.from_parameters(
                np.diag([1, 2]),
                P("a"),
            )
