"""
Various tests for the `fitk` module.
"""

import os

import numpy as np
import pytest
from cosmicfish_pylib.fisher_matrix import fisher_matrix as CFFisherMatrix
from cosmicfish_pylib.fisher_operations import (
    eliminate_parameters,
    information_gain,
    marginalise,
    marginalise_over,
    reshuffle,
)
from fitk import FisherMatrix, FisherPlotter, bayes_factor, kl_divergence
from fitk.fisher_utils import (
    ParameterNotFoundError,
    float_to_latex,
    get_index_of_other_array,
    is_iterable,
    is_positive_semidefinite,
    is_square,
    is_symmetric,
    make_default_names,
    process_units,
    reindex_array,
)
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, ortho_group

DATADIR_INPUT = os.path.join(os.path.dirname(__file__), "data_input")
DATADIR_OUTPUT = os.path.join(os.path.dirname(__file__), "data_output")


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


class TestFisherOperations:
    """
    Tests for any functions operating on multiple Fisher objects
    """

    def test_bayes_factor(self):
        fisher_base = FisherMatrix(np.diag([1, 2, 3]))
        fisher_extended = FisherMatrix(np.diag([1, 2, 3, 4, 5]))
        priors = [1, 1]
        offsets = [0, 0, 0]

        assert np.allclose(
            0.03558812717085886,
            bayes_factor(fisher_base, fisher_extended, priors=priors, offsets=offsets),
        )

    def test_kl_divergence(self):
        fisher1 = FisherMatrix(np.diag([1, 2, 3]))
        fisher2 = FisherMatrix(np.diag([4, 5, 6]))
        fisher_prior = FisherMatrix(np.diag([1, 1, 1]))

        cf1 = CFFisherMatrix(fisher1.values)
        cf2 = CFFisherMatrix(fisher2.values)
        cf_prior = CFFisherMatrix(fisher_prior.values)

        assert np.allclose(
            information_gain(cf1, cf2, cf_prior, stat=False),
            kl_divergence(fisher1, fisher2, fisher_prior),
        )


class TestFisherTensor:
    """
    Tests for the Fisher object.
    """

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

    def test_rename(self):
        m1 = FisherMatrix(np.diag([1, 2, 3]), names=list("abc"))
        m2 = m1.rename({"a": dict(name="x", latex_name=None, fiducial=1)})

        assert m2 == FisherMatrix(
            m1.values,
            names=list("xbc"),
            latex_names=list("xbc"),
            fiducials=[1, 0, 0],
        )

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

    @pytest.mark.skip(reason="Implementation needs to be fixed")
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

    def test_sort(self):
        m = FisherMatrix(
            [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            fiducials=[-1, 0, 1],
            names=["p3", "p1", "p2"],
        )

        assert m.sort() == FisherMatrix(
            [[33, 31, 32], [13, 11, 12], [23, 21, 22]],
            fiducials=[1, -1, 0],
            names=["p1", "p2", "p3"],
            latex_names=["p1", "p2", "p3"],
        )

        assert m.sort(key=[2, 1, 0]) == FisherMatrix(
            [[33, 32, 31], [23, 22, 21], [13, 12, 11]],
            fiducials=[1, 0, -1],
            names=["p2", "p1", "p3"],
        )

        m = FisherMatrix(
            [[1, 0, 0], [0, 3, 0], [0, 0, 5]],
            fiducials=[-1, 0, 1],
            names=["p3", "p1", "p2"],
        )

        assert np.allclose(
            m.sort(key=["p1", "p3", "p2"]).values,
            reshuffle(
                CFFisherMatrix(m.values, param_names=m.names), ["p1", "p3", "p2"]
            ).get_fisher_matrix(),
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

    def test_drop(self):
        data = FisherMatrix(np.diag([1, 2, 3]), fiducials=[-1, 0, 1])
        data_new = data.drop("p1")

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

    def test_add(self):
        m1 = FisherMatrix(np.diag([1, 2, 3]))
        m2 = FisherMatrix(np.diag([6, 5, 4]))
        assert m1 + m2 == FisherMatrix(m1.values + m2.values)
        assert m1 + 3 == FisherMatrix(m1.values + 3)
        assert 3 + m1 == FisherMatrix(3 + m1.values)

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

    def test_matmul(self):
        """
        Test for matrix multiplication.
        """
        m1 = FisherMatrix([[1, -2], [-2, 4]])
        m2 = FisherMatrix([[2, 0], [0, 3]])

        assert m1 @ m2 == FisherMatrix(m1.values @ m2.values)
        assert m2 @ m1 == FisherMatrix(m2.values @ m1.values)

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


class TestFisherPlotter:
    def test_init(self):
        names = list("abcde")
        latex_names = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", "d", "e"]
        val1 = np.diag([1, 2, 3, 9.3, 3])
        val2 = np.diag([6, 7, 20, 1.5, 0.6])
        fid1 = [0, 0, 0, 1, 2]
        fid2 = [-1, 0.1, 5, -1, 3]
        m1 = FisherMatrix(val1, names=names, fiducials=fid1, latex_names=latex_names)
        m2 = FisherMatrix(val2, names=names, fiducials=fid2, latex_names=latex_names)
        fp = FisherPlotter(m1, m2, labels=["first", "second"])

        assert fp.values[0] == m1
        assert fp.values[1] == m2

        with pytest.raises(ValueError):
            m1 = FisherMatrix([[3]], names=["a"])
            m2 = FisherMatrix([[5]], names=["b"])
            FisherPlotter(m1, m2)

    def test_plot_1d(self):
        names = list("abcde")
        latex_names = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", "d", "e"]
        val1 = np.diag([1, 2, 3, 9.3, 3])
        val2 = np.diag([6, 7, 20, 1.5, 0.6])
        val3 = np.diag([10, 4.2, 6.4, 0.2, 0.342])
        fid1 = [0, 0, 0, 1, 2]
        fid2 = [-1, 0.1, 5, -1, 3]
        fid3 = fid1
        m1 = FisherMatrix(val1, names=names, fiducials=fid1, latex_names=latex_names)
        m2 = FisherMatrix(val2, names=names, fiducials=fid2, latex_names=latex_names)
        m3 = FisherMatrix(val3, names=names, fiducials=fid3, latex_names=latex_names)
        fp = FisherPlotter(m1, m2, m3, labels=["first", "second", "third"])

        ffigure = fp.plot_1d(
            legend=True,
            title=True,
            max_cols=1,
        )

        ffigure["a"].plot(
            np.linspace(-2, 2, 100),
            [0.5 for _ in np.linspace(-2, 2, 100)],
            ls="--",
            label="another line",
        )

        ffigure["a"].legend()
        ffigure.savefig(os.path.join(DATADIR_OUTPUT, "test_plot_1d.pdf"))

    def test_plot_1d_euclid(self):
        fm_optimistic = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_optimistic.json")
        )
        fm_pessimistic = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_pessimistic.json")
        )

        fp1 = FisherPlotter(
            fm_optimistic,
            fm_pessimistic,
            labels=["optimistic case", "pessimistic case"],
        )

        ff1 = fp1.plot_1d(
            legend=True,
            title=r"Forecast for $\mathit{Euclid}$ IST:F, $w_0,w_a$ cosmology",
            max_cols=5,
        )

        # accessing an element and plotting some stuff on it
        ff1["h"].vlines(
            0.67,
            0,
            norm.pdf(
                0, loc=0, scale=fm_pessimistic.constraints("h", marginalized=True)
            ),
            color="blue",
        )

        # non-existing parameter
        with pytest.raises(ValueError):
            ff1["asdf"]

        fp2 = FisherPlotter(
            fm_optimistic.marginalize_over("Omegam", "Omegab", "h", "ns", invert=True),
            fm_pessimistic.marginalize_over("Omegam", "Omegab", "h", "ns", invert=True),
            labels=["optimistic case", "pessimistic case"],
        )

        ff2 = fp2.plot_1d(
            legend=True,
            title=r"Forecast for $\mathit{Euclid}$ IST:F, $w_0,w_a$ cosmology",
            max_cols=5,
        )

        # generate another figure from the same data, with different layout and title
        ff2_2 = fp2.plot_1d(
            legend=True,
            title=r"Forecast for $\mathit{Euclid}$ IST:F, $w_0,w_a$ cosmology (take two)",
            max_cols=2,
        )

        # add just one element
        fp3 = FisherPlotter(
            fm_optimistic.marginalize_over("Omegam", invert=True),
            fm_pessimistic.marginalize_over("Omegam", invert=True),
        )

        ff3 = fp3.plot_1d()

        with PdfPages(os.path.join(DATADIR_OUTPUT, "test_plot_1d_euclid.pdf")) as pdf:
            for ff in [ff1, ff2, ff2_2, ff3]:
                pdf.savefig(ff.figure, bbox_inches="tight")

    def test_plot_2d_euclid(self):
        fm_optimistic = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_optimistic.json")
        )
        fm_pessimistic = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_pessimistic.json")
        )
        fm_optimistic = fm_optimistic.drop("Omegam", "Omegab", invert=True)
        fm_pessimistic = fm_pessimistic.drop("Omegam", "Omegab", invert=True)

        fp = FisherPlotter(
            fm_pessimistic,
            fm_optimistic,
            labels=["pessimistic case", "optimistic case"],
        )

        ffigure = fp.plot_triangle(
            #legend=True,
            #title=r'Forecast for $\mathit{Euclid}$ IST:F, $w_0,w_a$ cosmology',
            plot_gaussians=True,
        )

        ffigure.savefig(os.path.join(DATADIR_OUTPUT, "test_plot_triangle_euclid.pdf"))

    def test_plot_2d(self):
        fm1 = FisherMatrix([[0.3, -0.5], [-0.5, 0.9]])
        fp = FisherPlotter(fm1)

        ffigure = fp.plot_triangle(
            # legend=True,
            # title=r'Forecast for $\mathit{Euclid}$ IST:F, $w_0,w_a$ cosmology',
            plot_gaussians=True,
        )

        ffigure.savefig(os.path.join(DATADIR_OUTPUT, "test_plot_triangle.pdf"))
