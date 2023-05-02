"""
Various tests for the `fitk` module.
"""

from __future__ import annotations

# standard library imports
from pathlib import Path

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cosmicfish_pylib.fisher_matrix import fisher_matrix as CFFisherMatrix
from cosmicfish_pylib.fisher_plot import CosmicFishPlotter as CFFisherPlotter
from cosmicfish_pylib.fisher_plot_analysis import (
    CosmicFish_FisherAnalysis as CFFisherAnalysis,
)
from matplotlib import ticker
from scipy.stats import norm

# first party imports
from fitk.graphics import (
    EmptyFigureError,
    FisherBarFigure,
    FisherFigure1D,
    FisherFigure2D,
    _add_plot_1d,
    _add_shading_1d,
    _get_ellipse,
    plot_curve_1d,
    plot_curve_2d,
)
from fitk.tensors import FisherMatrix
from fitk.utilities import (
    MismatchingSizeError,
    MismatchingValuesError,
    ParameterNotFoundError,
    get_default_rcparams,
)

DATADIR_INPUT = Path(__file__).resolve().parent / "data_input"
DATADIR_OUTPUT = Path(__file__).resolve().parent / "data_output"


class TestFisherFigure:
    """
    Tests the `FisherFigure1D`/`FisherFigure2D` class in various ways
    """

    def test_contour_levels(self):
        """
        Test to make sure the contour levels parsing works
        """
        # wrong format (no alpha specified)
        with pytest.raises(ValueError):
            FisherFigure1D(
                contour_levels=[1, 2, 3],
            )

        # wrong format (no alpha for second item)
        with pytest.raises(ValueError):
            FisherFigure1D(
                contour_levels=[(1, 2), 3],
            )

        # negative contour level
        with pytest.raises(ValueError):
            FisherFigure1D(
                contour_levels=[(-3, 2)],
            )

        # alpha outside the range (0, 1)
        with pytest.raises(ValueError):
            FisherFigure1D(
                contour_levels=[(3, -1)],
            )

        # correct specification
        FisherFigure1D(
            contour_levels=[(1, 0.2), (2, 0.1)],
        )

    def test_set_tick_params(self):
        ff = FisherFigure1D()

        ff.plot(FisherMatrix(np.diag([1, 2])))
        with pytest.raises(ValueError):
            ff.set_tick_params(which="asdf")


@pytest.fixture
def m1():
    names = list("abcde")
    latex_names = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", "d", "e"]

    val = np.diag([1, 2, 3, 9.3, 3])
    fid = [0, 0, 0, 1, 2]
    return FisherMatrix(
        val,
        names=names,
        fiducials=fid,
        latex_names=latex_names,
    )


@pytest.fixture
def m2():
    names = list("abcde")
    latex_names = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", "d", "e"]

    val = np.diag([6, 7, 20, 1.5, 0.6])
    fid = [-1, 0.1, 5, -1, 3]
    return FisherMatrix(
        val,
        names=names,
        fiducials=fid,
        latex_names=latex_names,
    )


@pytest.fixture
def m3():
    names = list("abcde")
    latex_names = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", "d", "e"]
    val = np.diag([10, 4.2, 6.4, 0.2, 0.342])
    fid = [0, 0, 0, 1, 2]
    return FisherMatrix(
        val,
        names=names,
        fiducials=fid,
        latex_names=latex_names,
    )


@pytest.fixture
def euclid_opt():
    return FisherMatrix.from_file(
        DATADIR_INPUT / "EuclidISTF_WL_w0wa_flat_optimistic.json"
    )


@pytest.fixture
def euclid_pes():
    return FisherMatrix.from_file(
        DATADIR_INPUT / "EuclidISTF_WL_w0wa_flat_pessimistic.json"
    )


def test_plot_2d_repr(m1):
    fp = FisherFigure2D()
    fp.plot(m1)

    repr(fp)
    str(fp)


def test_plot_2d_size():
    fp = FisherFigure2D()

    with pytest.raises(ValueError):
        m = FisherMatrix([[3]], names=["a"])
        fp.plot(m)


def test_plot_2d_names(m1, m2):
    fp = FisherFigure2D()
    fp.plot(m1)

    m = FisherMatrix(m2.values, names=m2.names, fiducials=m2.fiducials).rename(
        {m2.names[0]: "o"},
    )

    with pytest.raises(ValueError):
        fp.plot(m)


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d(m1, m2, m3):
    fp = FisherFigure1D()

    fp.plot(m1, label="first")
    fp.plot(m2, label="second", ls="--", mark_fiducials=True)
    fp.plot(
        m3,
        label="third",
        ls=":",
        color="red",
        mark_fiducials=dict(ls="--", color="black", lw=1),
    )

    fp.legend(ncol=3, bbox_to_anchor=[0.5, 1])
    fp.set_major_locator(ticker.MaxNLocator(3))
    fp.set_tick_params(which="x", rotation=45)

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_options(m1):
    fp = FisherFigure1D(options={})

    fp.plot(m1, label="first")
    fp.legend()

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_cols(m1, m2, m3):
    fp = FisherFigure1D(max_cols=3, hspace=0.8)

    fp.plot(m1, label="first")
    fp.plot(m2, label="second", ls="--")
    fp.plot(m3, label="third", ls=":", color="red")

    fp.legend(ncol=3, bbox_to_anchor=[0.5, 1])
    fp.set_major_locator(ticker.MaxNLocator(3))
    fp.set_tick_params(which="x", rotation=45)
    fp.set_title("Constrained columns")

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_add_artist_to_legend(m1, m2, m3):
    fp = FisherFigure1D(max_cols=3, hspace=0.8)

    fp.plot(m1, label="first")
    fp.plot(m2, label="second", ls="--")
    fp.plot(m3, label="third", ls=":", color="red")

    fp.legend(ncol=3, bbox_to_anchor=[0.5, 1])
    fp.set_major_locator(ticker.MaxNLocator(3))
    fp.set_tick_params(which="x", rotation=45)
    fp.set_title("Constrained columns")

    (handle,) = fp["a"].plot(
        np.linspace(-3, 3, 100),
        np.sin(np.linspace(-3, 3, 100)),
        color="green",
    )

    fp.add_artist_to_legend(handle, "sine function")

    fp.legend(ncol=4)

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_euclid(euclid_opt, euclid_pes):
    fp1 = FisherFigure1D(max_cols=5)

    fp1.plot(
        euclid_opt,
        label="optimistic case",
    )
    fp1.plot(
        euclid_pes,
        ls="--",
        label="pessimistic case",
    )

    # accessing an element and plotting some stuff on it
    fp1.draw(
        "h",
        "vlines",
        0.67,
        0,
        norm.pdf(
            0,
            loc=0,
            scale=euclid_pes.constraints("h", marginalized=True),
        ),
        color="blue",
        label="blue line",
    )

    fp1.legend(ncol=3)
    fp1.set_major_locator(ticker.MaxNLocator(3))
    fp1.set_tick_params(which="x", fontsize=10)

    # non-existing parameter
    with pytest.raises(ValueError):
        fp1["asdf"]

    return fp1.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_locator(euclid_opt, euclid_pes):
    fp2 = FisherFigure1D(max_cols=5)

    fp2.plot(
        euclid_opt.marginalize_over(
            "Omegam",
            "Omegab",
            "h",
            "ns",
            invert=True,
        ),
        label="optimistic case",
    )
    fp2.plot(
        euclid_pes.marginalize_over(
            "Omegam",
            "Omegab",
            "h",
            "ns",
            invert=True,
        ),
        label="pessimistic case",
        ls="--",
        color="red",
    )

    fp2.legend(ncol=2)
    fp2.set_major_locator(ticker.MaxNLocator(3))

    return fp2.figure


def test_plot_2d_getitem(euclid_opt):
    fp = FisherFigure2D()

    fp.plot(
        euclid_opt.marginalize_over(
            "Omegam",
            "Omegab",
            "h",
            "ns",
            invert=True,
        ),
        label="optimistic case",
    )

    with pytest.raises(TypeError):
        fp[1]


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_euclid3(euclid_opt):
    # add just one element
    fp3 = FisherFigure1D()
    fp3.plot(
        euclid_opt.marginalize_over("Omegam", invert=True),
        label=r"Parameters = $\Omega_\mathrm{m}$",
    )

    fp3.legend(
        overwrite=True,
        bbox_to_anchor=(0.1, 1.0),
    )

    fp3.set_label_params(fontsize=30)

    return fp3.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_euclid(euclid_opt, euclid_pes):
    fm_opt, fm_pes = euclid_opt, euclid_pes

    i = 4

    fm_optimistic = fm_opt.drop(
        *fm_opt.names[:i],
        invert=True,
    )

    fm_pessimistic = fm_pes.drop(
        *fm_pes.names[:i],
        invert=True,
    )

    fp = FisherFigure2D(
        show_1d_curves=True,
        show_joint_dist=True,
    )

    fp.plot(
        fm_pessimistic,
        label="pessimistic",
        mark_fiducials=True,
    )

    fm_shifted = fm_pessimistic
    rng = np.random.default_rng(2021)
    fm_shifted.fiducials = fm_shifted.fiducials * (
        1 + rng.uniform(-0.05, 0.05, len(fm_shifted))
    )
    fm_shifted.values *= 2

    fp.plot(
        fm_shifted,
        ls=":",
        color="green",
        label="shifted",
        mark_fiducials=dict(linestyle="-.", linewidth=1, color="orange"),
    )

    fp.plot(
        fm_optimistic,
        ls="--",
        color="red",
        label="optimistic",
    )
    fp.legend(fontsize=20)
    fp.set_label_params(fontsize=30)
    fp.set_tick_params(fontsize=8)
    fp.set_major_locator(ticker.MaxNLocator(3))
    fp.set_minor_locator(ticker.MaxNLocator(5))
    fp.set_major_formatter(ticker.ScalarFormatter())
    fp.set_tick_params(which="x", rotation=45)

    fp.savefig(DATADIR_OUTPUT / "test_plot_2d_euclid.pdf")

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_legend(euclid_opt):
    # add just one element
    fp = FisherFigure2D(
        options={"style": "ggplot"},
    )
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
        label="Euclid",
    )

    fp.set_label_params(fontsize=30)

    fp.legend()

    fp.legend()

    fp.legend(
        overwrite=True,
        bbox_to_anchor=(0.2, 1),
    )

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_contours(euclid_opt):
    # add just one element
    fp = FisherFigure2D(
        contour_levels_1d=[(1, 0.3)],
        contour_levels_2d=[(1, 0.2)],
        show_joint_dist=True,
    )
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
        label="Euclid",
    )

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_mark_fiducials(euclid_opt):
    # add just one element
    fp = FisherFigure2D(
        show_joint_dist=True,
        show_1d_curves=True,
    )
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
        label="Euclid",
        mark_fiducials=True,
    )

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_set_title(euclid_opt):
    # add just one element
    fp = FisherFigure2D()
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
        label="Euclid",
    )

    fp.set_label_params(fontsize=30)

    fp.set_title("My title", overwrite=True)
    fp.set_title(r"$\mathit{Sample}$ survey")

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_set_title2(euclid_opt):
    fp = FisherFigure2D(show_1d_curves=True)
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
        label="Euclid",
    )

    fp.set_label_params(fontsize=30)

    fp.set_title("My title", overwrite=True)
    fp.set_title(r"$\mathit{Sample}$ survey")

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_options_file(euclid_opt):
    # add just one element
    fp = FisherFigure2D(
        options={"style": DATADIR_INPUT / "ggplot.mplstyle"},
    )
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
    )

    fp.set_label_params(fontsize=30)

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d_draw(euclid_opt):
    # add just one element
    fp = FisherFigure2D(show_1d_curves=True)
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
    )

    fp.draw(
        "Omegam",
        "Omegam",
        "plot",
        np.linspace(0.3, 0.4, 100),
        100 * np.sin(100 * np.linspace(0.3, 0.4, 100)),
        ls="--",
        label="sine function",
    )

    fp.draw(
        "Omegam",
        "Omegam",
        "arrow",
        euclid_opt.fiducials[np.where(euclid_opt.names == "Omegam")][0],
        0,
        0.05,
        100,
        label="arrow",
    )

    fp.legend()

    with pytest.raises(AttributeError):
        fp.draw("Omegam", "Omegam", "asdf", 10, 0, 10, label="nothing")

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_draw(euclid_opt):
    # add just one element
    fp = FisherFigure1D()
    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
    )

    x = np.linspace(0.3, 0.4, 100)

    fp.draw(
        "Omegam",
        "plot",
        x,
        100 * np.sin(100 * x),
        label="sine function",
    )

    fp.draw(
        "Omegam",
        "plot",
        x,
        100 * np.sin(100 * x),
        x,
        100 * np.cos(100 * x),
        ls=":",
        label="mutiple plots",
    )

    # doing it manually
    (line,) = fp["Omegam"].plot(
        x,
        100 * np.sin(80 * x),
    )

    fp.legend(
        [line],
        ["sample line"],
        overwrite=True,
        bbox_transform=fp.figure.transFigure,
    )

    with pytest.raises(AttributeError):
        fp.draw("Omegam", "asdf", 10, 0, 10, label="nothing")

    return fp.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_2d():
    fm = FisherMatrix([[0.3, -0.5], [-0.5, 0.9]])
    fp = FisherFigure2D()

    fm_cf = CFFisherMatrix(fm.values)
    fl_cf = CFFisherAnalysis()

    fl_cf.add_fisher_matrix(fm_cf)

    fp_cf = CFFisherPlotter(fishers=fl_cf)

    fp_cf.new_plot()
    fp_cf.plot_tri(D2_alphas=[0.1, 0], D2_filled=[True, False], D1_norm_prob=True)
    ffigure = fp_cf.figure

    ax = fp_cf.figure.axes[0]
    ffigure, _, __ = plot_curve_1d(fm, fm.names[0], ax=ax)
    ax.set_ylim(0, 0.1)

    ax = fp_cf.figure.axes[2]
    ffigure, _, __ = plot_curve_1d(fm, fm.names[1], ax=ax)
    ax.set_ylim(0, 0.2)

    ax = fp_cf.figure.axes[1]

    ffigure, _, __ = plot_curve_2d(fm, fm.names[0], fm.names[-1], ax=ax)
    ffigure, _, __ = plot_curve_2d(
        fm,
        fm.names[0],
        fm.names[-1],
        ax=_,
        scaling_factor=2,
    )

    fp = FisherFigure2D()

    fp.plot(fm)
    fp.axes.flat[0].set_xlim(-14, 14)
    fp.axes.flat[2].set_ylim(-8, 8)
    fp.axes.flat[-1].set_xlim(-8, 8)

    # both should return the same thing
    assert fp[fp.names[1], fp.names[0]] == fp[fp.names[0], fp.names[1]]

    # non-existing parameters
    with pytest.raises(ParameterNotFoundError):
        fp["asdf", "qwerty"]
    with pytest.raises(ParameterNotFoundError):
        fp[fp.names[0], "qwerty"]

    return ffigure


def test_plot_2d_one_element():
    fp = FisherFigure2D()

    # cannot make a plot with one element
    with pytest.raises(ValueError):
        fp.plot(FisherMatrix(np.diag([1])))


def test_plot_1d_figure_error(euclid_opt):
    """
    Check that we can't draw anything before calling `plot`
    """
    fp = FisherFigure1D()

    with pytest.raises(EmptyFigureError):
        fp.legend()

    with pytest.raises(EmptyFigureError):
        fp.draw("omegam", "asdf", 1, 1)

    with pytest.raises(EmptyFigureError):
        fp.set_major_locator(ticker.MaxNLocator(3))

    with pytest.raises(EmptyFigureError):
        fp.set_minor_locator(ticker.MaxNLocator(3))

    with pytest.raises(EmptyFigureError):
        fp.set_major_formatter(ticker.ScalarFormatter())

    with pytest.raises(EmptyFigureError):
        fp.set_minor_formatter(ticker.ScalarFormatter())

    with pytest.raises(EmptyFigureError):
        fp.set_title("title")

    with pytest.raises(EmptyFigureError):
        fp.set_label_params(fontsize=30)

    with pytest.raises(EmptyFigureError):
        fp.set_tick_params(fontsize=30)

    with pytest.raises(EmptyFigureError):
        fp.savefig("path.pdf")

    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
    )


def test_plot_2d_figure_error(euclid_opt):
    """
    Check that we can't draw anything before calling `plot`
    """
    fp = FisherFigure2D()

    with pytest.raises(EmptyFigureError):
        fp.legend()

    with pytest.raises(EmptyFigureError):
        fp.draw("omegam", "omegab", "asdf", 1, 1)

    with pytest.raises(EmptyFigureError):
        fp.set_major_locator(ticker.MaxNLocator(3))

    with pytest.raises(EmptyFigureError):
        fp.set_minor_locator(ticker.MaxNLocator(3))

    with pytest.raises(EmptyFigureError):
        fp.set_major_formatter(ticker.ScalarFormatter())

    with pytest.raises(EmptyFigureError):
        fp.set_minor_formatter(ticker.ScalarFormatter())

    with pytest.raises(EmptyFigureError):
        fp.set_title("title")

    with pytest.raises(EmptyFigureError):
        fp.set_label_params(fontsize=30)

    with pytest.raises(EmptyFigureError):
        fp.set_tick_params(fontsize=30)

    with pytest.raises(EmptyFigureError):
        fp.savefig("path.pdf")

    fp.plot(
        euclid_opt.marginalize_over("Omegam", "Omegab", invert=True),
    )


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_curve_1d(m1):
    fig, ax, _ = plot_curve_1d(m1, m1.names[0])

    return fig


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_curve_2d(m1):
    fig, ax, _ = plot_curve_2d(m1, m1.names[0], m1.names[-1])

    ax.autoscale()
    ax.relim()
    ax.autoscale_view()

    return fig


def test_ellipse(m1):
    with pytest.raises(ValueError):
        _get_ellipse(m1, "omegam", "omegam")


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_absolute_constraints(euclid_opt, euclid_pes):
    euclid_opt = euclid_opt[:7]
    euclid_pes = euclid_pes[:7]
    euclid1 = euclid_pes / 5
    euclid2 = 3 * euclid_pes
    fig = FisherBarFigure()

    fig.plot_absolute_constraints(
        [euclid_opt, euclid_pes, euclid1, euclid2],
        "bar",
    )

    # kwarg `space` outside of allowed range
    with pytest.raises(ValueError):
        fig.plot_absolute_constraints(
            [euclid_opt],
            "bar",
            space=1.3,
        )

    # mismatching number of parameters in the Fisher matrices
    with pytest.raises(MismatchingValuesError):
        fig.plot_absolute_constraints(
            [euclid_opt, euclid_pes.drop(euclid_pes.names[0])],
            "bar",
        )

    # mismatching number of labels
    with pytest.raises(MismatchingSizeError):
        fig.plot_absolute_constraints(
            [euclid_opt, euclid_pes],
            "bar",
            labels=["first"],
        )

    # mismatching number of colors
    with pytest.raises(MismatchingSizeError):
        fig.plot_absolute_constraints(
            [euclid_opt, euclid_pes],
            "bar",
            colors=["blue"],
        )

    # invalid scale
    with pytest.raises(ValueError):
        fig.plot_absolute_constraints(
            [euclid_opt, euclid_pes],
            "bar",
            scale="asdf",
        )

    return fig.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_absolute_constraints_barh():
    fm1 = FisherMatrix(np.diag([1, 2, 3]), names=["a", "b", "c"], fiducials=[3, 2, 1])
    fm2 = FisherMatrix(
        np.diag([4, 5, 6]), names=["a", "b", "c"], fiducials=[3.5, 2.5, 1.5]
    )
    fig = FisherBarFigure()

    fig.plot_absolute_constraints(
        [fm1, fm2],
        kind="barh",
        labels=["first", "second"],
    )
    with plt.rc_context(get_default_rcparams()):
        fig.figure.legend(bbox_to_anchor=(0.5, 0.87), loc="lower center")

    return fig.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_relative_constraints(euclid_opt, euclid_pes):
    euclid_opt = euclid_opt[:7]
    euclid_pes = euclid_pes[:7]
    euclid1 = euclid_pes / 5
    euclid2 = 3 * euclid_pes
    fig = FisherBarFigure()

    fig.plot_relative_constraints(
        [euclid_opt, euclid_pes, euclid1, euclid2],
        "bar",
    )

    # kwarg `scale` outside of allowed range
    with pytest.raises(ValueError):
        fig.plot_relative_constraints(
            [euclid_opt],
            "bar",
            space=1.3,
        )

    # not the same parameters
    with pytest.raises(MismatchingValuesError):
        temp = euclid_pes[:]
        temp.names[0] = "asdf"
        fig.plot_relative_constraints(
            [euclid_opt, temp],
            "bar",
        )

    # mismatching number of labels
    with pytest.raises(MismatchingSizeError):
        fig.plot_relative_constraints(
            [euclid_opt, euclid_pes],
            "bar",
            labels=["first"],
        )

    # mismatching number of colors
    with pytest.raises(MismatchingSizeError):
        fig.plot_relative_constraints(
            [euclid_opt, euclid_pes],
            "bar",
            colors=["blue"],
        )

    # invalid scale
    with pytest.raises(ValueError):
        fig.plot_relative_constraints(
            [euclid_opt, euclid_pes],
            "bar",
            scale="asdf",
        )

    return fig.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_relative_constraints_barh():
    fm1 = FisherMatrix(np.diag([1, 2, 3]), names=["a", "b", "c"], fiducials=[3, 2, 1])
    fm2 = FisherMatrix(
        np.diag([4, 5, 6]), names=["a", "b", "c"], fiducials=[3.5, 2.5, 1.5]
    )
    fig = FisherBarFigure()

    fig.plot_relative_constraints(
        [fm1, fm2],
        kind="barh",
        labels=["first", "second"],
    )
    with plt.rc_context(get_default_rcparams()):
        fig.figure.legend(bbox_to_anchor=(0.5, 0.87), loc="lower center")

    return fig.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_relative_constraints_percent(euclid_opt, euclid_pes):
    euclid_opt = euclid_opt[:7]
    euclid_pes = euclid_pes[:7]
    euclid1 = euclid_pes / 5
    euclid2 = 3 * euclid_pes
    fig = FisherBarFigure()

    fig.plot_relative_constraints(
        [euclid_opt, euclid_pes, euclid1, euclid2],
        "bar",
        percent=True,
    )

    return fig.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_add_plot_1d(euclid_opt):
    fig, ax, handle = _add_plot_1d(
        fiducial=euclid_opt.fiducial(euclid_opt.names[0]) * (1 + 0.5),
        sigma=euclid_opt.constraints(name=euclid_opt.names[0])[0] * 0.5,
    )

    return fig


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_add_shading_1d(euclid_opt):
    fig, ax = _add_shading_1d(
        fiducial=euclid_opt.fiducial(euclid_opt.names[0]) * (1 + 0.5),
        sigma=euclid_opt.constraints(name=euclid_opt.names[0])[0] * 0.5,
    )

    return fig


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_absolute_constraints_ordering():
    fm1 = FisherMatrix([[1, 0], [0, 3]], names=["a", "b"], fiducials=[1, 2])
    fm2 = FisherMatrix([[3.1, 0], [0, 0.9]], names=["b", "a"], fiducials=[2, 1])
    ff = FisherBarFigure()
    ff.plot_absolute_constraints([fm1, fm2], kind="bar")

    return ff.figure


@pytest.mark.mpl_image_compare(
    tolerance=20,
    savefig_kwargs=dict(dpi=300),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_relative_constraints_ordering():
    fm1 = FisherMatrix([[1, 0], [0, 3]], names=["a", "b"], fiducials=[1, 2])
    fm2 = FisherMatrix([[3.1, 0], [0, 0.9]], names=["b", "a"], fiducials=[2, 1])
    ff = FisherBarFigure()
    ff.plot_relative_constraints([fm1, fm2], kind="bar")

    return ff.figure
