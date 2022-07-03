"""
Various tests for the `fitk` module.
"""

from __future__ import annotations

# standard library imports
import os
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
from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_plotter import (
    FisherFigure1D,
    FisherFigure2D,
    plot_curve_1d,
    plot_curve_2d,
)
from fitk.fisher_utils import ParameterNotFoundError

DATADIR_INPUT = Path(os.path.join(os.path.dirname(__file__), "data_input"))
DATADIR_OUTPUT = Path(os.path.join(os.path.dirname(__file__), "data_output"))


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
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d(m1, m2, m3):
    fp = FisherFigure1D()

    fp.plot(m1, label="first")
    fp.plot(m2, label="second", ls="--")
    fp.plot(m3, label="third", ls=":", color="red")

    fp.legend(ncol=3, bbox_to_anchor=[0.5, 1])
    fp.set_major_locator(ticker.MaxNLocator(3))
    fp.set_tick_params(which="x", rotation=45)

    return fp.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
    baseline_dir=DATADIR_INPUT,
    style="default",
)
def test_plot_1d_euclid3(euclid_opt):
    # add just one element
    fp3 = FisherFigure1D()
    fp3.plot(
        euclid_opt.marginalize_over("Omegam", invert=True),
    )

    fp3.set_label_params(fontsize=30)

    return fp3.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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
    )

    fp.plot(
        fm_optimistic,
        ls="--",
        color="red",
        label="optimistic",
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
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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

    return fp.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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

    fp.legend()

    with pytest.raises(AttributeError):
        fp.draw("Omegam", "Omegam", "asdf", 10, 0, 10, label="nothing")

    return fp.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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

    fp.legend([line], ["sample line"], overwrite=True)

    with pytest.raises(AttributeError):
        fp.draw("Omegam", "asdf", 10, 0, 10, label="nothing")

    return fp.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
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
