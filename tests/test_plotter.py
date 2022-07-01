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


def test_init():
    names = list("abcde")
    latex_names = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", "d", "e"]
    val1 = np.diag([1, 2, 3, 9.3, 3])
    val2 = np.diag([6, 7, 20, 1.5, 0.6])
    fid1 = [0, 0, 0, 1, 2]
    fid2 = [-1, 0.1, 5, -1, 3]
    m1 = FisherMatrix(val1, names=names, fiducials=fid1, latex_names=latex_names)
    m2 = FisherMatrix(val2, names=names, fiducials=fid2, latex_names=latex_names)
    fp = FisherFigure2D()

    with pytest.raises(ValueError):
        m1 = FisherMatrix([[3]], names=["a"])
        m2 = FisherMatrix([[5]], names=["b"])
        fp.plot(m1)
        fp.plot(m2)


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
    baseline_dir=DATADIR_INPUT,
)
def test_plot_1d():
    names = list("abcde")
    latex_names = [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{C}$", "d", "e"]
    val1 = np.diag([1, 2, 3, 9.3, 3])
    val2 = np.diag([6, 7, 20, 1.5, 0.6])
    val3 = np.diag([10, 4.2, 6.4, 0.2, 0.342])
    fid1 = [0, 0, 0, 1, 2]
    fid2 = [-1, 0.1, 5, -1, 3]
    fid3 = fid1
    m1 = FisherMatrix(
        val1,
        names=names,
        fiducials=fid1,
        latex_names=latex_names,
    )
    m2 = FisherMatrix(
        val2,
        names=names,
        fiducials=fid2,
        latex_names=latex_names,
    )
    m3 = FisherMatrix(
        val3,
        names=names,
        fiducials=fid3,
        latex_names=latex_names,
    )
    fp = FisherFigure1D()

    fp.plot(m1, label="first")
    fp.plot(m2, label="second", ls="--")
    fp.plot(m3, label="third", ls=":", color="red")

    fp.legend(ncol=3, bbox_to_anchor=[0.5, 1])
    fp.set_major_locator(ticker.MaxNLocator(3))
    fp.set_tick_params(which="x", rotation=45)

    return fp.figure


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


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
    baseline_dir=DATADIR_INPUT,
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
        norm.pdf(0, loc=0, scale=euclid_pes.constraints("h", marginalized=True)),
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
)
def test_plot_1d_euclid2(euclid_opt, euclid_pes):
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
    fp2.set_tick_params(which="x", rotation=45)

    return fp2.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
    baseline_dir=DATADIR_INPUT,
)
def test_plot_1d_euclid3(euclid_opt):
    # add just one element
    fp3 = FisherFigure1D()
    fp3.plot(
        euclid_opt.marginalize_over("Omegam", invert=True),
    )

    fp3.set_label_params(fontsize=30)
    fp3.set_tick_params(rotation=30)

    return fp3.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
    baseline_dir=DATADIR_INPUT,
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
    fp.set_tick_params(rotation=45, which="x")
    fp.set_major_locator(ticker.MaxNLocator(5))
    fp.set_tick_params(fontsize=7, which="y")

    return fp.figure


@pytest.mark.mpl_image_compare(
    savefig_kwargs=dict(dpi=300, bbox_inches="tight"),
    baseline_dir=DATADIR_INPUT,
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


def test_plot_2d_continuation():
    fp = FisherFigure2D()

    with pytest.raises(ValueError):
        fp.plot(FisherMatrix(np.diag([1])))
