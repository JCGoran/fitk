"""
Various tests for the `fitk` module.
"""

from __future__ import annotations

# standard library imports
import json
import os
from itertools import product

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
from matplotlib.backends.backend_pdf import PdfPages
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

DATADIR_INPUT = os.path.join(os.path.dirname(__file__), "data_input")
DATADIR_OUTPUT = os.path.join(os.path.dirname(__file__), "data_output")


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
        fp = FisherFigure2D()

        with pytest.raises(ValueError):
            m1 = FisherMatrix([[3]], names=["a"])
            m2 = FisherMatrix([[5]], names=["b"])
            FisherFigure2D().plot(m1).plot(m2)

    def test_plot_1d(self):
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
        fp.savefig(os.path.join(DATADIR_OUTPUT, "test_plot_1d.pdf"))

    def test_plot_1d_euclid(self):
        fm_optimistic = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_optimistic.json")
        )
        fm_pessimistic = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_pessimistic.json")
        )

        fp1 = FisherFigure1D(max_cols=5)

        fp1.plot(
            fm_optimistic,
            label="optimistic case",
        )
        fp1.plot(
            fm_pessimistic,
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
                0, loc=0, scale=fm_pessimistic.constraints("h", marginalized=True)
            ),
            color="blue",
            label="blue line",
        )

        fp1.legend()

        # non-existing parameter
        with pytest.raises(ValueError):
            fp1["asdf"]

        fp2 = FisherFigure1D(max_cols=5)
        fp2.plot(
            fm_optimistic.marginalize_over("Omegam", "Omegab", "h", "ns", invert=True),
            label="optimistic case",
        )
        fp2.plot(
            fm_pessimistic.marginalize_over("Omegam", "Omegab", "h", "ns", invert=True),
            label="pessimistic case",
            ls="--",
            color="red",
        )

        fp2.legend(ncol=2)

        # add just one element
        fp3 = FisherFigure1D()
        fp3.plot(
            fm_optimistic.marginalize_over("Omegam", invert=True),
        )

        fp3.set_label_params(fontsize=30)
        fp3.set_tick_params(rotation=30)

        with PdfPages(os.path.join(DATADIR_OUTPUT, "test_plot_1d_euclid.pdf")) as pdf:
            for ff in [fp1, fp2, fp3]:
                pdf.savefig(ff.figure, bbox_inches="tight")

    def test_plot_2d_euclid(self):
        fm_opt = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_optimistic.json")
        )
        fm_pes = FisherMatrix.from_file(
            os.path.join(DATADIR_INPUT, "EuclidISTF_WL_w0wa_flat_pessimistic.json")
        )

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
        fm_shifted.fiducials = fm_shifted.fiducials * (
            1 + np.random.uniform(-0.05, 0.05, len(fm_shifted))
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
        fp.set_tick_params(fontsize=12, which="y")

        fp.savefig(os.path.join(DATADIR_OUTPUT, "test_plot_2d_euclid.pdf"))

    def test_plot_2d(self):
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

        fp.savefig(os.path.join(DATADIR_OUTPUT, "test_plot_triangle_mine.pdf"))

        ffigure.savefig(os.path.join(DATADIR_OUTPUT, "test_plot_triangle.pdf"))

    def test_plot_2d_continuation(self):
        fm = FisherMatrix(np.diag([1]))
        fp = FisherFigure2D()

        with pytest.raises(ValueError):
            fp.plot(fm)
