"""
Package for plotting of Fisher objects.
See here for documentation of `FisherPlotter`, `FisherFigure1D`, and `FisherFigure2D`.
"""

from __future__ import annotations

# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Optional, Tuple, Union

import matplotlib.pyplot as plt

# third party imports
import numpy as np
from cycler import cycler

# first party imports
from fitk.fisher_matrix import FisherMatrix

# first party imports
from fitk.fisher_utils import (
    MismatchingSizeError,
    ParameterNotFoundError,
    get_default_rcparams,
    get_index_of_other_array,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.ticker import LinearLocator, StrMethodFormatter
from matplotlib.transforms import Bbox
from scipy.stats import chi2, norm


class FisherBaseFigure(ABC):
    def __init__(
        self,
        figure: Figure,
        axes: Collection[Axes],
        names: Collection[str],
    ):
        """
        Constructor.

        Parameters
        ----------
        figure : Figure
            the figure plotted by `FisherPlotter`

        axes : Collection[Axes]
            the axes of the above figure

        names : Collection[str]
            the names of the parameters that are plotted
        """
        self._figure = figure
        self._axes = axes
        self._names = names

    @abstractmethod
    def __getitem__(self, key):
        """
        Implements element access.
        """
        pass

    @property
    def figure(self):
        """
        Returns the underlying figure, an instance of `matplotlib.figure.Figure`.
        """
        return self._figure

    @property
    def axes(self):
        """
        Returns the axes of the figure as a numpy array.
        """
        return self._axes

    @property
    def names(self):
        """
        Returns the names of the parameters plotted.
        """
        return self._names

    def savefig(
        self,
        path: Path,
        dpi: float = 300,
        bbox_inches: Union[str, Bbox] = "tight",
        **kwargs,
    ):
        """
        Convenience wrapper for `figure.savefig`.

        Parameters
        ----------
        path : Path
            the path where to save the figure

        dpi : float = 300
            the resolution of the saved figure

        bbox_inches : Union[str, Bbox] = "tight"
            what is the bounding box for the figure

        kwargs
            any other keyword arguments that should be passed to
            `figure.savefig`
        """
        return self.figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


class FisherFigure1D(FisherBaseFigure):
    """
    Container for easy access to elements in the 1D plot.
    """

    def __getitem__(
        self,
        key: str,
    ):
        """
        Returns the axis associated to the name `key`.
        """
        if key not in self.names:
            raise ParameterNotFoundError(key, self.names)

        return self.axes.flat[np.where(self.names == key)][0]


class FisherFigure2D(FisherBaseFigure):
    """
    Container for easy access to elements in the 2D plot.
    """

    def __getitem__(
        self,
        key: Tuple[str, str],
    ):
        if not isinstance(key, tuple):
            raise TypeError(
                f"Incompatible type for element access: expected {type(tuple)}, got {type(key)}"
            )

        name1, name2 = key

        if name1 not in self.names:
            raise ParameterNotFoundError(name1, self.names)
        if name2 not in self.names:
            raise ParameterNotFoundError(name2, self.names)

        index1 = np.where(self.names == name1)
        index2 = np.where(self.names == name2)

        if index1 < index2:
            index1, index2 = index2, index1

        return self.axes[index1, index2][0, 0]


class FisherPlotter:
    """
    Class for plotting FisherMatrix objects.
    """

    def __init__(
        self,
        *args: FisherMatrix,
        labels: Optional[Collection[str]] = None,
        ylabel1d: str = r"$p (\theta)$",
    ):
        """
        Constructor.

        Parameters
        ----------
        args : FisherMatrix
            `FisherMatrix` objects which we want to plot.
            Can have different fiducial values.
            The order of plotting of the parameters and the LaTeX labels to use
            are determined by the first argument.

        labels : Optional[Collection[str]] = None
            the list of labels to put in the legend of the plots.
            If not set, defaults to `0, ..., len(args) - 1`

        Raises
        ------
        * `MismatchingSizeError` if the sizes of the inputs are not equal
        * `ValueError` if the names of the inputs do not match

        Examples
        --------
        Create a Fisher plotter with two objects:
        >>> fm1 = FisherMatrix(np.diag([1, 2, 3]), names=list('abc'))
        >>> fm2 = FisherMatrix(np.diag([4, 5, 6]), names=list('abc'))
        >>> fp = FisherPlotter(fm1, fm2, labels=['first', 'second'])
        """
        # make sure all of the Fisher objects have the same sizes
        if not all(len(args[0]) == len(arg) for arg in args):
            raise MismatchingSizeError(*args)

        # check names
        if not all(set(args[0].names) == set(arg.names) for arg in args):
            raise ValueError("The names of the inputs do not match")

        if labels is not None:
            if len(labels) != len(args):
                raise MismatchingSizeError(labels, args)
        # default labels are 0, ..., n - 1
        else:
            labels = list(map(str, np.arange(len(args))))

        # in case the names are shuffled, we sort them according to the FIRST
        # input
        indices = np.array(
            [get_index_of_other_array(args[0].names, arg.names) for arg in args],
            dtype=object,
        )

        self._values = [arg.sort(key=index) for index, arg in zip(indices, args)]

        self._labels = np.array(labels, dtype=object)

        self._ylabel1d = ylabel1d

    @property
    def values(self):
        """
        Returns the input array of Fisher objects.
        """
        return self._values

    @property
    def labels(self):
        """
        Returns the labels of the Fisher objects.
        """
        return self._labels

    @property
    def ylabel1d(self):
        """
        Returns the y-label used for labelling the 1D curves.
        """
        return self._ylabel1d

    @ylabel1d.setter
    def ylabel1d(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Expected `str` for `ylabel1d`, got {type(value)}")
        self._ylabel1d = value

    def plot_1d(
        self,
        scale: float = 1.0,
        max_cols: Optional[int] = None,
        mpl_options: Optional[dict] = None,
        **kwargs,
    ):
        """
        Makes a 1D plot (Gaussians) of the Fisher objects

        Parameters
        ----------
        max_cols : Optional[int] = None
            the maximum number of columns to force the plot into.
            By default, the parameters are always plotted horizontally; if you
            need to spread it over `max_cols`, pass a non-negative integer
            here.

        mpl_options : Optional[dict] = None
            any parameters meant for `matplotlib.rcParams`. By default, only
            sets default font to cm serif.
            See [Matplotlib documentation](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
            for more information.

        Returns
        -------
        An instance of `FisherFigure1D`.
        """
        size = len(self.values[0])

        if max_cols is not None and max_cols <= size:
            full = size % max_cols == 0
            layout = size // max_cols if full else size // max_cols + 1, max_cols
        else:
            layout = 1, size
            full = True

        if not mpl_options:
            mpl_options = get_default_rcparams()

        with plt.rc_context(mpl_options):
            # general figure setup
            fig = plt.figure(
                clear=True,
                figsize=(scale * 2 * layout[1], scale * 2 * layout[0]),
            )
            gs = fig.add_gridspec(
                nrows=layout[0],
                ncols=layout[1],
                hspace=0.5,
                wspace=0.1,
            )
            axes = gs.subplots()
            if size == 1:
                axes = np.array([axes])

            names = self.values[0].names
            latex_names = self.values[0].latex_names

            for (index, name), latex_name in zip(enumerate(names), latex_names):
                ax = axes.flat[index]

                for c, fm in zip(get_default_cycler(), self.values):
                    plot_curve_1d(fm, name, ax)

                    add_shading_1d(
                        fm.fiducials[np.where(fm.names == name)],
                        fm.constraints(name, marginalized=True),
                        ax,
                        color=c["color"],
                        alpha=0.3,
                        ec=None,
                    )

                    add_shading_1d(
                        fm.fiducials[np.where(fm.names == name)],
                        fm.constraints(name, marginalized=True),
                        ax,
                        level=2,
                        color=c["color"],
                        alpha=0.1,
                        ec=None,
                    )

                ax.set_xlabel(latex_name)
                ax.relim()
                ax.autoscale_view()

                # the y axis should start at 0 since we're plotting a PDF
                ax.set_ylim(0, ax.get_ylim()[-1])

                if index == 0:
                    ax.set_ylabel(self.ylabel1d)

                ax.set_yticks([])

            if kwargs.get("legend") is True:
                # remove any axes which are not drawn
                if not full:
                    for index in range(
                        (layout[0] - 1) * layout[1] + 1, layout[0] * layout[1]
                    ):
                        axes.flat[index].remove()

            # remove any axes which are not drawn
            if not full:
                for index in range(
                    (layout[0] - 1) * layout[1] + 1, layout[0] * layout[1]
                ):
                    axes.flat[index].remove()

        return FisherFigure1D(fig, axes, names)

    def plot_triangle(
        self,
        mpl_options: Optional[dict] = None,
        plot_1d_curves: bool = True,
        **kwargs,
    ):
        """
        Plots the 2D contours (and optionally 1D curves) of the Fisher objects.

        Parameters
        ----------
        mpl_options : Optional[dict] = None
            any parameters meant for `matplotlib.rcParams`. By default, only
            sets default font to cm serif.
            See [Matplotlib documentation](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
            for more information.
        plot_1d_curves : bool = True
            whether or not the 1D (marginalized) curves should be plotted

        Returns
        -------
        An instance of `FisherFigure2D`.
        """
        size = len(self.values[0])

        if size < 2:
            raise ValueError("Unable to make a 2D plot with < 2 parameters")

        if not mpl_options:
            mpl_options = get_default_rcparams()

        with plt.rc_context(mpl_options):
            # general figure setup
            fig = plt.figure(figsize=(2 * size, 2 * size), clear=True)
            gs = fig.add_gridspec(nrows=size, ncols=size, hspace=0.2, wspace=0.2)
            # TODO make it work with a shared xcol
            ax = gs.subplots(sharex="col", sharey=False)

            names = self.values[0].names
            latex_names = self.values[0].latex_names

            # set automatic limits
            for i in range(size):
                for j in range(size):
                    if i == j:
                        ax[i, i].set_yticks([])
                        ax[i, i].set_yticklabels([])
                    if i > 0 and 0 < j < size - 1:
                        ax[i, j].set_yticks([])
                        ax[i, j].set_yticklabels([])

            for (i, namey), latex_namey in zip(enumerate(names), latex_names):
                for (j, namex), latex_namex in zip(enumerate(names), latex_names):
                    # labels for 2D contours (increasing y)
                    if i > 0 and j == 0:
                        ax[i, j].set_ylabel(latex_namey)

                    # labels for 2D contours (increasing x)
                    if i == size - 1:
                        ax[i, j].set_xlabel(latex_namex)

                    # removing any unnecessary axes from the gridspec
                    # TODO should they be removed, or somehow just made invisible?
                    if i < j:
                        ax[i, j].remove()
                        # ax[i, j].axis('off')
                    # plotting the 2D contours
                    elif i > j:
                        for c, fm in zip(get_default_cycler(), self.values):
                            # plot 1-sigma 2D curves
                            # NOTE this is the "68% of the probability of a
                            # single parameter lying within the bounds projected
                            # onto a parameter axis"
                            plot_curve_2d(
                                fm,
                                namex,
                                namey,
                                ax=ax[i, j],
                                fill=False,
                                color=c["color"],
                                zorder=20,
                            )

                            # the 2-sigma
                            plot_curve_2d(
                                fm,
                                namex,
                                namey,
                                ax=ax[i, j],
                                scaling_factor=2,
                                fill=False,
                                color=c["color"],
                                zorder=20,
                            )

                    else:
                        # plotting the 1D Gaussians
                        if plot_1d_curves is True:
                            for c, fm in zip(get_default_cycler(), self.values):
                                plot_curve_1d(
                                    fm,
                                    namex,
                                    ax=ax[i, i],
                                    color=c["color"],
                                )

                                # 1 and 2 sigma shading
                                add_shading_1d(
                                    fiducial=fm.fiducials[np.where(fm.names == namex)],
                                    sigma=fm.constraints(namex, marginalized=True),
                                    ax=ax[i, i],
                                    level=1,
                                    alpha=0.2,
                                    color=c["color"],
                                    ec=None,
                                )
                                add_shading_1d(
                                    fiducial=fm.fiducials[np.where(fm.names == namex)],
                                    sigma=fm.constraints(namex, marginalized=True),
                                    ax=ax[i, i],
                                    level=2,
                                    alpha=0.1,
                                    color=c["color"],
                                    ec=None,
                                )

                        else:
                            ax[i, i].remove()

            # set automatic limits
            for i in range(size):
                for j in range(size):
                    try:
                        ax[i, j].relim()
                        ax[i, j].autoscale_view()
                        if i == j:
                            ax[i, i].set_ylim(0, ax[i, i].get_ylim()[-1])
                            ax[i, i].set_yticks([])
                            ax[i, i].set_yticklabels([])
                    except AttributeError:
                        pass

        return FisherFigure2D(fig, ax, names)


def get_chisq(
    sigma: float = 1,
    df: int = 2,
):
    r"""
    Returns \(\Delta \chi^2\).
    To obtain the scaling coefficient \(\alpha\), just take the square root of
    the output.

    Parameters
    ----------
    sigma : float = 1
        the error on the parameter

    df : int = 2
        the number of degrees of freedom

    Returns
    -------
    float
    """
    return chi2.ppf(norm.cdf(sigma) - norm.cdf(-sigma), df=df)


def add_plot_1d(
    fiducial: float,
    sigma: float,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Adds a 1D Gaussian with marginalized constraints `sigma` close to fiducial
    value `fiducial` to axis `ax`.
    """
    if ax is None:
        _, ax = plt.subplots()

    x = np.linspace(
        fiducial - 4 * sigma,
        fiducial + 4 * sigma,
        100,
    )

    ax.plot(
        x,
        [norm.pdf(_, loc=fiducial, scale=sigma) for _ in x],
        **kwargs,
    )

    return (ax.get_figure(), ax)


def add_shading_1d(
    fiducial: float,
    sigma: float,
    ax: Optional[Axes] = None,
    level: float = 1,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Add shading to a 1D axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    x = np.ndarray.flatten(
        np.linspace(
            fiducial - level * sigma,
            fiducial + level * sigma,
            100,
        )
    )

    ax.fill_between(
        x,
        np.ndarray.flatten(
            np.array([norm.pdf(_, loc=fiducial, scale=sigma) for _ in x])
        ),
        **kwargs,
    )

    return (ax.get_figure(), ax)


def plot_curve_1d(
    fisher: FisherMatrix,
    name: str,
    ax: Optional[Axes] = None,
    **kwargs,
):
    """
    Plots a 1D curve (usually marginalized Gaussian) from a Fisher object.

    Parameters
    --------
    fisher : FisherMatrix
        the Fisher matrix which we want to plot

    name : str
        the name of the parameter which we want to plot

    ax : Optional[matplotlib.axes.Axes] = None
        the axis on which we want to plot the contour. By default, plots to a
        new figure.

    Returns
    -------
    A 2-tuple `(Figure, Axes)`
    """
    if ax is None:
        _, ax = plt.subplots()

    fid = fisher.fiducials[np.where(fisher.names == name)]
    sigma = fisher.constraints(name, marginalized=True)

    add_plot_1d(fid, sigma, ax=ax, **kwargs)

    return (ax.get_figure(), ax)


def plot_shading_1d(
    fisher: FisherMatrix,
    name: str,
    p: float = 0.68,
    ax: Optional[Axes] = None,
    **kwargs,
):
    r"""
    Plots shading at some confidence interval.

    Parameters
    ----------
    fisher : FisherMatrix
        the Fisher matrix which we want to plot

    name : str
        the name of the parameter to plot

    p : float = 0.68
        the confidence interval around the fiducial value for which we want to
        plot the shading. The intervals are computed as:
        \[
            p = \int\limits_a^b p(x)\, \mathrm{d}x
        \]
        where \((a, b)\) contains the fiducial value, and \(p(x)\) is the PDF
        (not necessarily Gaussian) of the distribution

    ax : Optional[Axes] = None
        the axis on which to plot the shading
    """
    if ax is None:
        _, ax = plt.subplots()

    sigma = fisher.constraints(name, marginalized=True)


def plot_curve_2d(
    fisher: FisherMatrix,
    name1: str,
    name2: str,
    ax: Optional[Axes] = None,
    scaling_factor: float = 1,
    **kwargs,
) -> Tuple[Union[None, Figure], Axes]:
    """
    Plots a 2D curve (usually ellipse) from two parameters of a Fisher object.

    Parameters
    ----------
    fisher : FisherMatrix
        the Fisher matrix which we want to plot

    name1 : str
        the name of the first parameter which we want to plot

    name2 : str
        the name of the second parameter which we want to plot

    ax : Optional[matplotlib.axes.Axes] = None
        the axis on which we want to plot the contour. By default, plots to a
        new figure.

    Returns
    -------
    A 2-tuple `(Figure, Axes)`
    """
    if ax is None:
        _, ax = plt.subplots()

    fidx = fisher.fiducials[np.where(fisher.names == name1)]
    fidy = fisher.fiducials[np.where(fisher.names == name2)]

    a, b, angle = get_ellipse(fisher, name1, name2)

    ax.add_patch(
        Ellipse(
            xy=(fidx, fidy),
            width=2 * a * scaling_factor,
            height=2 * b * scaling_factor,
            angle=angle,
            **kwargs,
        ),
    )

    return (ax.get_figure(), ax)


def get_ellipse(
    fm: FisherMatrix,
    name1: str,
    name2: str,
) -> Tuple[float, float, float]:
    """
    Constructs parameters for a Gaussian ellipse from the names.
    """
    if name1 == name2:
        raise ValueError(f"Names must be different")

    # the inverse
    inv = FisherMatrix(fm.inverse(), names=fm.names)

    sigmax2 = inv[name1, name1]
    sigmay2 = inv[name2, name2]
    sigmaxy = inv[name1, name2]

    # basically the eigenvalues of the submatrix
    a = np.sqrt(
        (sigmax2 + sigmay2) / 2 + np.sqrt((sigmax2 - sigmay2) ** 2 / 4 + sigmaxy**2)
    )

    b = np.sqrt(
        (sigmax2 + sigmay2) / 2 - np.sqrt((sigmax2 - sigmay2) ** 2 / 4 + sigmaxy**2)
    )

    angle = np.rad2deg(
        np.arctan2(
            2 * sigmaxy,
            sigmax2 - sigmay2,
        )
        / 2
    )

    return a, b, angle


def set_xticks(
    ax: Axes,
):
    locator = LinearLocator(3)
    formatter = StrMethodFormatter("{x:.2f}")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticks(locator())
    ax.set_xticklabels(
        formatter.format_ticks(locator()),
        rotation=45,
        rotation_mode="anchor",
        fontdict={
            "verticalalignment": "top",
            "horizontalalignment": "right",
        },
    )


def set_yticks(
    ax: Axes,
):
    locator = LinearLocator(3)
    formatter = StrMethodFormatter("{x:.2f}")
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_yticks(locator())
    ax.set_yticklabels(
        formatter.format_ticks(locator()),
        fontdict={
            "verticalalignment": "top",
            "horizontalalignment": "right",
        },
    )


def get_default_cycler():
    return cycler(color=["C0", "C3", "C2"])
