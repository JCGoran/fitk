"""
Package for plotting of Fisher objects.
See here for documentation of `FisherPlotter`, `FisherFigure1D`, and `FisherFigure2D`.
"""

from __future__ import annotations

# standard library imports
from abc import ABC, abstractmethod
from typing import Collection, Optional, Tuple, Union

import matplotlib.pyplot as plt

# third party imports
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.transforms import Bbox
from scipy.stats import chi2, norm

from matplotlib.ticker import (
    MaxNLocator,
    LinearLocator,
    ScalarFormatter,
    StrMethodFormatter,
)
from scipy.stats import chi2, norm

# first party imports
from fitk.fisher_matrix import FisherMatrix

# first party imports
from fitk.fisher_utils import (
    MismatchingSizeError,
    ParameterNotFoundError,
    float_to_latex,
    get_default_rcparams,
    get_index_of_other_array,
    is_iterable,
)


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
        path: str,
        dpi: float = 300,
        bbox_inches: Union[str, Bbox] = "tight",
        **kwargs,
    ):
        """
        Convenience wrapper for `figure.savefig`.

        Parameters
        ----------
        path : str
            the path where to save the figure

        dpi : float, default = 300
            the resolution of the saved figure

        bbox_inches : Union[str, Bbox], default = 'tight'
            what is the bounding box for the figure

        kwargs
            any other keyword arguments that should be passed to `figure.savefig`
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
        key: Tuple[str],
    ):
        pass


class FisherPlotter:
    """
    Class for plotting FisherMatrix objects.
    """

    def __init__(
        self,
        *args: FisherMatrix,
        labels: Optional[Collection[str]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        args : FisherMatrix
            `FisherMatrix` objects which we want to plot.
            Must have the same parameter names.
            Can have different fiducial values.
            The order of plotting of the parameters and the LaTeX labels to use
            are determined by the first argument.

        labels : array of strings, default None
            the list of labels to put in the legend of the plots.
            If not set, defaults to `0, ..., len(args) - 1`

        Raises
        ------
        `ValueError` if parameter names of the inputs do not match
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

    def find_limits_1d(
        self,
        name: str,
        sigma: float = 3,
    ):
        """
        Finds "nice" 1D limits for a given parameter taking into account fiducials
        and constraints.

        Parameters
        ----------
        name : str
            the name of the parameter

        sigma : float, default = 3
            how many sigmas away to plot

        Returns
        -------
        `tuple` with lower and upper limits
        """
        sigmas = np.array(
            [_.constraints(name, marginalized=True, sigma=sigma) for _ in self.values]
        )
        fiducials = np.array(
            [_.fiducials[np.where(_.names == name)] for _ in self.values]
        )

        xleft, xright = np.min(fiducials - sigmas), np.max(fiducials + sigmas)

        return xleft, xright

    def plot_1d(
        self,
        scale: float = 1.0,
        max_cols: Optional[int] = None,
        rc: Optional[dict] = None,
        **kwargs,
    ):
        """
        Makes a 1D plot (Gaussians) of the Fisher objects

        Parameters
        ----------
        max_cols : Optional[int], default = None
            the maximum number of columns to force the plot into.
            By default, the parameters are always plotted horizontally; if you
            need to spread it over `max_cols`, pass a non-negative integer
            here.

        rc : Optional[dict], default = None
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

        if not rc:
            rc = get_default_rcparams()

        with plt.rc_context(rc):
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

            ylabel1d = r"$p (\theta)$"

            handles = []

            for (index, name), name_latex in zip(enumerate(names), latex_names):
                ax = axes.flat[index]
                title_list = [
                    "{0} = ${1}^{{+{2}}}_{{-{2}}}$".format(
                        name_latex,
                        float_to_latex(float(_.fiducials[np.where(_.names == name)])),
                        float_to_latex(float(_.constraints(name, marginalized=True))),
                    )
                    for _ in self.values
                ]

                # the scaling factor here is so that we don't cutoff the peak
                ymax = (
                    np.max(
                        [
                            gaussian(0, 0, _.constraints(name, marginalized=True))
                            for _ in self.values
                        ]
                    )
                    * 1.03
                )

                for fm in self.values:
                    (handle,) = add_plot_1d(
                        fm.fiducials[np.where(fm.names == name)],
                        fm.constraints(name, marginalized=True),
                        ax,
                    )

                    add_shading_1d(
                        fm.fiducials[np.where(fm.names == name)],
                        fm.constraints(name, marginalized=True),
                        ax,
                        color=handle.get_color(),
                        alpha=0.3,
                        ec=None,
                    )

                    add_shading_1d(
                        fm.fiducials[np.where(fm.names == name)],
                        fm.constraints(name, marginalized=True),
                        ax,
                        level=2,
                        color=handle.get_color(),
                        alpha=0.1,
                        ec=None,
                    )

                    if index == 0:
                        handles.append(handle)

                ax.set_xlabel(name_latex)
                ax.set_xlim(*self.find_limits_1d(name))
                ax.set_ylim(0, ymax)

                if kwargs.get("title") is True:
                    ax.set_title("\n".join(title_list))

                if index == 0:
                    ax.set_ylabel(ylabel1d)

                ax.set_yticks([])

            if kwargs.get("legend") is True:
                # remove any axes which are not drawn
                if not full:
                    for index in range(
                        (layout[0] - 1) * layout[1] + 1, layout[0] * layout[1]
                    ):
                        axes.flat[index].remove()

                fig.legend(
                    np.array(handles, dtype=object),
                    self.labels,
                    frameon=False,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=len(self.values),
                )

            if isinstance(kwargs.get("title"), str):
                fig.suptitle(kwargs.get("title"))

            # remove any axes which are not drawn
            if not full:
                for index in range(
                    (layout[0] - 1) * layout[1] + 1, layout[0] * layout[1]
                ):
                    axes.flat[index].remove()

        return FisherFigure1D(fig, axes, names)

    def plot_triangle(
        self,
        rc: Optional[dict] = None,
        plot_gaussians: bool = True,
        **kwargs,
    ):
        """
        Plots the contours (and optionally 1D Gaussians) of the Fisher objects.

        Parameters
        ----------
        rc : Optional[dict], default = None
            any parameters meant for `matplotlib.rcParams`. By default, only
            sets default font to cm serif.
            See [Matplotlib documentation](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
            for more information.
        plot_gaussians : bool, default = True
            whether or not the Gaussians should be plotted

        Returns
        -------
        An instance of `FisherFigure2D`.
        """
        size = len(self.values[0])

        if size < 2:
            raise ValueError("Unable to make a 2D plot with < 2 parameters")

        if not rc:
            rc = get_default_rcparams()

        with plt.rc_context(rc):
            # general figure setup
            fig = plt.figure(figsize=(2 * size, 2 * size), clear=True)
            gs = fig.add_gridspec(nrows=size, ncols=size, hspace=0.5, wspace=0.5)
            # TODO make it work with a shared xcol
            ax = gs.subplots(sharex=False, sharey=False)

            names = self.values[0].names
            latex_names = self.values[0].latex_names

            for i in range(len(ax)):
                for j in range(len(ax)):
                    # TODO is this the right order?
                    namex, namey = (
                        names[np.where(names == names[i])],
                        names[np.where(names == names[j])],
                    )
                    # labels for 2D contours (increasing y)
                    if i > 0 and j == 0:
                        ax[i, j].set_ylabel(latex_names[i])

                    # labels for 2D contours (increasing x)
                    if i == len(ax) - 1:
                        ax[i, j].set_xlabel(latex_names[j])

                    # removing any unnecessary axes from the gridspec
                    # TODO should they be removed, or somehow just made invisible?
                    if i < j:
                        ax[i, j].remove()
                        # ax[i, j].axis('off')
                    # plotting the 2D contours
                    elif i > j:
                        for fm in self.values:
                            # get parameters of the ellipse
                            width, height, angle = get_ellipse(fm, namex, namey)
                            fidx, fidy = (
                                fm.fiducials[np.where(fm.names == namex)],
                                fm.fiducials[np.where(fm.names == namey)],
                            )
                            alpha = np.sqrt(get_chisq())
                            add_plot_2d(
                                (fidx, fidy),
                                width=width * alpha,
                                height=height * alpha,
                                angle=angle,
                                ax=ax[i, j],
                                alpha=0.3,
                                fill=True,
                                color="red",
                            )

                            add_plot_2d(
                                (fidx, fidy),
                                width=width * alpha * 2,
                                height=height * alpha * 2,
                                angle=angle,
                                ax=ax[i, j],
                                alpha=0.1,
                                fill=True,
                                color="red",
                            )

                            # ax[i, j].set_ylim(*self.find_limits_1d(namex, sigma=1))
                            # ax[i, j].relim()

                        # remove ticks from any plots that aren't on the edges
                    #                        if j > 0:
                    #                            ax[i, j].set_yticks([])
                    #                            ax[i, j].set_xticklabels([])
                    #                            ax[i, j].set_yticklabels([])
                    #
                    #                        if i < len(ax) - 1:
                    #                            ax[i, j].set_xticks([])
                    #                            ax[i, j].set_xticklabels([])
                    #                            ax[i, j].set_yticklabels([])

                    # plotting the 1D Gaussians
                    elif plot_gaussians is True:
                        for fm in self.values:
                            (handle,) = add_plot_1d(
                                fiducial=fm.fiducials[np.where(fm.names == namex)],
                                sigma=fm.constraints(namex, marginalized=True),
                                ax=ax[i, i],
                            )
                            # 1 and 2 sigma shading
                            add_shading_1d(
                                fiducial=fm.fiducials[np.where(fm.names == namex)],
                                sigma=fm.constraints(namex, marginalized=True),
                                ax=ax[i, i],
                                level=1,
                                alpha=0.3,
                                color=handle.get_color(),
                                ec=None,
                            )
                            add_shading_1d(
                                fiducial=fm.fiducials[np.where(fm.names == namex)],
                                sigma=fm.constraints(namex, marginalized=True),
                                ax=ax[i, i],
                                level=2,
                                alpha=0.1,
                                color=handle.get_color(),
                                ec=None,
                            )

                    else:
                        ax[i, i].remove()

            #                        ax[i, i].set_yticks([])

            # set automatic limits
            for i in range(len(ax)):
                for j in range(len(ax)):
                    try:
                        ax[i, j].relim()
                        ax[i, j].autoscale_view()
                        if i == j:
                            ax[i, i].set_ylim(0, ax[i, i].get_ylim()[-1])
                        set_xticks(ax[i, j])
                        set_yticks(ax[i, j])
                    except AttributeError:
                        pass
        #                ax[i, i].set_xlim(*self.find_limits_1d(names[i]))

        return FisherFigure2D(fig, ax, names)


def gaussian(
    x: float,
    mu: float = 0,
    sigma: float = 1,
):
    """
    Returns a normalized Gaussian.
    """
    if sigma <= 0:
        raise ValueError(f"Invalid parameter: sigma = {sigma}")

    return np.exp(-((x - mu) ** 2) / 2 / sigma**2) / sigma / np.sqrt(2 * np.pi)


def get_chisq(
    sigma: float = 1,
    df: int = 2,
):
    r"""
    Returns \(\Delta \chi^2\).
    To obtain the scaling coefficient \(\alpha\), just take the square root of the output.

    Parameters
    ----------
    sigma : float, default = 1
        the error on the parameter

    df : int, default = 2
        the number of degrees of freedom

    Returns
    -------
    float
    """
    return chi2.ppf(norm.cdf(sigma) - norm.cdf(-sigma), df=df)


def add_plot_1d(
    fiducial: float,
    sigma: float,
    ax: Axes,
    **kwargs,
):
    """
    Adds a 1D Gaussian with marginalized constraints `sigma` close to fiducial
    value `fiducial` to axis `ax`.
    """
    x = np.linspace(
        fiducial - 4 * sigma,
        fiducial + 4 * sigma,
        100,
    )

    temp = ax.plot(
        x,
        [gaussian(_, mu=fiducial, sigma=sigma) for _ in x],
        **kwargs,
    )

    return temp


def add_shading_1d(
    fiducial: float,
    sigma: float,
    ax: Axes,
    level: float = 1,
    **kwargs,
):
    """
    Add shading to a 1D axes.
    """
    fiducial = fiducial.flatten()[0]
    sigma = sigma.flatten()[0]

    x = np.linspace(
        fiducial - level * sigma,
        fiducial + level * sigma,
        100,
    )

    temp = ax.fill_between(
        x,
        [gaussian(_, mu=fiducial, sigma=sigma) for _ in x],
        **kwargs,
    )

    return temp


def add_plot_2d(
    fiducials: Collection[float],
    width: float,
    height: float,
    angle: float,
    ax,
    **kwargs,
):
    """
    Adds a 2D ellipse with parameters `width`, `height`, and `angle`, centered
    at fiducial values `fiducial` to axis `ax`.
    """
    fidy, fidx = fiducials
    patch = ax.add_patch(
        Ellipse(
            xy=(fidx, fidy),
            width=width,
            height=height,
            angle=angle,
            **kwargs,
        )
    )

    return patch


def get_ellipse(
    fm: FisherMatrix,
    name1: str,
    name2: str,
):
    """
    Constructs parameters for an ellipse from the names.
    """
    if name1 == name2:
        return None

    # the inverse
    inv = fm.inverse()

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

    if sigmax2 < sigmay2:
        a, b = b, a

    angle = np.rad2deg(
        np.arctan2(
            2 * sigmaxy,
            sigmax2 - sigmay2,
        )
        / 2
    )

    return a, b, angle


def set_xticks(
    ax,
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
    ax,
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
