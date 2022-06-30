"""
Submodule for plotting of Fisher objects.
See here for documentation of `FisherFigure1D` and `FisherFigure2D`.
"""

# needed for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
import copy
from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from itertools import product
from pathlib import Path
from typing import Any, Optional, Union

# third party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.ticker import Formatter, Locator
from matplotlib.transforms import Bbox
from scipy.stats import chi2, norm

# first party imports
from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_utils import ParameterNotFoundError, get_default_rcparams, is_iterable


class FisherBaseFigure(ABC):
    """
    The abstract base class for plotting Fisher objects.
    """

    def __init__(self, *args, options: Optional[dict] = None, **kwargs):
        """
        Constructor
        """
        self._figure = None
        self._axes = None
        self._handles: MutableSequence[Artist] = []
        self._names = None
        self._labels: MutableSequence[str] = []
        self._options = get_default_rcparams()
        self._ndim = 0

        self.cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.current_color = iter(self.cycler)

        if options and "style" in options:
            style = options.pop("style")
            # maybe it's one of the built-in styles
            if style in plt.style.available:
                self._options = {
                    **plt.style.library[style],
                    **get_default_rcparams(),
                    **options,
                }
                # we need to reset the color cycler
                self.cycler = (
                    plt.style.library[style]
                    .get(
                        "axes.prop_cycle",
                        plt.rcParams["axes.prop_cycle"],
                    )
                    .by_key()["color"]
                )
                self.current_color = iter(self.cycler)

            # maybe it's a file
            else:
                # we need to reset the color cycler
                self.cycler = (
                    plt.style.core.rc_params_from_file(style)
                    .get(
                        "axes.prop_cycle",
                        plt.rcParams["axes.prop_cycle"],
                    )
                    .by_key()["color"]
                )
                self.current_color = iter(self.cycler)

                self._options = {
                    **plt.style.core.rc_params_from_file(style),
                    **options,
                }

    @abstractmethod
    def plot(self, fisher: FisherMatrix, *args, **kwargs):
        """
        Implements plotting of Fisher objects.
        """
        return NotImplemented

    @abstractmethod
    def __getitem__(self, key):
        """
        Implements element access.
        """
        return NotImplemented

    @property
    def figure(self):
        """
        Returns the underlying figure, an instance of
        `matplotlib.figure.Figure`.
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
        Returns the names of the original parameters plotted.
        """
        return self._names

    @property
    def handles(self):
        """
        Returns the handles of the currently drawn artists.
        """
        return self._handles

    @property
    def labels(self):
        """
        Returns the legend labels of the currently drawn artists.
        """
        return self._labels

    @property
    def options(self):
        """
        Returns the matplotlib options which were used for plotting.
        """
        return self._options

    def __repr__(self):
        """
        Returns the representation of the figure
        """
        return (
            f"<FisherFigure(\n"
            f"    names={repr(self.names)},\n"
            f"    figure={repr(self.figure)},\n"
            f"    axes={repr(self.axes)})>"
        )

    def __str__(self):
        """
        Returns the string representation of the figure
        """
        return (
            f"FisherFigure(\n"
            f"    names={str(self.names)},\n"
            f"    figure={str(self.figure)},\n"
            f"    axes={str(self.axes)})"
        )

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
        self.figure.savefig(
            path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **kwargs,
        )

    def set_label_params(
        self,
        which: str = "both",
        **kwargs,
    ):
        """
        Collectively sets both x and y label parameters.

        Parameters
        ----------
        which
            which axes to change: 'x', 'y', or 'both' (default: 'both')

        kwargs
            any keyword arguments that are also valid for
            `matplotlib.text.Text`, see [the
            documentation](https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text)
            for details

        Returns
        -------
        None
        """
        for nameiter in product(self.names, repeat=self._ndim):
            # only set the parameters which are not empty (should this be done
            # for all of them instead?)
            if (
                self[nameiter]
                and self[nameiter].get_xlabel()
                and which in ["both", "x"]
            ):
                self[nameiter].set_xlabel(
                    self[nameiter].get_xlabel(),
                    **kwargs,
                )
            if (
                self[nameiter]
                and self[nameiter].get_ylabel()
                and which in ["both", "y"]
            ):
                self[nameiter].set_ylabel(
                    self[nameiter].get_ylabel(),
                    **kwargs,
                )

    def set_tick_params(
        self,
        which: str = "both",
        **kwargs,
    ):
        """
        Collectively sets both x and y tick parameters.

        Parameters
        ----------
        which
            which axes to change: 'x', 'y', or 'both' (default: 'both')

        kwargs
            any keyword arguments passed to the methods for handling tick
            parameters (such as 'fontsize', 'rotation', etc.)
        """
        for nameiter in product(self.names, repeat=self._ndim):
            if self[nameiter] and which in ["both", "x"]:
                for item in self[nameiter].get_xticklabels():
                    for key, value in kwargs.items():
                        getattr(item, f"set_{key}")(value)
            if self[nameiter] and which in ["both", "y"]:
                for item in self[nameiter].get_yticklabels():
                    for key, value in kwargs.items():
                        getattr(item, f"set_{key}")(value)
            # also alter any exponential offsets
            for key, value in kwargs.items():
                if self[nameiter] and which in ["both", "x"]:
                    getattr(
                        self[nameiter].get_xaxis().get_offset_text(),
                        f"set_{key}",
                    )(value)
                if self[nameiter] and which in ["both", "y"]:
                    getattr(
                        self[nameiter].get_yaxis().get_offset_text(),
                        f"set_{key}",
                    )(value)

    def set_major_locator(
        self,
        locator: Locator,
        which: str = "both",
    ):
        """
        Sets the major locator for all of the axes.

        Parameters
        ----------
        locator
            the instance of `matplotlib.ticker.Locator` to use.

        which
            which axes to change: 'x', 'y', or 'both' (default: 'both')
        """
        for nameiter in product(self.names, repeat=self._ndim):
            # for some reason, we cannot reuse the same instance, so we just
            # make a deep copy of it instead
            if (
                self[nameiter]
                and np.any(self[nameiter].get_xticks())
                and which in ["both", "x"]
            ):
                xloc = copy.deepcopy(locator)
                self[nameiter].xaxis.set_major_locator(xloc)
            if (
                self[nameiter]
                and np.any(self[nameiter].get_yticks())
                and which in ["both", "y"]
            ):
                yloc = copy.deepcopy(locator)
                self[nameiter].yaxis.set_major_locator(yloc)

    def set_minor_locator(
        self,
        locator: Locator,
        which: str = "both",
    ):
        """
        Sets the minor locator for all of the axes.

        Parameters
        ----------
        locator
            the instance of `matplotlib.ticker.Locator` to use.

        which
            which axes to change: 'x', 'y', or 'both' (default: 'both')
        """
        for nameiter in product(self.names, repeat=self._ndim):
            if (
                self[nameiter]
                and np.any(self[nameiter].get_xticks())
                and which in ["both", "x"]
            ):
                # same reason as the major locator
                xloc = copy.deepcopy(locator)
                self[nameiter].xaxis.set_minor_locator(xloc)
            if (
                self[nameiter]
                and np.any(self[nameiter].get_yticks())
                and which in ["both", "y"]
            ):
                yloc = copy.deepcopy(locator)
                self[nameiter].yaxis.set_minor_locator(yloc)

    def set_major_formatter(
        self,
        formatter: Formatter,
        which: str = "both",
    ):
        """
        Sets the major formatter for all of the axes.

        Parameters
        ----------
        formatter
            the instance of `matplotlib.ticker.Formatter` to use.

        which
            which axes to change: 'x', 'y', or 'both' (default: 'both')
        """
        for nameiter in product(self.names, repeat=self._ndim):
            if self[nameiter] and which in ["both", "x"]:
                self[nameiter].xaxis.set_major_formatter(formatter)
            if self[nameiter] and which in ["both", "y"]:
                self[nameiter].yaxis.set_major_formatter(formatter)

    def set_minor_formatter(
        self,
        formatter: Formatter,
        which: str = "both",
    ):
        """
        Sets the minor formatter for all of the axes.

        Parameters
        ----------
        formatter
            the instance of `matplotlib.ticker.Formatter` to use.

        which
            which axes to change: 'x', 'y', or 'both' (default: 'both')
        """
        for nameiter in product(self.names, repeat=self._ndim):
            if self[nameiter] and which in ["both", "x"]:
                self[nameiter].xaxis.set_minor_formatter(formatter)
            if self[nameiter] and which in ["both", "y"]:
                self[nameiter].yaxis.set_minor_formatter(formatter)


class FisherFigure1D(FisherBaseFigure):
    """
    Container for easy access to elements in the 1D plot.
    """

    def __init__(
        self,
        options: Optional[dict] = None,
        max_cols: Optional[int] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        options
            the dictionary containing the options for plotting. If the special
            key 'style' is present, it attempts to use that plotting style (can
            be one of the outputs of `matplotlib.pyplot.style.available`, or a
            path to a file. If using a file, does not use the default rc
            parameters).

        max_cols
            the maximum number of columns in the final plot

        Notes
        -----
        For the style sheet reference, please consult [the matplotlib
        documentation](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
        """
        super().__init__()
        self.max_cols = max_cols
        self._ndim = 1

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

    def draw(
        self,
        name: str,
        method: str,
        *args,
        **kwargs,
    ):
        """
        Implements drawing of other objects on the axis.
        """
        if not hasattr(self[name], method):
            raise AttributeError(
                f"The method `{method}` is not a valid plotting method"
            )

        # hopefully the return type should be some artist, or a collection of
        # artists
        handles = getattr(self[name], method)(*args, **kwargs)
        if is_iterable(handles):
            for handle in handles:
                if isinstance(handle, Artist) and kwargs.get("label"):
                    self.labels.append(kwargs.get("label"))
                    self.handles.append(handle)
        else:
            if isinstance(handles, Artist) and kwargs.get("label"):
                self.labels.append(kwargs.get("label"))
                self.handles.append(handles)

    def plot(
        self,
        fisher: FisherMatrix,
        *args,
        **kwargs,
    ):
        """
        Makes a 1D plot (Gaussians) of the Fisher objects

        Parameters
        ----------
        fisher
            the Fisher object which we want to plot

        Returns
        -------
        An instance of `FisherFigure1D`.

        Notes
        -----
        Also modifies the current instance
        """
        size = len(fisher)

        added = False

        # in order to preserve colors for each object, we only use the
        # (default) cycler when the color is not explicitly set
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = next(self.current_color)
            self.current_color = iter(self.current_color)

        if self.max_cols is not None and self.max_cols <= size:
            full = size % self.max_cols == 0
            layout = (
                size // self.max_cols if full else size // self.max_cols + 1,
                self.max_cols,
            )
        else:
            layout = 1, size
            full = True

        with plt.rc_context(self._options):
            # general figure setup
            if self.figure:
                fig = self.figure
                axes = self.axes
            else:
                fig = plt.figure(
                    clear=True,
                    figsize=(2 * layout[1], 2 * layout[0]),
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

            for (index, name), latex_name in zip(
                enumerate(fisher.names),
                fisher.latex_names,
            ):
                ax = axes.flat[index]

                _, __, handle = plot_curve_1d(fisher, name, ax, **kwargs)

                if not added and kwargs.get("label"):
                    self.handles.append(handle)
                    self.labels.append(kwargs["label"])
                    added = True

                _add_shading_1d(
                    fisher.fiducials[np.where(fisher.names == name)],
                    fisher.constraints(name, marginalized=True),
                    ax,
                    alpha=0.3,
                    ec=None,
                    **kwargs,
                )

                _add_shading_1d(
                    fisher.fiducials[np.where(fisher.names == name)],
                    fisher.constraints(name, marginalized=True),
                    ax,
                    level=2,
                    alpha=0.1,
                    ec=None,
                    **kwargs,
                )

                ax.autoscale()
                ax.set_xlabel(latex_name)
                ax.relim()
                ax.autoscale_view()

                # the y axis should start at 0 since we're plotting a PDF
                ax.set_ylim(0, ax.get_ylim()[-1])

                ax.set_yticks([])

            # remove any axes which are not drawn
            if not full:
                for index in range(
                    (layout[0] - 1) * layout[1] + 1, layout[0] * layout[1]
                ):
                    try:
                        axes.flat[index].remove()
                    except AttributeError:
                        pass

        self._figure = fig
        self._axes = axes
        self._names = fisher.names

    def legend(
        self,
        *args: Artist,
        overwrite: bool = False,
        loc: Union[str, tuple[float, float]] = "lower center",
        bbox_to_anchor: Any = [0.5, 1],
        **kwargs,
    ):
        """
        Creates a legend on the figure.
        """
        with plt.rc_context(self.options):
            if not overwrite:
                self.figure.legend(
                    self.handles,
                    self.labels,
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    **kwargs,
                )

            else:
                self.figure.legend(
                    args,
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    **kwargs,
                )


class FisherFigure2D(FisherBaseFigure):
    """
    Container for easy access to elements in the 2D plot.
    """

    def __init__(
        self,
        options: Optional[dict] = None,
        show_1d_curves: bool = False,
        show_joint_dist: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        options
            the dictionary containing the options for plotting. If the special
            key 'style' is present, it attempts to use that plotting style (can
            be one of the outputs of `matplotlib.pyplot.style.available`, or a
            path to a file. If using a file, does not use the default rc
            parameters).

        show_1d_curves
            whether the 1D marginalized curves should be plotted (default:
            False)

        show_joint_dist
            whether to plot the isocontours of the joint distribution (default:
            False)

        Notes
        -----
        For the style sheet reference, please consult [the matplotlib
        documentation](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
        """
        super().__init__()
        self.show_1d_curves = show_1d_curves
        self.show_joint_dist = show_joint_dist
        self._ndim = 2

    def __getitem__(
        self,
        key: tuple[str, str],
    ):
        if not isinstance(key, tuple):
            raise TypeError(
                f"Incompatible type for element access: expected `tuple`, got `{type(key)}`"
            )

        name1, name2 = key

        if name1 not in self.names:
            raise ParameterNotFoundError(name1, self.names)
        if name2 not in self.names:
            raise ParameterNotFoundError(name2, self.names)

        # edge case
        if not self.show_1d_curves and name1 == name2:
            return None

        index1 = np.where(self.names == name1)
        index2 = np.where(self.names == name2)

        if index1 < index2:
            index1, index2 = index2, index1

        return self.axes[index1, index2][0, 0]

    def legend(
        self,
        *args: Artist,
        overwrite: bool = False,
        loc: Union[str, tuple[float, float]] = "upper right",
        bbox_to_anchor: Optional[Any] = None,
        **kwargs,
    ):
        """
        Creates a legend on the figure.
        """
        with plt.rc_context(self.options):
            # this will always exist since we can't plot < 2 parameters
            ax = self[self.names[0], self.names[1]]
            if not overwrite:
                if self.show_1d_curves:
                    i, j = 0, -1
                else:
                    i, j = 1, -2

                bbox_to_anchor = (
                    self.axes[0, j].get_position().xmax,
                    self.axes[i, 0].get_position().ymax,
                )

                if ax.get_legend():
                    ax.get_legend().remove()

                ax.legend(
                    self.handles,
                    self.labels,
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=self.figure.transFigure,
                    **kwargs,
                )

            else:
                ax.legend(
                    *args,
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    **kwargs,
                )

    def set_title(
        self,
        *args,
        overwrite: bool = False,
        xscale: float = 0.5,
        yscale: float = 1.08,
        **kwargs,
    ):
        """
        Thin wrapper for setting the title of the figure with the correct
        options.
        """
        with plt.rc_context(self.options):
            if not overwrite:
                if self.show_1d_curves:
                    i, j = 0, -1
                else:
                    i, j = 1, -2

                x, y = (
                    self.axes[0, j].get_position().xmax,
                    self.axes[i, 0].get_position().ymax,
                )

                self.figure.suptitle(
                    *args,
                    x=x * xscale,
                    y=y * yscale,
                    **kwargs,
                )

            else:
                self.figure.suptitle(*args, **kwargs)

    def draw(
        self,
        name1: str,
        name2: str,
        method: str,
        *args,
        **kwargs,
    ):
        """
        Implements drawing of other objects on the axis.
        """
        if not hasattr(self[name1, name2], method):
            raise AttributeError(
                f"The method `{method}` is not a valid plotting method"
            )

        with plt.rc_context(self.options):
            # hopefully the return type should be some artist, or a collection
            # of artists
            handles = getattr(self[name1, name2], method)(*args, **kwargs)
            if is_iterable(handles):
                for handle in handles:
                    if isinstance(handle, Artist) and kwargs.get("label"):
                        self.labels.append(kwargs["label"])
                        self.handles.append(handle)
            else:
                if isinstance(handles, Artist) and kwargs.get("label"):
                    self.labels.append(kwargs["label"])
                    self.handles.append(handles)

    def plot(
        self,
        fisher: FisherMatrix,
        *args,
        **kwargs,
    ):
        r"""
        Plots the 2D contours (and optionally 1D curves) of the Fisher objects.

        Parameters
        ----------
        fisher
            the Fisher object which we want to plot

        joint
            whether to plot the p-value of the joint distribution, or the
            p-value of the probability of a single parameter lying within the
            bounds projected onto a parameter axis.

        Returns
        -------
        None
        """
        size = len(fisher)

        if size < 2:
            raise ValueError("Unable to make a 2D plot with < 2 parameters")

        if self.names is not None:
            # make sure they have the same names; if not, raise an error
            if set(self.names) != set(fisher.names):
                raise ValueError(
                    f"The Fisher object names ({fisher.names}) "
                    f"do not match those on the figure ({self.names})"
                )

            # otherwise, we reshuffle them
            fisher = fisher.sort(key=self.names)

        # in order to preserve colors for each object, we only use the
        # (default) cycler when the color is not explicitly set
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = next(self.current_color)
            self.current_color = iter(self.current_color)

        with plt.rc_context(self.options):
            # general figure setup
            if self.figure:
                fig = self.figure
                ax = self.axes
            else:
                fig = plt.figure(
                    figsize=(2 * size, 2 * size),
                    clear=True,
                )
                ax = fig.add_gridspec(
                    nrows=size,
                    ncols=size,
                    hspace=0.0,
                    wspace=0.0,
                ).subplots(
                    sharex="col",
                    sharey=False,
                )

            # set automatic limits
            for i, j in product(range(size), repeat=2):
                if i == j:
                    ax[i, i].set_yticks([])
                    ax[i, i].set_yticklabels([])
                    if i < size - 1:
                        ax[i, i].get_xaxis().set_visible(False)
                if i > 0 and 0 < j < size - 1:
                    ax[i, j].set_yticks([])
                    ax[i, j].set_yticklabels([])
                if i > j and i != size - 1:
                    ax[i, j].get_xaxis().set_visible(False)

            # flag for whether the current artist has already been added to the
            # legend handle
            added = False
            # loop over the parameters
            for i, j in product(range(len(fisher.names)), repeat=2):
                namey, latex_namey, namex, latex_namex = (
                    fisher.names[i],
                    fisher.latex_names[i],
                    fisher.names[j],
                    fisher.latex_names[j],
                )
                # labels for 2D contours (increasing y)
                if i > 0 and j == 0:
                    ax[i, j].set_ylabel(latex_namey)

                # labels for 2D contours (increasing x)
                if i == size - 1:
                    ax[i, j].set_xlabel(latex_namex)

                # removing any unnecessary axes from the gridspec
                if i < j and self.figure is None:
                    # calling `remove()` is better than `axis("off")` because
                    # then we obtain the proper bounding box
                    ax[i, j].remove()

                # plotting the 2D contours
                elif i > j:
                    # plot 1-sigma 2D curves
                    # NOTE this is the "68% of the probability of a
                    # single parameter lying within the bounds projected
                    # onto a parameter axis"
                    _, __, handle = plot_curve_2d(
                        fisher,
                        namex,
                        namey,
                        ax=ax[i, j],
                        scaling_factor=1
                        if not self.show_joint_dist
                        else np.sqrt(_get_chisq(1)),
                        fill=False,
                        zorder=20,
                        **kwargs,
                    )

                    if not added and kwargs.get("label"):
                        self.handles.append(handle)
                        self.labels.append(kwargs["label"])
                        added = True

                    # same thing, but shaded
                    plot_curve_2d(
                        fisher,
                        namex,
                        namey,
                        ax=ax[i, j],
                        scaling_factor=1
                        if not self.show_joint_dist
                        else np.sqrt(_get_chisq(1)),
                        fill=True,
                        alpha=0.2,
                        ec=None,
                        zorder=20,
                        **kwargs,
                    )

                    # the 2-sigma
                    plot_curve_2d(
                        fisher,
                        namex,
                        namey,
                        ax=ax[i, j],
                        scaling_factor=2
                        if not self.show_joint_dist
                        else np.sqrt(_get_chisq(2)),
                        fill=False,
                        zorder=20,
                        **kwargs,
                    )
                    # same thing, but shaded
                    plot_curve_2d(
                        fisher,
                        namex,
                        namey,
                        ax=ax[i, j],
                        scaling_factor=2
                        if not self.show_joint_dist
                        else np.sqrt(_get_chisq(2)),
                        fill=True,
                        alpha=0.1,
                        ec=None,
                        zorder=20,
                        **kwargs,
                    )

                if i == j:
                    # plotting the 1D Gaussians
                    if self.show_1d_curves is True:
                        plot_curve_1d(
                            fisher,
                            namex,
                            ax=ax[i, i],
                            **kwargs,
                        )

                    else:
                        if self.figure is None:
                            ax[i, i].remove()

            # set automatic limits
            for i, j in product(range(size), repeat=2):
                try:
                    ax[i, j].relim()
                    ax[i, j].autoscale_view()
                    if i == j:
                        ax[i, i].autoscale()
                        ax[i, i].set_ylim(0, ax[i, i].get_ylim()[-1])
                        ax[i, i].set_yticks([])
                        ax[i, i].set_yticklabels([])
                except AttributeError:
                    pass

        self._figure = fig
        self._axes = ax
        self._names = fisher.names


def _get_chisq(
    sigma: float = 1,
    df: int = 2,
):
    r"""
    Returns $\Delta \chi^2$.
    To obtain the scaling coefficient $\alpha$, just take the square root of
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


def _add_plot_1d(
    fiducial: float,
    sigma: float,
    ax: Optional[Axes] = None,
    **kwargs,
) -> tuple[Figure, Axes, Artist]:
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

    (handle,) = ax.plot(
        x,
        [norm.pdf(_, loc=fiducial, scale=sigma) for _ in x],
        **kwargs,
    )

    return (ax.get_figure(), ax, handle)


def _add_shading_1d(
    fiducial: float,
    sigma: float,
    ax: Optional[Axes] = None,
    level: float = 1,
    **kwargs,
) -> tuple[Figure, Axes]:
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
) -> tuple[Figure, Axes, Artist]:
    """
    Plots a 1D curve (usually marginalized Gaussian) from a Fisher object.

    Parameters
    ----------
    fisher : FisherMatrix
        the Fisher matrix which we want to plot

    name : str
        the name of the parameter which we want to plot

    ax : Optional[matplotlib.axes.Axes] = None
        the axis on which we want to plot the contour. By default, plots to a
        new figure.

    Returns
    -------
    A 3-tuple `(Figure, Axes, Artist)`
    """
    if ax is None:
        _, ax = plt.subplots()

    fid = fisher.fiducials[np.where(fisher.names == name)]
    sigma = fisher.constraints(name, marginalized=True)

    return _add_plot_1d(fid, sigma, ax=ax, **kwargs)


def plot_shading_1d(
    fisher: FisherMatrix,
    name: str,
    p: float = 0.68,
    ax: Optional[Axes] = None,
    **kwargs,
) -> tuple[Figure, Axes]:
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
        $$
            p = \int\limits_a^b p(x)\, \mathrm{d}x
        $$
        where $(a, b)$ contains the fiducial value, and $p(x)$ is the PDF
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
) -> tuple[Figure, Axes, Artist]:
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
    A 3-tuple `(Figure, Axes, Artist)`
    """
    if ax is None:
        _, ax = plt.subplots()

    fidx = fisher.fiducials[np.where(fisher.names == name1)]
    fidy = fisher.fiducials[np.where(fisher.names == name2)]

    a, b, angle = _get_ellipse(fisher, name1, name2)

    patch = ax.add_patch(
        Ellipse(
            xy=(fidx, fidy),
            width=2 * a * scaling_factor,
            height=2 * b * scaling_factor,
            angle=angle,
            **kwargs,
        ),
    )

    return (ax.get_figure(), ax, patch)


def _get_ellipse(
    fm: FisherMatrix,
    name1: str,
    name2: str,
) -> tuple[float, float, float]:
    """
    Constructs parameters for a Gaussian ellipse from the names.
    """
    if name1 == name2:
        raise ValueError("Names must be different")

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
