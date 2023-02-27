"""
Submodule for plotting of Fisher objects.
See here for documentation of `FisherFigure1D`, `FisherFigure2D`, and `FisherBarFigure`.
"""

# needed for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
import copy
import os  # pylint: disable=unused-import
from abc import ABC, abstractmethod
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from itertools import islice, product
from pathlib import Path
from typing import Any, Optional, Union

# third party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.patheffects import Normal, Stroke
from matplotlib.ticker import Formatter, Locator
from matplotlib.transforms import Bbox
from scipy.stats import chi2, norm

# first party imports
from fitk.tensors import FisherMatrix
from fitk.utilities import (
    MismatchingSizeError,
    MismatchingValuesError,
    ParameterNotFoundError,
    float_to_latex,
    get_default_rcparams,
    is_iterable,
)


@dataclass
class _TempContainer:
    labels: Sequence[str]
    colors: Sequence[str]
    x_array: Sequence[Sequence[float]]
    y_array: Sequence[Sequence[float]]
    space_per_object: float
    values_label: str
    scale: str


class _FisherBaseFigure:
    def __init__(
        self,
        options: Optional[dict] = None,
        **kwargs,
    ):
        """
        Constructor
        """
        self._figure = None
        self._axes: Optional[Union[Axes, Sequence[Axes]]] = None
        self._handles: MutableSequence[Artist] = []
        self._labels: MutableSequence[str] = []
        self._options: dict = get_default_rcparams()

        self.cycler = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        self.current_color = iter(self.cycler)

        if options and "style" in options:
            style = options.pop("style")
            # maybe it's one of the built-in styles
            if style in mpl.style.available:
                self._options = {
                    **mpl.style.library[style],
                    **get_default_rcparams(),
                    **options,
                }
                # we need to reset the color cycler
                self.cycler = (
                    mpl.style.library[style]
                    .get(
                        "axes.prop_cycle",
                        mpl.rcParams["axes.prop_cycle"],
                    )
                    .by_key()["color"]
                )
                self.current_color = iter(self.cycler)

            # maybe it's a file
            else:
                # we need to reset the color cycler
                self.cycler = (
                    mpl.rc_params_from_file(style)
                    .get(
                        "axes.prop_cycle",
                        mpl.rcParams["axes.prop_cycle"],
                    )
                    .by_key()["color"]
                )
                self.current_color = iter(self.cycler)

                self._options = {
                    **mpl.rc_params_from_file(style),
                    **options,
                }

    @property
    def options(self):
        """
        Returns the matplotlib options which were used for plotting.
        """
        return self._options

    @property
    def axes(self):
        """
        Returns the axes of the figure as a numpy array.
        """
        return self._axes

    @property
    def figure(self):
        """
        Returns the underlying figure, an instance of <a
        href="https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure"
        target="_blank" rel="noreferrer
        noopener">`matplotlib.figure.Figure`</a>.
        """
        return self._figure

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

    def savefig(
        self,
        path: Union[str, Path],
        dpi: float = 300,
        bbox_inches: Union[str, Bbox] = "tight",
        **kwargs,
    ):
        """
        Convenience wrapper for <a
        href="https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.savefig"
        target="_blank" rel="noreferrer
        noopener">`matplotlib.figure.Figure.savefig`</a>.

        Parameters
        ----------
        path : Path or str
            the path where to save the figure

        dpi : float, optional
            the resolution of the saved figure (default: 300)

        bbox_inches : str or Bbox, optional
            what is the bounding box for the figure (default: 'tight')

        **kwargs
            any other keyword arguments that should be passed to <a
            href="https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.savefig"
            target="_blank" rel="noreferrer
            noopener">`matplotlib.figure.Figure.savefig`</a>

        Returns
        -------
        None

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

        self.figure.savefig(
            path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **kwargs,
        )


class EmptyFigureError(Exception):
    """
    Error raised when the figure is empty
    """

    def __init__(self):
        self.message = (
            "The figure is empty; make sure to call one of the plotting methods before"
        )
        super().__init__(self.message)


class _FisherMultipleAxesFigure(_FisherBaseFigure, ABC):
    """
    The abstract base class for plotting Fisher objects.
    """

    def __init__(
        self,
        *args,
        options: Optional[dict] = None,
        hspace: float = 0.1,
        wspace: float = 0.1,
        contour_levels: Optional[Sequence[tuple[float, float]]] = None,
        **kwargs,
    ):
        """
        Constructor
        """
        self._names = None
        self._options: dict = {}
        self._ndim: int = 0
        self._hspace = hspace
        self._wspace = wspace

        super().__init__(options=options, **kwargs)

        # parse `contour_levels`
        if contour_levels is None:
            self.contour_levels = [(1.0, 0.3), (2.0, 0.1)]
        else:
            self.contour_levels = _parse_contour_levels(contour_levels)

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
    def hspace(self) -> float:
        """
        The amount of height reserved for space between subplots, expressed as
        a fraction of the average axis height.
        """
        return self._hspace

    @property
    def wspace(self) -> float:
        """
        The amount of width reserved for space between subplots, expressed as a
        fraction of the average axis width.
        """
        return self._wspace

    @property
    def names(self):
        """
        Returns the names of the original parameters plotted.
        """
        return self._names

    def __repr__(self):
        """
        Returns the representation of the figure
        """
        return (
            f"<{self.__class__.__name__}(\n"
            f"    names={repr(self.names)},\n"
            f"    figure={repr(self.figure)},\n"
            f"    axes={repr(self.axes)})>"
        )

    def __str__(self):
        """
        Returns the string representation of the figure
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"    names={str(self.names)},\n"
            f"    figure={str(self.figure)},\n"
            f"    axes={str(self.axes)})"
        )

    def add_artist_to_legend(
        self,
        artist: Artist,
        label: str,
    ):
        r"""
        This is a convenience function for correctly updating the legend after
        directly plotting some artist via `<instance>[<name(s)>].<method>`

        Parameters
        ----------
        artist
            the artist which we want to put on the legend

        label : str
            the label for the artist

        Returns
        -------
        None

        Examples
        --------
        Setup the plot:
        >>> fig = FisherFigure1D()
        >>> fig.plot(FisherMatrix(np.diag([1, 2]), names=["a", "b"]), label="Fisher matrix")

        Add something to the plot that has a handle:
        >>> handle, = fig["a"].plot([-1, 0, 1], [-1, 0, 1], color='red')

        Finally, add the handle to the legend, and draw the legend:
        >>> fig.add_artist_to_legend(handle, "linear function")
        >>> _ = fig.legend() # the artist will now be shown correctly in the legend

        Save it to a file:
        >>> fig.savefig(
        ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure2d_add_artist_to_legend1.png",
        ... dpi=150)

        <img width="100%" src="$IMAGE_PATH/fisher_figure2d_add_artist_to_legend1.png">

        Notes
        -----
        The limits of the plot are not updated automatically.
        """
        self.labels.append(label)
        self.handles.append(artist)

    def set_label_params(
        self,
        which: str = "both",
        **kwargs,
    ):
        """
        Collectively sets both x and y label parameters.

        Parameters
        ----------
        which : {'both', 'x', 'y'}
            which axis to change (default: 'both')

        **kwargs
            any keyword arguments that are also valid for <a
            href="https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text"
            target="_blank" rel="noopener
            noreferrer">`matplotlib.text.Text`</a>.

        Returns
        -------
        None

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

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
        which : {'both', 'x', 'y'}
            which axis to change (default: 'both')

        **kwargs
            any keyword arguments passed to the methods for handling tick
            parameters (such as 'fontsize', 'rotation', etc.)

        Returns
        -------
        None

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

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
            the instance of <a
            href="https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Locator"
            target="_blank" rel="noopener
            noreferrer">`matplotlib.ticker.Locator`</a> to use.

        which : {'both', 'x', 'y'}
            which axis to change (default: 'both')

        Returns
        -------
        None

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

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
            the instance of <a
            href="https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Locator"
            target="_blank" rel="noopener
            noreferrer">`matplotlib.ticker.Locator`</a> to use.

        which : {'both', 'x', 'y'}
            which axis to change (default: 'both')

        Returns
        -------
        None

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

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
            the instance of <a
            href="https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Formatter"
            target="_blank" rel="noopener
            noreferrer">`matplotlib.ticker.Formatter`</a> to use.

        which : {'both', 'x', 'y'}
            which axis to change (default: 'both')

        Returns
        -------
        None

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

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
            the instance of <a
            href="https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Formatter"
            target="_blank" rel="noopener
            noreferrer">`matplotlib.ticker.Formatter`</a> to use.

        which : {'both', 'x', 'y'}
            which axis to change (default: 'both')

        Returns
        -------
        None

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

        for nameiter in product(self.names, repeat=self._ndim):
            if self[nameiter] and which in ["both", "x"]:
                self[nameiter].xaxis.set_minor_formatter(formatter)
            if self[nameiter] and which in ["both", "y"]:
                self[nameiter].yaxis.set_minor_formatter(formatter)


class FisherBarFigure(_FisherBaseFigure):
    """
    Container for plotting single-axis bar-like figures
    (`plot_absolute_constraints` and `plot_relative_constraints`)
    """

    def _parse_fractional_constraints(
        self,
        args: Sequence[FisherMatrix],
        marginalized: bool = True,
        colors: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        values_label: Optional[str] = None,
        percent: bool = False,
        space: float = 0.3,
        scale: str = "linear",
    ):
        # error handling
        if not 0 <= space <= 1:
            raise ValueError("The value of `space` must be in the open interval (0, 1)")

        allowed_scales = ["linear", "log", "symlog"]
        if scale not in allowed_scales:
            raise ValueError(f"The value of `scale` must be one of: {allowed_scales}")

        for arg in args:
            if set(args[0].names) != set(arg.names):
                raise MismatchingValuesError(
                    "parameter name",
                    args[0].names,
                    arg.names,
                )

        # total number of parameters in the Fisher objects
        size = len(args[0])

        # the width or height of all columns for a single parameter
        total_space = 1 - space

        # the width or height of a single column for a single parameter
        space_per_object = total_space / len(args)

        if not colors:
            colors = list(islice(self.current_color, len(args)))  # type: ignore
        elif len(colors) != len(args):
            raise MismatchingSizeError(colors, args)

        if not labels:
            labels = [""] * len(args)
        elif len(labels) != len(args):
            raise MismatchingSizeError(labels, args)

        if not values_label:
            if percent:
                values_label = r"$\sigma / \theta_\mathrm{fid}\ (\%)$"
            else:
                values_label = r"$\sigma / \theta_\mathrm{fid}$"

        return _TempContainer(
            labels,
            colors,  # type: ignore
            [
                [
                    _ - total_space / 2 + (2 * index + 1) * space_per_object / 2
                    for _ in range(size)
                ]
                for index, arg in enumerate(args)
            ],
            [
                np.array(
                    [
                        arg.constraints(name=_, marginalized=marginalized)[0]
                        for _ in arg.names
                    ]
                )
                / np.array(
                    [
                        np.abs(fid) if not np.isclose(fid, 0) else 1
                        for fid in arg.fiducials
                    ]
                )
                for index, arg in enumerate(args)
            ],
            space_per_object=space_per_object,
            values_label=values_label,
            scale=scale,
        )

    def plot_absolute_constraints(
        self,
        args: Sequence[FisherMatrix],
        kind: str,
        marginalized: bool = True,
        colors: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        values_label: Optional[str] = None,
        scale: str = "linear",
        space: float = 0.3,
        capsize: Optional[float] = None,
        **kwargs,
    ):
        r"""
        Makes a plot of the constraints of the Fisher matrices

        Parameters
        ----------
        args : array_like of FisherMatrix
            the Fisher matrices for which we want to plot the constraints

        kind : str, {'bar', 'barh', 'errorbar'}
            the kind of plot we want (vertical bar, horizontal bar, errorbar)

        marginalized : bool, optional
            whether the marginalized or the unmarginalized constraints should
            be plotted (default: True)

        colors : array_like of str, optional
            the colors to use for the plotting (default: default matplotlib
            colors)

        labels : array_like of str, optional
            the labels for the Fisher matrices (default: None)

        values_label : str, optional
            the label for the axis containing the constraints (default:
            `$\theta_\mathrm{fid}$`)

        scale : str, {'linear', 'log'}
            the scale used for the y axis (default: 'linear')

        space : float, optional
            the space reserved between the bars (default: 0.3)

        capsize : float, optional
            the width of the errorbars (default: none, that is, determined
            automatically)

        **kwargs
            any keyword arguments passed to <a
            href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html"
            target="_blank" rel=noreferrer
            noopener>`matplotlib.pyplot.bar`</a>, or <a
            href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html"
            target="_blank" rel=noreferrer
            noopener>`matplotlib.pyplot.barh`</a>, or <a
            href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html"
            target="_blank" rel=noreferrer
            noopener>`matplotlib.pyplot.errorbar`</a>.
        Returns
        -------
        None

        Examples
        --------
        Define some Fisher matrices:
        >>> fm1 = FisherMatrix(np.diag([1, 2, 3]) * 1e3, names=["a", "b", "c"], latex_names=[r'$\Omega_\mathrm{m}$', '$h$', "$n_s$"], fiducials=[0.3, 0.7, 0.96])
        >>> fm2 = FisherMatrix(np.diag([4, 5, 6]) * 1e3, names=["a", "b", "c"], latex_names=[r'$\Omega_\mathrm{m}$', '$h$', "$n_s$"], fiducials=[0.35, 0.67, 0.9])

        Make a vertical bar plot:
        >>> fig = FisherBarFigure()
        >>> fig.plot_absolute_constraints([fm1, fm2], 'bar', labels=['Fisher 1', 'Fisher 2'])
        >>> _ = fig.figure.legend()

        Save it to a file:
        >>> fig.savefig(
        ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure_plot_absolute_constraints_bar.png",
        ... dpi=150)

        <img width="100%" src="$IMAGE_PATH/fisher_figure_plot_absolute_constraints_bar.png">

        Make a horizontal bar plot:
        >>> fig = FisherBarFigure()
        >>> fig.plot_absolute_constraints([fm1, fm2], 'barh', labels=['Fisher 1', 'Fisher 2'])
        >>> _ = fig.figure.legend()

        Save it to a file:
        >>> fig.savefig(
        ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure_plot_absolute_constraints_barh.png",
        ... dpi=150)

        <img width="100%" src="$IMAGE_PATH/fisher_figure_plot_absolute_constraints_barh.png">

        Make an errobar plot:
        >>> fig = FisherBarFigure()
        >>> fig.plot_absolute_constraints([fm1, fm2], 'errorbar', labels=['Fisher 1', 'Fisher 2'], capsize=4)
        >>> _ = fig.figure.legend()

        Save it to a file:
        >>> fig.savefig(
        ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure_plot_absolute_constraints_errorbar.png",
        ... dpi=150)

        <img width="100%" src="$IMAGE_PATH/fisher_figure_plot_absolute_constraints_errorbar.png">
        """
        size = len(args[0])
        latex_names = args[0].latex_names

        # error handling
        if not 0 <= space <= 1:
            raise ValueError("The value of `space` must be in the open interval (0, 1)")

        allowed_scales = ["linear", "log"]
        if scale not in allowed_scales:
            raise ValueError(f"The value of `scale` must be one of: {allowed_scales}")

        for arg in args:
            if set(args[0].names) != set(arg.names):
                raise MismatchingValuesError(
                    "parameter name",
                    args[0].names,
                    arg.names,
                )

        # the width or height of all columns for a single parameter
        total_space = 1 - space

        # the width or height of a single column for a single parameter
        space_per_object = total_space / len(args)

        if not colors:
            colors = list(islice(self.current_color, len(args)))  # type: ignore
        elif len(colors) != len(args):
            raise MismatchingSizeError(colors, args)

        if not labels:
            labels = [""] * len(args)
        elif len(labels) != len(args):
            raise MismatchingSizeError(labels, args)

        if not values_label:
            values_label = r"$\theta_\mathrm{fid}$"

        x_array = [
            [
                _ - total_space / 2 + (2 * index + 1) * space_per_object / 2
                for _ in range(size)
            ]
            for index, arg in enumerate(args)
        ]
        y_array = [arg.fiducials for arg in args]
        yerr_array = [
            np.array(
                [
                    arg.constraints(name=_, marginalized=marginalized)[0]
                    for _ in arg.names
                ]
            )
            for index, arg in enumerate(args)
        ]

        # TODO: maybe put this in a class so validation is easier
        options = {
            "bar": {
                "parameters_scale": "x",
                "values_scale": "y",
                "error_name": "yerr",
                "extra_kwargs": {
                    "capsize": total_space * 60 / len(args) / size,
                    "width": space_per_object,
                },
            },
            "barh": {
                "parameters_scale": "y",
                "values_scale": "x",
                "error_name": "xerr",
                "extra_kwargs": {
                    "capsize": total_space * 60 / len(args) / size,
                    "height": space_per_object,
                },
            },
            "errorbar": {
                "parameters_scale": "x",
                "values_scale": "y",
                "error_name": "yerr",
                "extra_kwargs": {
                    "capsize": total_space * 120 / len(args) / size,
                    "ls": kwargs.pop("ls", ""),
                    "marker": kwargs.pop("marker", "o"),
                },
            },
        }

        if capsize is not None:
            options[kind]["extra_kwargs"]["capsize"] = capsize  # type: ignore

        with plt.rc_context(self.options):
            fig, ax = plt.subplots()
            for color, label, x, y, yerr in zip(
                colors,  # type: ignore
                labels,
                x_array,
                y_array,
                yerr_array,
            ):
                getattr(ax, kind)(
                    x,
                    y,
                    color=color,
                    label=label,
                    **{options[kind]["error_name"]: yerr},
                    **options[kind]["extra_kwargs"],
                    **kwargs,
                )

            ax.set_yscale(scale)

            getattr(ax, f'set_{options[kind]["parameters_scale"]}ticks')(range(size))
            getattr(ax, f'set_{options[kind]["parameters_scale"]}ticklabels')(
                latex_names
            )
            getattr(ax, f'set_{options[kind]["values_scale"]}label')(values_label)

        self._figure = fig
        self._axes = ax

    def plot_relative_constraints(
        self,
        args: Sequence[FisherMatrix],
        kind: str,
        marginalized: bool = True,
        colors: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        values_label: Optional[str] = None,
        scale: str = "linear",
        space: float = 0.3,
        percent: bool = False,
        **kwargs,
    ):
        r"""
        Makes a plot of the constraints (relative to the fiducial) of the
        Fisher matrices

        Parameters
        ----------
        args : array_like of FisherMatrix
            the Fisher matrices for which we want to plot the constraints

        kind : str, {'bar', 'barh', 'errorbar'}
            the kind of plot we want (vertical bar, horizontal bar, errorbar)

        marginalized : bool, optional
            whether the marginalized or the unmarginalized constraints should
            be plotted (default: True)

        colors : array_like of str, optional
            the colors to use for the plotting (default: default matplotlib
            colors)

        labels : array_like of str, optional
            the labels for the Fisher matrices (default: None)

        values_label : str, optional
            the label for the values, that is, the y axis (default: `$\sigma /
            \theta_\mathrm{fid}$` if `percent` is false, otherwise `$\sigma /
            \theta_\mathrm{fid}\ (\%)$`

        scale : str, {'linear', 'log'}
            the scale used for the y axis (default: 'linear')

        space : float, optional
            the space reserved between the bars (default: 0.3)

        percent : bool, optional
            whether to plot in percentage units (default: false)

        **kwargs
            any keyword arguments passed to
            <a
            href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html"
            target="_blank" rel=noreferrer noopener>`matplotlib.pyplot.bar`</a>

        Returns
        -------
        None

        Examples
        --------
        Define some Fisher matrices:
        >>> fm1 = FisherMatrix(np.diag([1, 2, 3]) * 1e3, names=["a", "b", "c"], latex_names=[r'$\Omega_\mathrm{m}$', '$h$', "$n_s$"], fiducials=[0.3, 0.7, 0.96])
        >>> fm2 = FisherMatrix(np.diag([4, 5, 6]) * 1e3, names=["a", "b", "c"], latex_names=[r'$\Omega_\mathrm{m}$', '$h$', "$n_s$"], fiducials=[0.35, 0.67, 0.9])

        Make a vertical bar plot:
        >>> fig = FisherBarFigure()
        >>> fig.plot_relative_constraints([fm1, fm2], 'bar', labels=['Fisher 1', 'Fisher 2'])
        >>> _ = fig.figure.legend()

        Save it to a file:
        >>> fig.savefig(
        ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure_plot_relative_constraints_bar.png",
        ... dpi=150)

        <img width="100%" src="$IMAGE_PATH/fisher_figure_plot_relative_constraints_bar.png">

        Make a horizontal bar plot:
        >>> fig = FisherBarFigure()
        >>> fig.plot_relative_constraints([fm1, fm2], 'barh', labels=['Fisher 1', 'Fisher 2'])
        >>> _ = fig.figure.legend()

        Save it to a file:
        >>> fig.savefig(
        ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure_plot_relative_constraints_barh.png",
        ... dpi=150)

        <img width="100%" src="$IMAGE_PATH/fisher_figure_plot_relative_constraints_barh.png">
        """
        size = len(args[0])
        latex_names = args[0].latex_names

        computed_parameters = self._parse_fractional_constraints(
            args,
            marginalized=marginalized,
            colors=colors,
            labels=labels,
            scale=scale,
            space=space,
            values_label=values_label,
            percent=percent,
        )

        # TODO: maybe put this in a class so validation is easier
        options = {
            "bar": {
                "parameters_scale": "x",
                "values_scale": "y",
                "lines_prefix": "h",
                "extra_kwargs": {
                    "width": computed_parameters.space_per_object,
                },
            },
            "barh": {
                "parameters_scale": "y",
                "values_scale": "x",
                "lines_prefix": "v",
                "extra_kwargs": {
                    "height": computed_parameters.space_per_object,
                },
            },
        }

        with plt.rc_context(self.options):
            fig, ax = plt.subplots()
            for color, label, x, y in zip(
                computed_parameters.colors,
                computed_parameters.labels,
                computed_parameters.x_array,
                computed_parameters.y_array,
            ):
                getattr(ax, kind)(
                    x,
                    y * 100 if percent else y,
                    color=color,
                    label=label,
                    **options[kind]["extra_kwargs"],
                    **kwargs,
                )
                getattr(ax, kind)(
                    x,
                    -(y * 100 if percent else y),
                    color=color,
                    label=None,
                    **options[kind]["extra_kwargs"],
                    **kwargs,
                )

            limits = getattr(ax, f'get_{options[kind]["parameters_scale"]}lim')()

            getattr(ax, f'{options[kind]["lines_prefix"]}lines')(
                0, limits[0], limits[-1], color="black", ls="--"
            )
            getattr(ax, f'set_{options[kind]["parameters_scale"]}lim')(*limits)
            getattr(ax, f'set_{options[kind]["values_scale"]}scale')(
                computed_parameters.scale
            )

            getattr(ax, f'set_{options[kind]["parameters_scale"]}ticks')(range(size))
            getattr(ax, f'set_{options[kind]["parameters_scale"]}ticklabels')(
                latex_names
            )
            getattr(ax, f'set_{options[kind]["values_scale"]}label')(
                computed_parameters.values_label
            )

        self._figure = fig
        self._axes = ax


class FisherFigure1D(_FisherMultipleAxesFigure):
    r"""
    Container for easy access to elements in the 1D plot.

    Examples
    --------
    Define some Fisher matrices:
    >>> m1 = FisherMatrix(
    ... [[3, -2], [-2, 5]],
    ... names=['a', 'b'],
    ... latex_names=[r'$\mathcal{A}$', r'$\mathcal{B}$'],
    ... )
    >>> m2 = FisherMatrix(
    ... [[4, 1], [1, 6]],
    ... names=['a', 'b'],
    ... latex_names=[r'$\mathcal{A}$', r'$\mathcal{B}$'],
    ... )

    Instantiate a figure:
    >>> fig = FisherFigure1D()

    Draw marginalized 1D plots with the previously defined Fisher matrix:
    >>> fig.plot(m1, label='first')

    Add another plot with a different color and style:
    >>> fig.plot(m2, label='second', ls=':', color='green')

    Draw some other stuff on it:
    >>> artist, = fig.draw('a', 'plot', [-1, 0, 1], [-1, 0, 1], color='red', label='curve')
    >>> # alternatively, the below will accomplish the same thing
    >>> # artist, = fig['a'].plot([-1, 0, 1], [-1, 0, 1], color='red')
    >>> # note that in this case, due to certain limitations, you need to
    >>> # manually add the legend using a convenience function:
    >>> # fig.add_artist_to_legend(artist, label='curve')

    Add a legend:
    >>> _ = fig.legend(ncol=3, loc='center')

    Add a title:
    >>> _ = fig.set_title('Example Fisher matrix 1D plotting')

    Save it to a file:
    >>> fig.savefig(
    ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure1d_example1.png",
    ... dpi=150)

    <img width="100%" src="$IMAGE_PATH/fisher_figure1d_example1.png">
    """

    def __init__(
        self,
        options: Optional[dict] = None,
        max_cols: Optional[int] = None,
        hspace: float = 0.5,
        wspace: float = 0.1,
        contour_levels: Optional[Sequence[tuple[float, float]]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        options : dict, optional
            the dictionary containing the options for plotting. If the special
            key 'style' is present, it attempts to use that plotting style (can
            be one of the outputs of <a
            href="https://matplotlib.org/stable/api/style_api.html#matplotlib.style.matplotlib.style.available"
            target="_blank" rel="noopener
            noreferrer">`matplotlib.style.available`</a>, or a path to a file.
            If using a file, does not use the default rc parameters).

        max_cols : int, optional
            the maximum number of columns in the final plot

        hspace : float, optional
            The amount of height reserved for space between subplots, expressed
            as a fraction of the average axis height.

        wspace : float, optional
            The amount of width reserved for space between subplots, expressed
            as a fraction of the average axis width.

        contour_levels : array_like of 2-tuples, optional
            the points at which the sigma level should be shaded, along with
            the associated opacity (default: None). If not specified, defaults
            to `[(1, 0.3), (2, 0.1)]`.

        Notes
        -----
        For the style sheet reference, please consult <a
        href="https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html"
        target="_blank" rel="noreferrer noopener">the matplotlib
        documentation</a>.
        """
        super().__init__(
            options=options,
            wspace=wspace,
            hspace=hspace,
            contour_levels=contour_levels,
        )
        self.max_cols = max_cols
        self._ndim = 1
        self._hspace = hspace
        self._wspace = wspace

    def __getitem__(
        self,
        key: str,
    ):
        """
        Returns the axis associated to the name `key`.
        """
        if not self.figure:
            raise EmptyFigureError

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
        Implements drawing of other objects on the axis where the parameter
        `name` has already been plotted.

        Parameters
        ----------
        name : str
            the name (label) of the parameter where we want to draw

        method : str
            the name of the method which we want to plot (such as <a
            href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html"
            target="_blank" rel="noopener noreferrer">`plot`</a>, <a
            href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html"
            target="_blank" rel="noopener noreferrer">`scatter`</a>, etc.)

        *args
            any positional arguments passed to the method (usually some data)

        **kwargs
            any keyword arguments passed to the method (usually styling options)

        Returns
        -------
        Artist
            the artist that was drawn on the axis

        Raises
        ------
        EmptyFigureError
            if `figure` is not set

        AttributeError
            if `method` is not a valid plotting method
        """
        if not hasattr(self[name], method):
            raise AttributeError(
                f"The method `{method}` is not a valid plotting method"
            )

        # get the right color
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = next(self.current_color)
            self.current_color = iter(self.current_color)

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

        self[name].autoscale()
        self[name].relim()
        self[name].autoscale_view()

        return handles

    def plot(
        self,
        fisher: FisherMatrix,
        *args,
        mark_fiducials: Union[bool, dict] = False,
        **kwargs,
    ):
        """
        Plots the 1D curves of the Fisher objects.

        Parameters
        ----------
        fisher
            the Fisher object which we want to plot

        mark_fiducials : bool or dict, optional
            whether or not the fiducials should be marked on the plots
            (default: False). If set to `True`, uses `linestyles='--'` and
            `colors='black'` as the default. If a dictionary, accepts the same
            keyword arguments as <a
            href="https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.Collection"
            target="_blank" rel="noreferrer
            noopener">`matplotlib.collections.Collection`</a>.
            If an empty dictionary, uses the default style (determined
            automatically).

        **kwargs
            any keyword arguments used for plotting (same as `plot_curve_1d`)

        Returns
        -------
        None
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

        with plt.rc_context(self.options):
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
                    hspace=self.hspace,
                    wspace=self.wspace,
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

                for contour_level in self.contour_levels:
                    _add_shading_1d(
                        fisher.fiducials[index],
                        fisher.constraints(name, marginalized=True),
                        ax,
                        level=contour_level[0],
                        alpha=contour_level[1],
                        **kwargs,
                    )

                ax.autoscale()
                ax.set_xlabel(latex_name)
                ax.relim()
                ax.autoscale_view()

                # the y axis should start at 0 since we're plotting a PDF
                ax.set_ylim(0, ax.get_ylim()[-1])

                ax.set_yticks([])

                # marking the fiducials
                if not mark_fiducials is False:
                    if mark_fiducials is True:
                        mark_fiducials = dict(linestyles="--", colors="black")

                    _mark_fiducial_1d(
                        fisher,
                        fisher.names[index],
                        ax,
                        **mark_fiducials,
                    )

            # remove any axes which are not drawn
            if not full:
                for index in range(
                    size,
                    layout[0] * layout[1],
                ):
                    # it's simpler to turn off the axis, than completely remove
                    # it
                    axes.flat[index].axis("off")

        self._figure = fig
        self._axes = axes
        self._names = fisher.names

    def legend(
        self,
        *args: Artist,
        overwrite: bool = False,
        loc: Union[str, tuple[float, float]] = "lower center",
        bbox_to_anchor: Any = (0.5, 1),
        **kwargs,
    ):
        """
        Creates a legend on the figure.

        Parameters
        ----------
        *args
            any positional arguments for the legend

        overwrite : bool, optional
            whether to overwrite any "nice" options set by `fitk` (default:
            False)

        loc : str or tuple of floats, optional
            which corner of the legend to use as the positioning anchor
            (default: 'lower center')

        bbox_to_anchor : Any, optional
            the location of where to place the legend (default: `[0.5, 1]`)

        **kwargs
            any other keyword arguments for the legend

        Returns
        -------
        Artist
            the legend that was drawn on the figure

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

        with plt.rc_context(self.options):
            ax = self.axes.flat[0]
            if not overwrite:
                if ax.get_legend():
                    ax.get_legend().remove()

                return ax.legend(
                    self.handles,
                    self.labels,
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=self.figure.transFigure,
                    **kwargs,
                )

            return ax.legend(
                *args,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                **kwargs,
            )

    def set_title(
        self,
        *args,
        x: float = 0.5,
        y: float = 1.2,
        **kwargs,
    ):
        """
        Thin wrapper for setting the title of the figure with the correct
        options.

        Parameters
        ----------
        *args
            any positional arguments to `figure.suptitle`

        x : float, optional
            the x position of the title (default: `0.5`)

        y : float, optional
            the y position of the title (default: `1.2`)

        **kwargs
            any keyword arguments to `figure.suptitle`

        Returns
        -------
        Artist
            the title that was drawn on the figure

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

        with plt.rc_context(self.options):
            return self.figure.suptitle(*args, x=x, y=y, **kwargs)


class FisherFigure2D(_FisherMultipleAxesFigure):
    r"""
    Container for easy access to elements in the 2D plot.

    Examples
    --------
    Define some Fisher matrices:
    >>> m1 = FisherMatrix(
    ... [[3, -2], [-2, 5]],
    ... names=['a', 'b'],
    ... latex_names=[r'$\mathcal{A}$', r'$\mathcal{B}$'],
    ... )
    >>> m2 = FisherMatrix(
    ... [[4, 1], [1, 6]],
    ... names=['a', 'b'],
    ... latex_names=[r'$\mathcal{A}$', r'$\mathcal{B}$'],
    ... )

    Instantiate a figure:
    >>> fig = FisherFigure2D(show_1d_curves=True)

    Draw a "triangle plot" with the previously defined Fisher matrix:
    >>> fig.plot(m1, label='first')

    Add a plot with a different color and style:
    >>> fig.plot(m2, label='second', ls=':', color='green')

    Draw some other stuff on it:
    >>> artist, = fig.draw('a', 'b', 'plot', [-1, 0, 1], [-1, 0, 1], color='red', label='curve')
    >>> # alternatively, the below will accomplish the same thing
    >>> # artist, = fig['a', 'b'].plot([-1, 0, 1], [-1, 0, 1], color='red')
    >>> # note that in this case, due to certain limitations, you need to
    >>> # manually add the legend using a convenience function:
    >>> # fig.add_artist_to_legend(artist, label='curve')

    Add a legend:
    >>> _ = fig.legend()

    Add a title:
    >>> _ = fig.set_title('Example Fisher matrix 2D plotting')

    Save it to a file:
    >>> fig.savefig(
    ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure2d_example1.png",
    ... dpi=150)

    <img width="100%" src="$IMAGE_PATH/fisher_figure2d_example1.png">
    """

    def __init__(
        self,
        options: Optional[dict] = None,
        hspace: float = 0,
        wspace: float = 0,
        show_1d_curves: bool = False,
        show_joint_dist: bool = False,
        contour_levels: Optional[Sequence[tuple[float, float]]] = None,
        contour_levels_1d: Optional[Sequence[tuple[float, float]]] = None,
        contour_levels_2d: Optional[Sequence[tuple[float, float]]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        options : dict, optional
            the dictionary containing the options for plotting. If the special
            key 'style' is present, it attempts to use that plotting style (can
            be one of the outputs of <a
            href="https://matplotlib.org/stable/api/style_api.html#matplotlib.style.matplotlib.style.available"
            target="_blank" rel="noopener
            noreferrer">`matplotlib.style.available`</a>, or a path to a file.
            If using a file, does not use the default rc parameters).

        hspace : float, optional
            The amount of height reserved for space between subplots, expressed
            as a fraction of the average axis height.

        wspace : float, optional
            The amount of width reserved for space between subplots, expressed
            as a fraction of the average axis width.

        show_1d_curves : bool, optional
            whether the 1D marginalized curves should be plotted (default:
            False)

        show_joint_dist : bool, optional
            whether to plot the isocontours of the joint distribution (default:
            False)

        contour_levels : array_like of 2-tuples, optional
            the points at which the sigma level should be shaded, along with
            the associated opacity (default: None). If not specified, defaults
            to `[(1, 0.3), (2, 0.1)]`.

        contour_levels_1d : array_like of 2-tuples, optional
            the points at which the sigma level should be shaded (on the 1D
            plots), along with the associated opacity (default: None). If not
            specified, defaults to `[(1, 0.3), (2, 0.1)]`.

        contour_levels_2d : array_like of 2-tuples, optional
            the points at which the sigma level should be shaded (on the 2D
            plots), along with the associated opacity (default: None). If not
            specified, defaults to `[(1, 0.3), (2, 0.1)]`.

        Notes
        -----
        For the style sheet reference, please consult <a
        href="https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html"
        target="_blank" rel="noreferrer noopener">the matplotlib
        documentation</a>.

        **Regarding `show_joint_dist`**: this argument specifies whether we
        wish to plot the p-value of the *joint* distribution, or the p-value of
        the probability of a single parameter lying within the bounds projected
        onto a parameter axis. For more details, see <a
        href="https://arxiv.org/abs/0906.0664" target="_blank" rel="noopener
        noreferrer">arXiv:0906.0664</a>, section 2.
        """
        super().__init__(
            options=options,
            wspace=wspace,
            hspace=hspace,
            contour_levels=contour_levels,
        )
        self.show_1d_curves = show_1d_curves
        self.show_joint_dist = show_joint_dist
        self._ndim = 2

        if contour_levels_1d is None:
            self.contour_levels_1d = self.contour_levels
        else:
            self.contour_levels_1d = _parse_contour_levels(contour_levels_1d)

        if contour_levels_2d is None:
            self.contour_levels_2d = self.contour_levels
        else:
            self.contour_levels_2d = _parse_contour_levels(contour_levels_2d)

    def __getitem__(
        self,
        key: tuple[str, str],
    ):
        if not self.figure:
            raise EmptyFigureError

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

        Parameters
        ----------
        *args
            any positional arguments for the legend

        overwrite : bool, optional
            whether to overwrite any "nice" options set by `fitk` (default:
            False)

        loc : str or tuple of floats, optional
            which corner of the legend to use as the positioning anchor
            (default: 'upper right')

        bbox_to_anchor : Any, optional
            the location of where to place the legend (default: None, picked by
            `fitk` based on information about the plot)

        **kwargs
            any other keyword arguments for the legend

        Returns
        -------
        Artist
            the legend that was drawn on the figure

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

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

                return ax.legend(
                    self.handles,
                    self.labels,
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=self.figure.transFigure,
                    **kwargs,
                )

            return ax.legend(
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

        Parameters
        ----------
        *args
            any positional arguments to `figure.suptitle`

        xscale : float, optional
            the x position of the title (default: `0.5`)

        yscale : float, optional
            the y position of the title (default: `1.08`)

        **kwargs
            any keyword arguments to `figure.suptitle`

        Returns
        -------
        Artist
            the title that was drawn on the figure

        Raises
        ------
        EmptyFigureError
            if `figure` is not set
        """
        if not self.figure:
            raise EmptyFigureError

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

                return self.figure.suptitle(
                    *args,
                    x=x * xscale,
                    y=y * yscale,
                    **kwargs,
                )

            return self.figure.suptitle(*args, **kwargs)

    def draw(
        self,
        name1: str,
        name2: str,
        method: str,
        *args,
        **kwargs,
    ):
        """
        Implements drawing of other objects on the axis where the parameters
        `name1` and `name2` have already been plotted.

        Parameters
        ----------
        name1 : str
            the name (label) of the first parameter where we want to draw

        name2 : str
            the name (label) of the second parameter where we want to draw

        method : str
            the name of the method which we want to plot (such as `plot`,
            `scatter`, etc.)

        *args
            any positional arguments passed to the method (usually some data)

        **kwargs
            any keyword arguments passed to the method (usually styling options)

        Returns
        -------
        Artist
            the artist that was drawn on the axis

        Raises
        ------
        EmptyFigureError
            if `figure` is not set

        AttributeError
            if `method` is not a valid plotting method
        """
        if not hasattr(self[name1, name2], method):
            raise AttributeError(
                f"The method `{method}` is not a valid plotting method"
            )

        # get the right color
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = next(self.current_color)
            self.current_color = iter(self.current_color)

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

        self[name1, name2].autoscale()
        self[name1, name2].relim()
        self[name1, name2].autoscale_view()

        return handles

    def plot(
        self,
        fisher: FisherMatrix,
        *args,
        mark_fiducials: Union[bool, dict] = False,
        **kwargs,
    ):
        r"""
        Plots the 2D contours (and optionally 1D curves) of the Fisher objects.

        Parameters
        ----------
        fisher : FisherMatrix
            the Fisher object which we want to plot

        mark_fiducials : bool or dict, optional
            whether or not the fiducials should be marked on the plots
            (default: False). If set to `True`, uses `linestyles='--'` and
            `colors='black'` as the default. If a dictionary, accepts the same
            keyword arguments as <a
            href="https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.Collection"
            target="_blank" rel="noreferrer
            noopener">`matplotlib.collections.Collection`</a>.
            If an empty dictionary, uses the default style (determined
            automatically).

        **kwargs
            any keyword arguments used for plotting (same as `plot_curve_2d`)

        Returns
        -------
        None

        Examples
        --------
        Define a Fisher matrix:
        >>> m = FisherMatrix([[3, -2], [-2, 5]], names=['a', 'b'], latex_names=[r'$\mathcal{A}$', r'$\mathcal{B}$'])

        Initiate a figure:
        >>> fig = FisherFigure2D(show_1d_curves=True)

        Draw a "triangle plot" with the previously defined Fisher matrix:
        >>> fig.plot(m)

        Save it to a file:
        >>> fig.savefig(
        ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "fisher_figure2d_plot_example.png",
        ... dpi=150)

        <img width="100%" src="$IMAGE_PATH/fisher_figure2d_plot_example.png">
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
                    hspace=self.hspace,
                    wspace=self.wspace,
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
                    try:
                        ax[i, j].remove()
                    except KeyError:
                        pass

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
                        scaling_factor=self.contour_levels_2d[0][0]
                        if not self.show_joint_dist
                        else np.sqrt(_get_chisq(self.contour_levels_2d[0][0])),
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
                        scaling_factor=self.contour_levels_2d[0][0]
                        if not self.show_joint_dist
                        else np.sqrt(_get_chisq(self.contour_levels_2d[0][0])),
                        fill=True,
                        alpha=self.contour_levels_2d[0][1],
                        ec=None,
                        zorder=20,
                        **kwargs,
                    )

                    for index in range(1, len(self.contour_levels_2d)):
                        # the 2-sigma
                        plot_curve_2d(
                            fisher,
                            namex,
                            namey,
                            ax=ax[i, j],
                            scaling_factor=self.contour_levels_2d[index][0]
                            if not self.show_joint_dist
                            else np.sqrt(_get_chisq(self.contour_levels_2d[index][0])),
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
                            scaling_factor=self.contour_levels_2d[index][0]
                            if not self.show_joint_dist
                            else np.sqrt(_get_chisq(self.contour_levels_2d[index][0])),
                            fill=True,
                            alpha=self.contour_levels_2d[index][1],
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

                        for contour_level in self.contour_levels_1d:
                            _add_shading_1d(
                                fisher.fiducials[i],
                                fisher.constraints(
                                    name=namex,
                                    marginalized=True,
                                )[0],
                                level=contour_level[0],
                                ax=ax[i, i],
                                alpha=contour_level[1],
                                **kwargs,
                            )

                    else:
                        if self.figure is None:
                            try:
                                ax[i, i].remove()
                            except KeyError:
                                pass

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

                        # marking the fiducials (1D)
                        if mark_fiducials is not False:
                            if mark_fiducials is True:
                                mark_fiducials = dict(linestyles="--", colors="black")

                            _mark_fiducial_1d(
                                fisher,
                                fisher.names[i],
                                ax[i, i],
                                **mark_fiducials,
                            )

                    else:
                        # marking the fiducials (2D)
                        if mark_fiducials is not False:
                            if mark_fiducials is True:
                                mark_fiducials = dict(linestyles="--", colors="black")

                            _mark_fiducial_2d(
                                fisher,
                                fisher.names[i],
                                fisher.names[j],
                                ax[i, j],
                                **mark_fiducials,
                            )

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

    # keyword args which have no effect on the plotting
    unused_kwargs = [
        "lw",
        "linewidth",
        "ec",
        "edgecolor",
        "fc",
        "facecolor",
    ]
    for kwarg in unused_kwargs:
        kwargs.pop(kwarg, None)

    # the kwarg `color` supersedes `facecolor`, hence we need to remove it from
    # the passed kwargs, and call `fill_between` with `facecolor=color`
    color = kwargs.pop("color", None)

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
        edgecolor=None,
        facecolor=color,
        **kwargs,
    )

    return (ax.get_figure(), ax)


def plot_curve_1d(
    fisher: FisherMatrix,
    name: str,
    ax: Optional[Axes] = None,
    **kwargs,
) -> tuple[Figure, Axes, Artist]:
    r"""
    Plots a 1D curve (usually marginalized Gaussian) from a Fisher object.

    Parameters
    ----------
    fisher : FisherMatrix
        the Fisher matrix which we want to plot

    name : str
        the name of the parameter which we want to plot

    ax : matplotlib.axes.Axes, optional
        the axis on which we want to plot the contour (default: None). If None,
        plots to a new figure.

    **kwargs
        any keyword arguments passed to <a
        href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html"
        target="_blank" rel="noopener noreferrer">`matplotlib.pyplot.plot`</a>.

    Returns
    -------
    A 3-tuple `(Figure, Axes, Artist)`
        The figure and the axis on which the artist was drawn on, as well as
        the artist itself.

    Examples
    --------
    Define a Fisher matrix:
    >>> m = FisherMatrix(
    ... [[3, -2], [-2, 5]],
    ... names=['a', 'b'],
    ... latex_names=[r'$\mathcal{A}$', r'$\mathcal{B}$'],
    ... )

    Draw the marginalized parameter `a`:
    >>> fig, ax, artist = plot_curve_1d(m, name='a', ls='--', color='red')

    Save it to a file:
    >>> fig.savefig(
    ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "plot_curve_1d_example1.png",
    ... dpi=100)

    <img width="100%" src="$IMAGE_PATH/plot_curve_1d_example1.png">
    """
    if ax is None:
        _, ax = plt.subplots()

    fid = fisher.fiducials[np.where(fisher.names == name)]
    sigma = fisher.constraints(name, marginalized=True)

    return _add_plot_1d(fid, sigma, ax=ax, **kwargs)


def plot_curve_2d(
    fisher: FisherMatrix,
    name1: str,
    name2: str,
    ax: Optional[Axes] = None,
    scaling_factor: float = 1,
    **kwargs,
) -> tuple[Figure, Axes, Artist]:
    r"""
    Plots a 2D curve (usually ellipse) from two parameters of a Fisher object.

    Parameters
    ----------
    fisher : FisherMatrix
        the Fisher matrix which we want to plot

    name1 : str
        the name of the first parameter which we want to plot

    name2 : str
        the name of the second parameter which we want to plot

    ax : matplotlib.axes.Axes, optional
        the axis on which we want to plot the contour (default: None). If None,
        plots to a new figure.

    **kwargs
        any keyword arguments passed to <a
        href="https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Ellipse.html"
        target="_blank" rel="noopener
        noreferrer">`matplotlib.patches.Ellipse`</a>.

    Returns
    -------
    A 3-tuple `(Figure, Axes, Artist)`
        The figure and the axis on which the artist was drawn on, as well as
        the artist itself.

    Examples
    --------
    Define a Fisher matrix:
    >>> m = FisherMatrix(
    ... [[3, -2], [-2, 5]],
    ... names=['a', 'b'],
    ... fiducials=[1, -2],
    ... latex_names=[r'$\mathcal{A}$', r'$\mathcal{B}$'],
    ... )

    Draw the 1$\sigma$ contour of the parameter combination `a`, `b`:
    >>> fig, ax, artist = plot_curve_2d(m, name1='a', name2='b', ls='--', color='red', fill=False)

    Rescale the view a bit:
    >>> _ = ax.autoscale()
    >>> _ = ax.relim()
    >>> _ = ax.autoscale_view()

    Save it to a file:
    >>> fig.savefig(
    ... Path(__file__).parent.parent / os.environ.get("TEMP_IMAGE_DIR", "") / "plot_curve_2d_example1.png",
    ... dpi=100)

    <img width="100%" src="$IMAGE_PATH/plot_curve_2d_example1.png">

    Notes
    -----
    The limits of the plot are not updated automatically.
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


def _mark_fiducial_1d(
    fisher: FisherMatrix,
    name: str,
    ax: Axes,
    npoints: int = 100,
    autolim: bool = False,
    **kwargs,
):
    # always put the fiducials on top (unless specified otherwise)
    zorder = kwargs.pop("zorder", 999)

    # only need to draw a vertical line on 1D plots
    x_vline = np.repeat(
        fisher.fiducials[np.where(name == fisher.names)][0],
        npoints,
    )

    y_vline = np.linspace(0, 1e5, npoints)

    vline = LineCollection(
        [np.column_stack((x_vline, y_vline))],
        zorder=zorder,
        **kwargs,
    )

    ax.add_collection(vline, autolim=autolim)


def _mark_fiducial_2d(
    fisher: FisherMatrix,
    name1: str,
    name2: str,
    ax: Axes,
    npoints: int = 100,
    autolim: bool = False,
    **kwargs,
):
    # always put the fiducials on top (unless specified otherwise)
    zorder = kwargs.pop("zorder", 999)

    index1 = np.where(fisher.names == name1)[0]
    index2 = np.where(fisher.names == name2)[0]
    fiducial1 = fisher.fiducials[index1]
    fiducial2 = fisher.fiducials[index2]

    # drawing the horizontal line
    x_hline = np.linspace(
        fiducial2 - fisher.constraints(name2, sigma=100)[0],
        fiducial2 + fisher.constraints(name2, sigma=100)[0],
        npoints,
    )
    y_hline = np.repeat(fiducial1, npoints)

    hline = LineCollection(
        [np.column_stack((x_hline, y_hline))],
        zorder=zorder,
        **kwargs,
    )

    ax.add_collection(hline, autolim=autolim)

    # drawing the vertical line
    x_vline = np.repeat(fiducial2, npoints)

    y_vline = np.linspace(
        fiducial1 - fisher.constraints(name1, sigma=100)[0],
        fiducial1 + fisher.constraints(name1, sigma=100)[0],
        npoints,
    )

    vline = LineCollection(
        [np.column_stack((x_vline, y_vline))],
        zorder=zorder,
        **kwargs,
    )

    ax.add_collection(vline, autolim=autolim)


def _parse_contour_levels(contour_levels: Sequence[tuple[float, float]]):
    try:
        contour_levels = [
            (float(level), float(alpha)) for level, alpha in contour_levels
        ]
    except Exception as exc:
        raise ValueError(
            f"The object {contour_levels} cannot be converted to a list of 2-tuples"
        ) from exc
    for level, alpha in contour_levels:
        if level < 0:
            raise ValueError(f"Negative contour level found in: {contour_levels}")
        if not 0 <= alpha <= 1:
            raise ValueError(
                f"Opacity value outside of (0, 1) found in: {contour_levels}"
            )

    return contour_levels
