"""
Package for plotting of Fisher objects.
See here for documentation of `FisherPlotter`, `FisherFigure1D`, and `FisherFigure2D`.
"""

from __future__ import annotations

# standard library imports
from abc import ABC, abstractmethod
from typing import \
    AnyStr, \
    Iterable, \
    List, \
    Optional, \
    SupportsFloat, \
    Tuple, \
    Union

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats import chi2, norm

# first party imports
from .fisher_utils import \
    MismatchingSizeError, \
    ParameterNotFoundError, \
    is_iterable, \
    get_index_of_other_array, \
    float_to_latex
from .fisher_matrix import FisherMatrix



class FisherBaseFigure(ABC):
    def __init__(
        self,
        figure : Figure,
        axes : Iterable[Axes],
        names : Iterable[AnyStr],
    ):
        """
        Constructor.

        Parameters
        ----------
        figure : Figure
            the figure plotted by `FisherPlotter`

        axes : Iterable[Axes]
            the axes of the above figure

        names : Iterable[AnyStr]
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



class FisherFigure1D(FisherBaseFigure):
    """
    Container for easy access to elements in the 1D plot.
    """
    def __getitem__(
        self,
        key : AnyStr,
    ):
        """
        Returns the axis associated to the name `key`.
        """
        if key not in self.names:
            raise ParameterNotFoundError(key, self.names)

        return self.axes[np.where(self.names == key)][0]



class FisherFigure2D(FisherBaseFigure):
    """
    Container for easy access to elements in the 2D plot.
    """
    def __getitem__(
        self,
        key : Tuple[AnyStr],
    ):
        pass



class FisherPlotter:
    """
    Class for plotting FisherMatrix objects.
    """
    def __init__(
        self,
        *args : FisherMatrix,
        labels : Optional[Iterable[AnyStr]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        args : FisherMatrix
            `FisherMatrix` objects which we want to plot.
            Must have the same names, otherwise throws an error.
            Can have different fiducial values.
            The order of plotting of the parameters and the LaTeX labels to use
            are determined by the first argument.

        labels : array of strings, default None
            the list of labels to put in the legend of the plots.
            If not set, defaults to `0, ..., len(args) - 1`
        """
        # make sure all of the Fisher objects have the same sizes
        if not all(len(args[0]) == len(arg) for arg in args):
            raise MismatchingSizeError(*args)

        # check names
        if not all(set(args[0].names) == set(arg.names) for arg in args):
            raise ValueError(
                'The names of the inputs do not match'
            )

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


    def plot_1d(
        self,
        rc : dict = {},
        **kwargs,
    ):
        """
        Makes a 1D plot (Gaussians) of the Fisher objects and returns a
        `FisherFigure1D`.

        Parameters
        ----------
        rc : dict = {}
            any parameters meant for `matplotlib.rcParams`.
            See [Matplotlib documentation](https://matplotlib.org/stable/tutorials/introductory/customizing.html) for more information.

        Returns
        -------
        An instance of `FisherFigure1D`.
        """
        size = len(self.values[0])
        with plt.rc_context(rc):
            # general figure setup
            fig = plt.figure(clear=True, figsize=(2 * size, 2))
            gs = fig.add_gridspec(nrows=1, ncols=size, hspace=0.2, wspace=0.1)
            axes = gs.subplots()

            names = self.values[0].names
            names_latex = self.values[0].names_latex

            ylabel1d = r'$P (\theta)$'

            handles = []

            for (index, name), name_latex in zip(enumerate(names), names_latex):
                ax = axes[index]
                sigmas = np.array(
                    [_.constraints(name, marginalized=True, sigma=3) for _ in self.values]
                )
                fiducial = np.array(
                    [_.fiducial[np.where(_.names == name)] for _ in self.values]
                )
                title_list = [
                    '{0} = ${1}^{{+{2}}}_{{-{2}}}$'.format(
                    name_latex,
                    float_to_latex(float(_.fiducial[np.where(_.names == name)])),
                    float_to_latex(float(_.constraints(name, marginalized=True))),
                ) for _ in self.values
                ]

                xleft, xright = np.min(fiducial - sigmas), np.max(fiducial + sigmas)

                ymax = np.max(
                    [gaussian(0, 0, _.constraints(name, marginalized=True)) for _ in self.values]
                )

                for fm in self.values:
                    handle, = add_plot_1d(
                        fm.fiducial[np.where(fm.names == name)],
                        fm.constraints(name, marginalized=True),
                        ax,
                    )

                    if index == 0:
                        handles.append(handle)

                ax.set_xlabel(name_latex)
                ax.set_xlim(xleft, xright)
                ax.set_ylim(0, ymax)

                if kwargs.get('title') is True:
                    ax.set_title('\n'.join(title_list))

                if index == 0:
                    ax.set_ylabel(ylabel1d)

                else:
                    ax.set_yticks([])

            if kwargs.get('legend') is True:
                fig.legend(
                    np.array(handles, dtype=object),
                    self.labels,
                    frameon=False,
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=len(self.values),
                )

        return FisherFigure1D(fig, axes, names)


    def plot_2d(
        self,
        **kwargs,
    ):
        """
        Plots the 2D ellipses of the Fisher objects, and returns an instance of
        `FisherFigure2D`.
        """
        pass


    def plot_triangle(
        self,
        **kwargs,
    ):
        """
        Plots the 2D ellipses and 1D Gaussians of the Fisher objects, and
        returns an instance of `FisherFigure2D`.
        """
        pass



def gaussian(
    x : float,
    mu : float = 0,
    sigma : float = 1,
):
    """
    Returns a normalized Gaussian.
    """
    if sigma <= 0:
        raise ValueError(
            f'Invalid parameter: sigma = {sigma}'
        )

    return np.exp(-(x - mu)**2 / 2 / sigma**2) / sigma / np.sqrt(2 * np.pi)



def get_chisq(
    sigma : float = 1,
    df : int = 2,
):
    r"""
    Returns \(\Delta \chi^2\).
    To obtain the scaling coefficient \(\alpha\), just take the square root of the output.
    """
    return chi2.ppf(norm.cdf(sigma) - norm.cdf(-sigma), df=df)



def add_plot_1d(
    fiducial : float,
    sigma : float,
    ax : Axes,
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
