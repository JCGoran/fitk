"""
Package for plotting of Fisher objects.
See here for documentation of `FisherPlotter`, `FisherFigure1D`, and `FisherFigure2D`.
"""

from __future__ import annotations

# standard library imports
from abc import ABC, abstractmethod
from typing import \
    Collection, \
    Optional, \
    Tuple, \
    Union

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats import chi2, norm

# first party imports
from .fisher_utils import \
    get_default_rcparams, \
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
        axes : Collection[Axes],
        names : Collection[str],
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
        path : str,
        dpi : float = 300,
        bbox_inches : Union[str, Bbox] = 'tight',
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
        key : str,
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
        key : Tuple[str],
    ):
        pass



class FisherPlotter:
    """
    Class for plotting FisherMatrix objects.
    """
    def __init__(
        self,
        *args : FisherMatrix,
        labels : Optional[Collection[str]] = None,
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


    def find_limits_1d(
        self,
        name : str,
        sigma : float = 3,
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
        fiducial = np.array(
            [_.fiducial[np.where(_.names == name)] for _ in self.values]
        )

        xleft, xright = np.min(fiducial - sigmas), np.max(fiducial + sigmas)

        return xleft, xright


    def plot_1d(
        self,
        max_cols : Optional[int] = None,
        rc : dict = get_default_rcparams(),
        **kwargs,
    ):
        """
        Makes a 1D plot (Gaussians) of the Fisher objects and returns a
        `FisherFigure1D`.

        Parameters
        ----------
        max_cols : Optional[int], default = None
            the maximum number of columns to force the plot into.
            By default, the parameters are always plotted horizontally; if you
            need to spread it over `max_cols`, pass a non-negative integer
            here.

        rc : dict = get_default_rcparams()
            any parameters meant for `matplotlib.rcParams`.
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

        with plt.rc_context(rc):
            # general figure setup
            fig = plt.figure(clear=True, figsize=(2 * layout[1], 2 * layout[0]))
            gs = fig.add_gridspec(
                nrows=layout[0], ncols=layout[1],
                hspace=0.5, wspace=0.1,
            )
            axes = gs.subplots()
            if size == 1:
                axes = np.array([axes])

            names = self.values[0].names
            latex_names = self.values[0].latex_names

            ylabel1d = r'$p (\theta)$'

            handles = []

            for (index, name), name_latex in zip(enumerate(names), latex_names):
                ax = axes.flat[index]
                title_list = [
                    '{0} = ${1}^{{+{2}}}_{{-{2}}}$'.format(
                    name_latex,
                    float_to_latex(float(_.fiducial[np.where(_.names == name)])),
                    float_to_latex(float(_.constraints(name, marginalized=True))),
                ) for _ in self.values
                ]

                # the scaling factor here is so that we don't cutoff the peak
                ymax = np.max(
                    [gaussian(0, 0, _.constraints(name, marginalized=True)) for _ in self.values]
                ) * 1.03

                for fm in self.values:
                    handle, = add_plot_1d(
                        fm.fiducial[np.where(fm.names == name)],
                        fm.constraints(name, marginalized=True),
                        ax,
                    )

                    if index == 0:
                        handles.append(handle)

                ax.set_xlabel(name_latex)
                ax.set_xlim(*self.find_limits_1d(name))
                ax.set_ylim(0, ymax)

                if kwargs.get('title') is True:
                    ax.set_title('\n'.join(title_list))

                if index == 0:
                    ax.set_ylabel(ylabel1d)

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

            if isinstance(kwargs.get('title'), str):
                fig.suptitle(kwargs.get('title'))

            # remove any axes which are not drawn
            if not full:
                for index in range(
                    (layout[0] - 1) * layout[1] + 1,
                    layout[0] * layout[1]
                ):
                    axes.flat[index].remove()

        return FisherFigure1D(fig, axes, names)


    def plot_2d(
        self,
        rc : dict = get_default_rcparams(),
        **kwargs,
    ):
        """
        Plots the 2D ellipses (and optionally 1D Gaussians) of the Fisher
        objects, and returns an instance of `FisherFigure2D`.
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
