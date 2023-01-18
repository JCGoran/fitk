"""
Various interfaces for computing derivatives w.r.t. parameters using the COFFE
code
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from typing import Optional

# third party imports
import numpy as np
from scipy.linalg import block_diag

try:
    import coffe
except ImportError:
    IMPORT_SUCCESS = False
else:
    IMPORT_SUCCESS = True

# first party imports
from fitk.derivatives import FisherDerivative


def _parse_and_set_args(
    **kwargs,
):
    """
    Parses kwargs for COFFE, and returns an instance of it.
    """
    # these can't be set using `setattr(<instance>, <name>, <value>)`,
    # so we need to process them separately
    special_kwargs = (
        "galaxy_bias1",
        "galaxy_bias2",
        "magnification_bias1",
        "magnification_bias2",
    )

    bias = {}
    for key, value in kwargs.items():
        if key in special_kwargs:
            bias[key] = value

    result = coffe.Coffe(
        **{key: value for key, value in kwargs.items() if key not in special_kwargs}
    )

    if bias:
        for key, value in bias.items():
            # basically equivalent to:
            # <instance>.set_<thing>_bias{1, 2}(x_array, y_array)
            getattr(result, f"set_{key}")(value[0], value[1])

    return result


class CoffeMultipolesDerivative(FisherDerivative):
    r"""
    Computes the derivative of the multipoles of the 2PCF w.r.t. parameters

    Examples
    --------
    Import the necessary modules:
    >>> from fitk import D

    Set some cosmology:
    >>> cosmo = CoffeMultipolesDerivative(
    ... config=dict(omega_m=0.32, sep=[10, 20, 30], l=[0], pixelsize=[5],
    ... number_density1=[1e-3], number_density2=[1e-3], fsky=[0.3]))

    Compute the derivative of the signal (multipoles of 2PCF) w.r.t. $h$ with a
    fiducial value of $0.67$ and an absolute step size $10^{-3}$:
    >>> cosmo.derivative('signal', D('h', 0.67, 1e-3))

    Compute the Fisher matrix with $\Omega_\mathrm{m}$ and $n_s$ as the
    parameters:
    >>> fm = cosmo.fisher_matrix(
    ... D(name='omega_m', fiducial=0.32, abs_step=1e-3),
    ... D(name='n_s', fiducial=0.96, abs_step=1e-3))
    """

    __software_name__ = "coffe"
    __url__ = "https://github.com/JCGoran/coffe"
    __version__ = "3.0.0"
    __maintainers__ = ["Goran Jelic-Cizmek <goran.jelic-cizmek@unige.ch>"]
    __imported__ = IMPORT_SUCCESS

    def __init__(
        self,
        *args,
        config: Optional[dict] = None,
        **kwargs,
    ):
        """
        The constructor for the class.

        Parameters
        ----------
        config : dict, optional
            the configuration for COFFE, as a dictionary (default: default
            COFFE configuration)
        """
        if not self.__imported__:
            raise ImportError(
                f"Unable to import the `{self.__software_name__}` module, "
                "please make sure it is installed; "
                f"for additional help, please consult the following URL: {self.__url__}"
            )

        self._config = config if config is not None else {}
        super().__init__(*args, **kwargs)

    @property
    def config(self):
        """
        Returns the current COFFE configuration as a dictionary.
        """
        return self._config

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Computes the multipoles of the 2PCF with some cosmology.

        Returns
        -------
        array_like : float
            the signal as a numpy array

        Notes
        -----
        The coordinates used are $(r, \ell, \bar{z})$, in that increasing order.
        The size of the output is $\text{size}(r) \times \text{size}(\ell)
        \times \text{size}(\bar{z})$.

        For more details on the exact theoetical modelling used, see
        [arXiv:1806.11090](https://arxiv.org/abs/1806.11090), section 2.
        """
        cosmo = _parse_and_set_args(**self.config)
        for arg, value in args:
            setattr(cosmo, arg, value)

        return np.array(
            [_.value for _ in cosmo.compute_multipoles_bulk()],
        )

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Computes the covariance of the multipoles of the 2PCF with some
        cosmology.

        Returns
        -------
        array_like : float
            the covariance matrix as a numpy array

        Notes
        -----
        The covariance does not take into account cross-correlations between
        the different redshifts, i.e. for $n$ redshift bins it has the form:
        $$
            \begin{pmatrix}
            \mathsf{C}(\bar{z}_1) & 0 & \ldots & 0\\\
            0 & \mathsf{C}(\bar{z}_2) & \ldots & 0\\\
            \vdots & \vdots & \ddots & \vdots\\\
            0 & 0 & \ldots & \mathsf{C}(\bar{z}_n)
            \end{pmatrix}
        $$

        For more details on the exact theoretical modelling used, see
        [arXiv:1806.11090](https://arxiv.org/abs/1806.11090), section 2.
        """
        cosmo = _parse_and_set_args(**self.config)
        for arg, value in args:
            setattr(cosmo, arg, value)

        covariance = cosmo.compute_covariance_bulk()

        result = block_diag(
            *[
                np.reshape(
                    [
                        _.value
                        for _ in covariance[
                            i : i + len(cosmo.sep) ** 2 * len(cosmo.l) ** 2
                        ]
                    ],
                    (len(cosmo.sep) * len(cosmo.l), len(cosmo.sep) * len(cosmo.l)),
                )
                for i in range(
                    0,
                    len(covariance),
                    len(cosmo.sep) ** 2 * len(cosmo.l) ** 2,
                )
            ]
        )

        return result

    @property
    def __credits__(self):
        return (
            f"Software: {self.__software_name__}\n"
            f"Version: {self.__version__}\n"
            f"URL: {self.__url__}\n"
            f"Interface maintainer(s): {self.__maintainers__}"
        )
