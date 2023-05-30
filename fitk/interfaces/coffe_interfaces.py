"""
COFFE interfaces for FITK.

Module containing various interfaces for computing derivatives w.r.t.
parameters using the COFFE code.

##### Notes

For consistency reasons, when computing derivatives or Fisher matrices, you
should explicitly pass to `config` the fiducial values of the parameters for
which you want to compute those quantities. In other words:

The wrong way
```python
cosmo = CoffeMultipolesDerivative()
derivative = cosmo.derivative('signal', D(P('omega_m', 0.3), 1e-3))
```

The right way
```python
cosmo = CoffeMultipolesDerivative(config={'omega_m' : 0.3})
derivative = cosmo.derivative('signal', D(P('omega_m', 0.3), 1e-3))
```
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
import re
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Sequence

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
from fitk.utilities import P


def _parse_and_set_args(
    **kwargs,
):
    """Parse kwargs for COFFE, and returns an instance of it."""
    # these can't be set using `setattr(<instance>, <name>, <value>)`,
    # so we need to process them separately
    special_kwargs = (
        "galaxy_bias1",
        "galaxy_bias2",
        "magnification_bias1",
        "magnification_bias2",
        "evolution_bias1",
        "evolution_bias2",
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


class CoffeBaseDerivative(ABC, FisherDerivative):
    r"""Base class for all COFFE interfaces."""

    software_names = "coffe"
    urls = dict(
        github="https://github.com/JCGoran/coffe",
        pypi="https://pypi.org/project/coffe/",
    )
    version = "1.0.0"
    authors = [
        dict(
            name="Goran Jelic-Cizmek",
            email="goran.jelic-cizmek@unige.ch>",
        )
    ]
    __imported__ = IMPORT_SUCCESS

    def __init__(
        self,
        *args,
        config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Create an instance.

        Parameters
        ----------
        config : dict, optional
            the configuration for COFFE, as a dictionary (default: default
            COFFE configuration)
        """
        if not self.__imported__:
            raise ImportError(
                f"Unable to import the `{self.software_names}` module, "
                "please make sure it is installed; "
                f"for additional help, please consult one of the following URL(s): {self.urls}"
            )

        self._config = config if config is not None else {}
        super().__init__(*args, **kwargs)

    @property
    def config(self):
        """Return the current COFFE configuration as a dictionary."""
        return self._config


class CoffeMultipolesDerivative(CoffeBaseDerivative):
    r"""
    Class for computing the derivatives of the multipoles of the 2PCF w.r.t. cosmological parameters.

    Examples
    --------
    Import the necessary modules:

    >>> from fitk import P, D

    Define some parameters:

    >>> parameters = [P('omega_m', 0.32), P('n_s', 0.96), P('h', 0.67)]

    Set a cosmology consistent with those parameters:

    >>> cosmo = CoffeMultipolesDerivative(config={**{_.name : _.fiducial for _ in parameters}})

    Compute the first derivative of the signal (multipoles of 2PCF), using a
    fourth-order central derivative scheme, w.r.t. :math:`h` with a fiducial value of
    :math:`0.67` and an absolute step size :math:`10^{-3}`:

    >>> derivative = cosmo.derivative('signal', D(parameters[-1], 1e-3))

    Compute the Fisher matrix with :math:`\Omega_\mathrm{m}` and :math:`n_s` as the
    parameters:

    >>> fm = cosmo.fisher_matrix(D(parameters[0], abs_step=1e-3), D(parameters[1], abs_step=1e-3))
    """

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the multipoles of the 2PCF with some cosmology.

        Returns
        -------
        array_like : float
            the signal as a numpy array

        Notes
        -----
        The coordinates used are :math:`(r, \ell, \bar{z})`, in that increasing order.
        The size of the output is :math:`\text{size}(r) \times \text{size}(\ell) \times \text{size}(\bar{z})`.

        For more details on the exact theoetical modelling used, see <a
        href="https://arxiv.org/abs/1806.11090" target="_blank" rel="noopener
        noreferrer">arXiv:1806.11090</a>, section 2.
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
        Compute the covariance of the multipoles of the 2PCF with some cosmology.

        Returns
        -------
        array_like : float
            the covariance matrix as a numpy array

        Notes
        -----
        The covariance does not take into account cross-correlations between
        the different redshifts, i.e. for :math:`n` redshift bins it has the form:

        .. math::
            \begin{pmatrix}
            \mathsf{C}(\bar{z}_1) & 0 & \ldots & 0\\\
            0 & \mathsf{C}(\bar{z}_2) & \ldots & 0\\\
            \vdots & \vdots & \ddots & \vdots\\\
            0 & 0 & \ldots & \mathsf{C}(\bar{z}_n)
            \end{pmatrix}

        For more details on the exact theoetical modelling used, see <a
        href="https://arxiv.org/abs/1806.11090" target="_blank" rel="noopener
        noreferrer">arXiv:1806.11090</a>, section 2.
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


class CoffeAverageMultipolesDerivative(CoffeBaseDerivative):
    r"""
    Class for computing the derivatives of the redshift-averaged multipoles of the 2PCF w.r.t. cosmological parameters.

    .. versionadded:: 0.9.2
    """

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the redshift-averaged multipoles of the 2PCF for some cosmology.

        Returns
        -------
        array_like : float
            the signal as a numpy array

        Notes
        -----
        The coordinates used are :math:`(r, \ell, z_\mathrm{min}, z_\mathrm{max})`,
        in that increasing order. The size of the output is :math:`\text{size}(r) \times \text{size}(\ell) \times \text{size}(z_\mathrm{min})`.

        For more details on the exact theoetical modelling used, see <a
        href="https://arxiv.org/abs/1806.11090" target="_blank" rel="noopener
        noreferrer">arXiv:1806.11090</a>, section 2.
        """
        cosmo = _parse_and_set_args(**self.config)
        for arg, value in args:
            setattr(cosmo, arg, value)

        return np.array(
            [_.value for _ in cosmo.compute_average_multipoles_bulk()],
        )

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the covariance of the redshift-averaged multipoles of the 2PCF for some cosmology.

        Returns
        -------
        array_like : float
            the covariance matrix as a numpy array

        Notes
        -----
        The covariance does not take into account cross-correlations between
        the different redshifts, i.e. for :math:`n` redshift bins it has the form:

        .. math::
            \begin{pmatrix}
            \mathsf{C}(\bar{z}_1) & 0 & \ldots & 0\\\
            0 & \mathsf{C}(\bar{z}_2) & \ldots & 0\\\
            \vdots & \vdots & \ddots & \vdots\\\
            0 & 0 & \ldots & \mathsf{C}(\bar{z}_n)
            \end{pmatrix}

        For more details on the exact theoetical modelling used, see <a
        href="https://arxiv.org/abs/1806.11090" target="_blank" rel="noopener
        noreferrer">arXiv:1806.11090</a>, section 2.
        """
        cosmo = _parse_and_set_args(**self.config)
        for arg, value in args:
            setattr(cosmo, arg, value)

        covariance = cosmo.compute_average_covariance_bulk()

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


class CoffeMultipolesTildeDerivative(CoffeMultipolesDerivative):
    r"""
    Class for computing the derivatives of the 2PCF w.r.t. :math:`\tilde{f}` and :math:`\tilde{b}` parameters.

    The parameters are defined as:

    .. math::
        \tilde{f}(z) = f(z) \sigma_8(z),
        \quad
        \tilde{b}(z) = b(z) \sigma_8(z)

    where :math:`f(z)` is the growth rate of matter, :math:`b(z)` is the linear galaxy
    bias, and :math:`\sigma_8(z)` is the size of the redshift-dependent matter
    perturbation at 8 :math:`\mathrm{Mpc}/h`.

    The valid parameters are:

    * :math:`\tilde{f}` - ``f[N]``, the parameter :math:`\sigma_8(z) D_1(z)` in a redshift
        bin, where ``[N]`` is a number from 1 to the number of redshift bins
    * :math:`\tilde{b}` - ``b[N]``, the parameter :math:`b(z) D_1(z)` in a redshift bin,
        where ``[N]`` is a number from 1 to the number of redshift bins

    Notes
    -----
    Just like for ``CoffeMultipolesDerivative``, one can set the configuration in
    the constructor by specifying the ``config`` argument.

    Only density and RSD effects are taken into account when computing the
    signal.

    The covariance is assumed to be parameter-independent.

    If the biases are set for two populations, the second population is ignored.
    """

    def validate_parameter(self, arg) -> bool:
        if re.search(r"^[bf][1-9][0-9]*$", arg.name) and int(arg.name[1:]) in range(
            1, len(self.config["z_mean"]) + 1
        ):
            return True
        return False

    def covariance(
        self,
        *args,
        **kwargs,
    ):
        return super().covariance()

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        cosmo = _parse_and_set_args(**self.config)

        def monopole(b: float, f: float):
            return (
                (b**2 + 2 * b * f / 3 + f**2 / 5)
                * np.array([cosmo.integral(r=_, l=0, n=0) for _ in cosmo.sep])
                / cosmo.sigma8**2
            )

        def quadrupole(b: float, f: float):
            return (
                -(4 * b * f / 3 + 4 * f**2 / 7)
                * np.array([cosmo.integral(r=_, l=2, n=0) for _ in cosmo.sep])
                / cosmo.sigma8**2
            )

        def hexadecapole(b: float, f: float):
            return (
                (8 * f**2 / 35)
                * np.array([cosmo.integral(r=_, l=4, n=0) for _ in cosmo.sep])
                / cosmo.sigma8**2
            )

        multipoles = {0: monopole, 2: quadrupole, 4: hexadecapole}

        b_array = np.array(
            [
                cosmo.galaxy_bias1(_) * cosmo.growth_factor(_) * cosmo.sigma8
                for _ in cosmo.z_mean
            ]
        )
        f_array = np.array(
            [
                cosmo.growth_rate(_) * cosmo.growth_factor(_) * cosmo.sigma8
                for _ in cosmo.z_mean
            ]
        )

        for arg in args:
            name, value = arg
            if name[0] == "b":
                b_array[int(name[1:]) - 1] = value
            elif name[0] == "f":
                f_array[int(name[1:]) - 1] = value

        return np.array(
            [
                np.array(
                    [
                        [
                            item(b, f)
                            for key, item in multipoles.items()
                            if key in cosmo.l
                        ]
                        for b, f in zip(b_array, f_array)
                    ]
                )
            ]
        ).flatten()


@dataclass
class _BiasParameter:
    longname: str
    shortname: str
    values: Optional[Sequence[float]] = None


class CoffeMultipolesBiasDerivative(CoffeMultipolesDerivative):
    r"""
    Class for computing the derivatives of the 2PCF w.r.t. biases.

    Class for computing the derivatives of the 2PCF w.r.t. the magnification,
    and evolution bias parameters in each redshift bin.

    The valid parameters are:

    * :math:`b_n` - ``b[N]``, the galaxy bias in a redshift bin, where ``[N]`` is a
        number from 1 to the number of redshift bins
    * :math:`s_n` - ``s[N]``, the magnification bias in a redshift bin, where ``[N]`` is
        a number from 1 to the number of redshift bins
    * :math:`e_n` - ``e[N]``, the evolution bias in a redshift bin, where ``[N]`` is a
        number from 1 to the number of redshift bins

    .. versionchanged:: 0.9.2
        Added the ability to compute derivatives w.r.t. the evolution bias

    Warns
    -----
    UserWarning
        if ``has_density=True`` or ``has_rsd=True`` and ``has_flatsky_local=False``
        in the configuration

    UserWarning
        if ``has_lensing=True`` and ``has_flatsky_nonlocal=False`` or
        ``has_flatsky_local_nonlocal=False`` in the configuration

    Notes
    -----
    Just like for ``CoffeMultipolesDerivative``, one can set the configuration in
    the constructor by specifying the ``config`` argument.

    If the biases are set for two populations, the second population is ignored.
    """

    _allowed_biases = [
        _BiasParameter(longname="galaxy", shortname="b"),
        _BiasParameter(longname="magnification", shortname="s"),
        _BiasParameter(longname="evolution", shortname="e"),
    ]

    def validate_parameter(self, arg) -> bool:
        valid_shortnames = "".join([_.shortname for _ in self._allowed_biases])
        if re.search(rf"^[{valid_shortnames}][1-9][0-9]*$", arg.name) and int(
            arg.name[1:]
        ) in range(1, len(self.config["z_mean"]) + 1):
            return True
        return False

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        cosmo = _parse_and_set_args(**self.config)

        # checking if we are indeed using the flat-sky approximation
        if (cosmo.has_rsd or cosmo.has_rsd) and not cosmo.has_flatsky_local:
            warnings.warn(
                "You have chosen to compute the derivative w.r.t. the bias "
                "including density or RSD, but have not specified `has_flatsky_local=True`; "
                "the computation will continue, but your results may not be consistent!"
            )

        if cosmo.has_lensing and not (
            cosmo.has_flatsky_local_nonlocal and cosmo.has_flatsky_nonlocal
        ):
            warnings.warn(
                "You have chosen to compute the derivative w.r.t. the bias including lensing, "
                "but have not specified `has_flatsky_local_nonlocal=True` or "
                "`has_flatsky_nonlocal=True`; "
                "the computation will continue, but your results may not be consistent!"
            )

        for index, bias in enumerate(self._allowed_biases):
            self._allowed_biases[index].values = [
                getattr(
                    cosmo,
                    f"{bias.longname}_bias1",
                )(z)
                for z in cosmo.z_mean
            ]

        for arg in args:
            name, value = arg
            for index, bias in enumerate(self._allowed_biases):
                if name[0] == bias.shortname:
                    self._allowed_biases[index].values[int(name[1:]) - 1] = value

        interp_size_limit = 5

        if len(cosmo.z_mean) < interp_size_limit:
            redshifts = np.concatenate(
                [
                    np.linspace(0, cosmo.z_mean[0] * (1 - 1e-3), interp_size_limit),
                    cosmo.z_mean,
                    np.linspace(cosmo.z_mean[-1] * (1 + 1e-3), 10, interp_size_limit),
                ]
            )
            for index, bias in enumerate(self._allowed_biases):
                self._allowed_biases[index].values = np.concatenate(
                    [
                        np.full(interp_size_limit, bias.values[0]),
                        bias.values,
                        np.full(interp_size_limit, bias.values[-1]),
                    ]
                )

        for index, bias in enumerate(self._allowed_biases):
            getattr(cosmo, f"set_{bias.longname}_bias1")(
                redshifts,
                bias.values,
            )
            getattr(cosmo, f"set_{bias.longname}_bias2")(
                redshifts,
                bias.values,
            )

        return np.array(
            [_.value for _ in cosmo.compute_multipoles_bulk()],
        )

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        cosmo = _parse_and_set_args(**self.config)

        for index, bias in enumerate(self._allowed_biases):
            self._allowed_biases[index].values = [
                getattr(
                    cosmo,
                    f"{bias.longname}_bias1",
                )(z)
                for z in cosmo.z_mean
            ]

        for arg in args:
            name, value = arg
            for index, bias in enumerate(self._allowed_biases):
                if name[0] == bias.shortname:
                    self._allowed_biases[index].values[int(name[1:]) - 1] = value

        interp_size_limit = 5

        if len(cosmo.z_mean) < interp_size_limit:
            redshifts = np.concatenate(
                [
                    np.linspace(0, cosmo.z_mean[0] * (1 - 1e-3), interp_size_limit),
                    cosmo.z_mean,
                    np.linspace(cosmo.z_mean[-1] * (1 + 1e-3), 10, interp_size_limit),
                ]
            )
            for index, bias in enumerate(self._allowed_biases):
                self._allowed_biases[index].values = np.concatenate(
                    [
                        np.full(interp_size_limit, bias.values[0]),
                        bias.values,
                        np.full(interp_size_limit, bias.values[-1]),
                    ]
                )

        for index, bias in enumerate(self._allowed_biases):
            getattr(cosmo, f"set_{bias.longname}_bias1")(
                redshifts,
                bias.values,
            )
            getattr(cosmo, f"set_{bias.longname}_bias2")(
                redshifts,
                bias.values,
            )

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
