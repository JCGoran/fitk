"""
Various interfaces that do not belong to any particular third-party software
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from typing import Optional

# third party imports
import numpy as np
from scipy.integrate import quad

# first party imports
from fitk.derivatives import D, FisherDerivative


def _hubble(config: dict):
    hubble_zero = 1 / 3000  # in units h/Mpc

    return hubble_zero * np.sqrt(
        config["omega_m"] * (1 + config["z"]) ** 3
        + (1 - config["omega_m"]) * (1 + config["z"]) ** (3 * (1 + config["w"]))
    )


def _luminosity_distance(
    config: dict,
):
    integrand = lambda zp: (1 + zp) / _hubble({**config, "z": zp})

    return np.array([quad(integrand, 0, _)[0] for _ in config["z"]])


def _validate_config(
    old_config: dict,
    new_config: dict,
):
    for key in new_config:
        if key not in old_config:
            raise KeyError(
                f"The key `{key}` is not one of: {list(old_config.keys())}",
            )

    result = {**old_config, **new_config}

    if len(result["z"]) != len(result["sigma"]):
        raise ValueError(
            f"The new sizes of the arrays for 'z' ({result['z']}) "
            f"and 'sigma' ({result['sigma']}) do not match"
        )


class SupernovaDerivative(FisherDerivative):
    r"""
    Computes derivatives w.r.t. cosmological parameters $\Omega_\mathrm{m}$ and
    $w$ for a supernova measurement.
    For definitions of the quantities used, refer to the documentation of
    `signal` and `covariance`.

    Examples
    --------
    Create an instance with the default config:
    >>> sn = SupernovaDerivative()

    Compute the derivative of the signal w.r.t. $\Omega_\mathrm{m}$:
    >>> sn.derivative('signal', D(name='omega_m', fiducial=0.32, abs_step=1e-3))
    array([-3.17838358])

    Compute the mixed derivative of the signal w.r.t. both $\Omega_\mathrm{m}$
    and $w$:
    >>> sn.derivative('signal', D('omega_m', 0.32, 1e-3), D('w', -1, 1e-3))
    array([2.94319875])

    Compute the Fisher with $\Omega_\mathrm{m}$ and $w$ as parameters:
    >>> fm1 = sn.fisher_matrix(
    ... D('omega_m', 0.32, 1e-3, latex_name=r'$\Omega_\mathrm{m}$'),
    ... D('w', -1, 1e-3, latex_name='$w$'))
    >>> fm1
    FisherMatrix(
        array([[10.1021222 ,  3.13019854],
           [ 3.13019854,  0.96990936]]),
        names=array(['omega_m', 'w'], dtype=object),
        latex_names=array(['$\\Omega_\\mathrm{m}$', '$w$'], dtype=object),
        fiducials=array([ 0.32, -1.  ]))

    Finally, check that we don't alter the result by using a smaller stepsize:
    >>> fm2 = sn.fisher_matrix(
    ... D('omega_m', 0.32, 1e-4, latex_name=r'$\Omega_\mathrm{m}$'),
    ... D('w', -1, 1e-4, latex_name='$w$'))
    >>> fm2 == fm1
    True
    """

    def __init__(
        self,
        *args,
        config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Constructor

        Parameters
        ----------
        config : dict, optional
            the configuration for the supernova measurement. Should be a
            dictionary with the following keys and types: 'omega_m' (`float`,
            default: 0.32), 'w' (`float`, default: -1), 'z', (array-like of
            `float`, default: [1]), 'sigma' (array-like of `float`, default:
            [1])
        """
        self._config = dict(
            omega_m=0.32,
            w=-1,
            z=np.array([1]),
            sigma=np.array([1]),
        )
        if config is not None:
            _validate_config(self._config, config)
            self._config.update(config)

    @property
    def config(self):
        """
        Returns the current configuration.
        """
        return self._config

    @config.setter
    def config(self, value: dict):
        _validate_config(self._config, value)
        self._config.update(value)

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        The signal is modelled as:

        $$
            m(z | \boldsymbol{\theta}) = 5 \log{ d_L (z | \boldsymbol{\theta})}
        $$

        where $m(z)$ is the distance modulus, $\boldsymbol{\theta} =
        (\Omega_\mathrm{m}, w)$, and $d_L$ is the luminosity distance.
        """

        config = {**self.config}
        for arg in args:
            name, value = arg
            config[name] = value

        return 5 * np.log(_luminosity_distance(config))

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        The covariance is modelled as:

        $$
            \mathsf{C} = \mathrm{diag}(\sigma_1^2, \ldots, \sigma_n^2)
        $$

        where $\sigma_i^2$ is the (parameter-independent!) variance at the
        $i$-th redshift bin.
        """
        return np.diag(np.array(self.config["sigma"], dtype=float) ** 2)
