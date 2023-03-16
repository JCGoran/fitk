"""
Interfaces for the CLASS (Cosmic Linear Anisotropy System Solver) code
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from abc import ABC
from typing import Optional

# third party imports
import numpy as np

try:
    import classy
except ImportError:
    IMPORT_SUCCESS = False
else:
    IMPORT_SUCCESS = True

# first party imports
from fitk.derivatives import FisherDerivative


class ClassBaseDerivative(ABC, FisherDerivative):
    """
    Base class for setting up the code
    """

    __software_name__ = "classy"
    __url__ = "https://github.com/lesgourg/class_public"
    __version__ = "3.2.0"
    __maintainers__ = ["Goran Jelic-Cizmek <goran.jelic-cizmek@unige.ch>"]
    __imported__ = IMPORT_SUCCESS

    def __init__(
        self,
        *args,
        config: Optional[dict] = None,
        **kwargs,
    ):
        """
        The constructor for the CLASS interface

        Parameters
        ----------
        config : dict, optional
            the CLASS configuration to use. All parameters except the `output`
            keyword are accepted.
        """
        self.config = config if config is not None else {}
        self.config.pop("output", None)
        super().__init__(*args, **kwargs)

    def _parse_signal(self):
        pass


class ClassNumberCountsDerivative(ClassBaseDerivative):
    """
    Interface for computing derivatives w.r.t. parameters of galaxy number
    counts using the [CLASS](https://github.com/lesgourg/class_public) code
    """

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Interface function for CLASS for computing the signal (number count
        $C_\ell$s, $P(k, z)$, etc.).
        Does not support multiple outputs (yet).

        Parameters
        ----------
        *args
            the name(s) and value(s) of the parameter(s) for which we want to
            compute the derivative

        **kwargs
            any other parameters which will be passed to `classy.Class`

        Returns
        -------
        array_like : float
            the signal as a numpy array
        """
        c_ells_cmb = {
            "tCl": "tt",
            "pCl": "ee",
            "lCl": "pp",
        }
        c_ells_galaxy = {
            "dCl": "dd",
            "nCl": "dd",
            "sCl": "ll",
        }

        # this isn't a parameter for CLASS so we're fine
        if "interface_ksteps" in self.config:
            ksteps = self.config.pop("interface_ksteps")
        else:
            ksteps = 1000

        cosmo = classy.Class()
        cosmo.set(self.config)

        output = cosmo.pars.get("output")
        if not output:
            raise ValueError("The argument `output` cannot be empty")

        cosmo.compute()

        outputs = [_.strip() for _ in output.split(", ")]
        if len(outputs) > 1:
            raise ValueError("Only one output is currently supported")

        index = outputs[0]

        # CMB C_ells
        if any(c_ell in outputs for c_ell in c_ells_cmb):
            return np.array(list(cosmo.raw_cl()[c_ells_cmb[index]]))

        # galaxy C_ells
        if any(c_ell in outputs for c_ell in c_ells_galaxy):
            redshifts = cosmo.pars.get("selection_mean")
            if redshifts:
                return np.ndarray.flatten(
                    np.array(list(cosmo.density_cl()[c_ells_galaxy[index]].values()))
                )
            return np.array(list(cosmo.density_cl()[c_ells_galaxy[index]][0]))

        # matter power spectrum (linear or nonlinear)
        if "mPk" in outputs:
            # CLASS does not populate the pars dictionary on its own, even
            # after the computation, so in case they're not specified, they are
            # set to the defaults
            k_min = cosmo.pars.get("k_min_tau0", 1e-2)
            k_max = cosmo.pars.get("P_k_max_h/Mpc", 1.0)
            redshifts = cosmo.pars.get("z_pk")
            if redshifts:
                redshift_values = np.array(
                    [float(_) for _ in cosmo.pars.get("z_pk").split(", ")],
                )
                return np.array(
                    [
                        cosmo.pk(k, z)
                        for k in np.linspace(k_min, k_max, ksteps)
                        for z in redshift_values
                    ],
                )
            return np.array([cosmo.pk(k, 0) for k in np.linspace(k_min, k_max, ksteps)])

        return np.array([])

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        c_ells_galaxy = {
            "dCl": "dd",
            "nCl": "dd",
            "sCl": "ll",
        }

        # this isn't a parameter for CLASS so we're fine
        if "interface_ksteps" in self.config:
            ksteps = self.config.pop("interface_ksteps")
        else:
            ksteps = 1000

        cosmo = classy.Class()
        cosmo.set(self.config)

        output = cosmo.pars.get("output")
        if not output:
            raise ValueError("The argument `output` cannot be empty")

        cosmo.compute()

        redshifts = cosmo.pars.get("selection_mean")
        return np.array([[]])
        # if redshifts:
        #    result = np.ndarray.flatten(
        #        np.array(list(cosmo.density_cl()[c_ells_galaxy[index]].values()))
        #    )
        # result = np.array(list(cosmo.density_cl()[c_ells_galaxy[index]][0]))

        # return np.diag(result)
