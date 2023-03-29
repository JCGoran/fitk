"""
Various interfaces for computing derivatives w.r.t. parameters from the CAMB
code
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from abc import ABC
from typing import Optional

# third party imports
import numpy as np

try:
    import camb
except ImportError:
    IMPORT_SUCCESS = False
else:
    IMPORT_SUCCESS = True

# first party imports
from fitk.derivatives import FisherDerivative


class CambBaseDerivative(ABC, FisherDerivative):
    """
    Computes derivatives w.r.t. parameters in CAMB
    """

    software_names = "camb"
    urls = {"github": "https://github.com/cmbant/CAMB"}
    version = "1.0.0"
    authors = [
        dict(
            name="Goran Jelic-Cizmek",
            email="goran.jelic-cizmek@unige.ch",
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
            the CAMB configuration passed to `set_params` as a dictionary. All
            parameters are accepted.
        """
        self._config = config if config is not None else {}
        super().__init__(*args, **kwargs)

    @property
    def config(self):
        return self._config

    def _run_camb(self, *args):
        r"""Run CAMB and return an instance of it."""
        final_kwargs = {**self.config}
        for name, value in args:
            final_kwargs[name] = value

        cosmo = camb.set_params(**final_kwargs)

        return camb.get_results(cosmo)


class CambCMBDerivative(CambBaseDerivative):
    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        result = self._run_camb(*args)
        c_ell = result.get_unlensed_total_cls(raw_cl=True)
        c_tt = c_ell[2:, 0]
        c_ee = c_ell[2:, 1]
        c_bb = c_ell[2:, 2]
        c_te = c_ell[2:, 3]

        if np.any(c_bb):
            return np.concatenate([c_tt, c_ee, c_te, c_bb])

        return np.concatenate([c_tt, c_ee, c_te])
