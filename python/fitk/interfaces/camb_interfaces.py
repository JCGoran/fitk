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
from scipy.linalg import block_diag

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

    def _parse_covariance_kwargs(self, **kwargs):
        """Parse any keyword arguments for the covariance."""
        final_kwargs = {}
        final_kwargs["fsky"] = float(kwargs.pop("fsky", 1))
        final_kwargs["delta_ell"] = int(
            kwargs.pop("delta_ell", 2 / final_kwargs["fsky"])
        )

        return final_kwargs


class CambCMBDerivative(CambBaseDerivative):
    """
    Interface for CMB quantities.

    Interface for computing derivatives using the CMB signal and covariance
    (temperature and polarization).
    """

    def _prefactor_covariance(self, size: int):
        l_max = int(self.config.get("max_l", 2500))
        matrix = np.diag([2 / (2 * ell + 1) for ell in range(2, l_max + 1)])
        return np.tile(matrix, (size, size))

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the CMB :math:`C_\ell` s.

        Parameters
        ----------
        *args
            the name(s) and value(s) of the parameter(s) for which we want to
            compute the derivative

        Returns
        -------
        result : array_like of float
            the signal as a numpy array
        """

        result = self._run_camb(*args)
        c_ell = result.get_unlensed_total_cls(raw_cl=True)

        c_tt = c_ell[2:, 0]
        c_ee = c_ell[2:, 1]
        c_bb = c_ell[2:, 2]
        c_te = c_ell[2:, 3]

        if np.any(c_bb):
            return np.concatenate([c_tt, c_ee, c_te, c_bb])

        return np.concatenate([c_tt, c_ee, c_te])

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the covariance of CMB :math:`C_\ell` s.

        Parameters
        ----------
        *args
            the name(s) and fiducial value(s) of the parameter(s) for which we want to
            compute the covariance

        **kwargs
            keyword arguments for the covariance. Supported values are:

            - ``fsky``: the sky fraction of the survey (default: 1)

        Returns
        -------
        result : array_like of float
            the signal as a numpy array

        Notes
        -----
        The covariance is the following block-matrix (with the notation :math:`X = C_\ell`):

        .. math::
            \frac{2}{2 \ell + 1}
            \begin{pmatrix}
            (X^{TT})^2 & (X^{TE})^2 & X^{TT} X^{TE} & 0 \\\\
            (X^{TE})^2 & (X^{EE})^2 & X^{EE} X^{TE} & 0 \\\\
            X^{TT} X^{TE} & X^{EE} X^{TE} & [(X^{TE})^2 + X^{TT} X^{EE}] / 2 & 0 \\\\
            0 & 0 & 0 & (X^{BB})^2
            \end{pmatrix}

        See the notes of the ``signal`` method about the order of the outputs in
        the matrix.

        The covariance has been taken from <a
        href="https://arxiv.org/abs/0911.3105" target="_blank" rel="noreferrer
        noopener">arXiv:0911.3105</a>, eq. (27).
        """
        result = self._run_camb(*args)
        c_ell = result.get_unlensed_total_cls(raw_cl=True)

        c_tt = c_ell[2:, 0]
        c_ee = c_ell[2:, 1]
        c_bb = c_ell[2:, 2]
        c_te = c_ell[2:, 3]

        # CMB C_ells
        result = np.block(
            [
                [np.diag(c_tt**2), np.diag(c_te**2), np.diag(c_tt * c_te)],
                [np.diag(c_te**2), np.diag(c_ee**2), np.diag(c_ee * c_te)],
                [
                    np.diag(c_tt * c_te),
                    np.diag(c_ee * c_te),
                    np.diag(c_te**2 + c_tt * c_ee) / 2,
                ],
            ]
        )

        if np.any(c_bb):
            result = self._prefactor_covariance(4) * block_diag(
                result,
                np.diag(c_bb**2),
            )
        else:
            result = self._prefactor_covariance(3) * result

        final_kwargs = self._parse_covariance_kwargs(**kwargs)

        return result / final_kwargs["fsky"]
