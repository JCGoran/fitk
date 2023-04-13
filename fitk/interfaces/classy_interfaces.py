"""
CLASS interfaces for FITK.

Module containing various interfaces for the CLASS (Cosmic Linear Anisotropy
System Solver) code.
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from abc import ABC
from configparser import ConfigParser
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

# third party imports
import numpy as np
from scipy.linalg import block_diag

try:
    import classy
except ImportError:
    IMPORT_SUCCESS = False
else:
    IMPORT_SUCCESS = True

# first party imports
from fitk.derivatives import FisherDerivative


class ClassyBaseDerivative(ABC, FisherDerivative):
    r"""Base interface for CLASS."""

    software_names = "classy"
    urls = dict(github="https://github.com/lesgourg/class_public")
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
            the CLASS configuration to use. All parameters are accepted.
        """
        if not self.__imported__:
            raise ImportError(
                f"Unable to import the `{self.software_names}` module, "
                "please make sure it is installed; "
                f"for additional help, please consult one of the following URL(s): {self.urls}"
            )

        self._config = config if config is not None else {}
        super().__init__(*args, **kwargs)
        self._run_classy = lru_cache(maxsize=None)(self._run_classy)  # type: ignore

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        r"""
        Load a CLASS configuration from a file.

        Parameters
        ----------
        path : str or Path
            the path to the configuration file
        """
        contents = Path(path).read_text(encoding="utf-8").split("\n")
        config = ConfigParser(inline_comment_prefixes=("#", ";"))
        config.optionxform = lambda option: option  # type: ignore
        section = "default"
        config.read_string(f"[{section}]" + "\n".join(contents))

        final_config = {
            key: value for key, value in dict(config[section]).items() if value
        }

        return cls(config=final_config)

    @property
    def config(self):
        """Return the current configuration used for running CLASS."""
        return self._config

    def _parse_outputs(self):
        raw_output = self.config.get("output", "")

        if isinstance(raw_output, (tuple, list, np.ndarray)):
            outputs = raw_output
        else:
            outputs = {item.strip() for item in raw_output.split(",")}

        return {
            "temperature": "tCl" in outputs,
            "polarization": "pCl" in outputs,
        }

    def _run_classy(self, *args):  # pylint: disable=method-hidden
        r"""
        Run classy and returns the instance of it after computation.

        Notes
        -----
        The method is cached using <a
        href="https://docs.python.org/3/library/functools.html#functools.lru_cache"
        target="_blank" rel="noreferrer noopener">`functools.lru_cache`</a>
        """
        cosmo = classy.Class()  # pylint: disable=c-extension-no-member
        final_kwargs = {**self.config}
        for name, value in args:
            final_kwargs[name] = value
        cosmo.set(final_kwargs)
        cosmo.compute()

        return cosmo

    def _parse_covariance_kwargs(self, **kwargs):
        """Parse any keyword arguments for the covariance."""
        final_kwargs = {}
        final_kwargs["fsky"] = float(kwargs.pop("fsky", 1))
        final_kwargs["delta_ell"] = int(
            kwargs.pop("delta_ell", 2 / final_kwargs["fsky"])
        )

        return final_kwargs


class ClassyCMBDerivative(ClassyBaseDerivative):
    """
    Interface for CMB quantities.

    Interface for computing derivatives using the CMB signal and covariance
    (temperature, polarization, or both).
    """

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
            the CLASS configuration to use. All parameters are accepted.
            If not specified, defaults to `{'output' : 'tCl'}`.
            If the key `output` is missing, it is inserted with a default value
            `tCl`.
        """
        super().__init__(*args, config=config, **kwargs)
        self._config = config if config is not None else {"output": "tCl"}
        if not self._config.get("output"):
            self._config["output"] = "tCl"

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the CMB $C_\ell$s.

        Parameters
        ----------
        *args
            the name(s) and value(s) of the parameter(s) for which we want to
            compute the derivative

        Returns
        -------
        result : array_like of float
            the signal as a numpy array

        Notes
        -----
        The signal is composed of the following contributions, in this order:
        - temperature $C_\ell$s if `output` contains `tCl`
        - E-mode $C_\ell$s if `output` contains `pCl`
        - cross-correlations between T and E-modes if `output` contains `tCl`
          and `pCl`
        - B-mode $C_\ell$s if `output` contains `pCl` *and they are non-zero*

        Note that $\ell = \\{0, 1\\}$ are not part of the output, i.e. we impose
        $\ell_\mathrm{min} = 2$.

        The output is ordered as follows:

        $$
            \mathbf{S} = \\{C_\ell^{TT}, C_\ell^{EE}, C_\ell^{TE}, C_\ell^{BB} \\}
        $$
        """
        cosmo = self._run_classy(*args)

        outputs = self._parse_outputs()

        result = []
        if outputs["temperature"]:
            result.extend(cosmo.raw_cl()["tt"][2:])
        if outputs["polarization"]:
            result.extend(cosmo.raw_cl()["ee"][2:])
        if outputs["temperature"] and outputs["polarization"]:
            result.extend(cosmo.raw_cl()["te"][2:])
        if outputs["polarization"] and np.any(cosmo.raw_cl()["bb"]):
            result.extend(cosmo.raw_cl()["bb"][2:])

        if result:
            return np.array(result)

        return NotImplemented

    def _prefactor_covariance(self, size: int):
        l_max = int(self.config.get("l_max_scalars", 2500))
        matrix = np.diag([2 / (2 * ell + 1) for ell in range(2, l_max + 1)])
        return np.tile(matrix, (size, size))

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the covariance of CMB $C_\ell$s.

        Parameters
        ----------
        *args
            the name(s) and fiducial value(s) of the parameter(s) for which we want to
            compute the covariance

        **kwargs
            keyword arguments for the covariance. Supported values are:
            - `fsky`: the sky fraction of the survey (default: 1)

        Returns
        -------
        result : array_like of float
            the signal as a numpy array

        Notes
        -----
        The covariance is the following block-matrix (with the notation $X = C_\ell$):

        $$
            \frac{2}{2 \ell + 1}
            \begin{pmatrix}
            (X^{TT})^2 & (X^{TE})^2 & X^{TT} X^{TE} & 0 \\\\
            (X^{TE})^2 & (X^{EE})^2 & X^{EE} X^{TE} & 0 \\\\
            X^{TT} X^{TE} & X^{EE} X^{TE} & [(X^{TE})^2 + X^{TT} X^{EE}] / 2 & 0 \\\\
            0 & 0 & 0 & (X^{BB})^2
            \end{pmatrix}
        $$

        See the notes of the `signal` method about the order of the outputs in
        the matrix.

        The covariance has been taken from <a
        href="https://arxiv.org/abs/0911.3105" target="_blank" rel="noreferrer
        noopener">arXiv:0911.3105</a>, eq. (27).
        """
        cosmo = self._run_classy(*args)

        outputs = self._parse_outputs()

        c_ell = cosmo.raw_cl()
        c_tt = c_ell.get("tt", [])[2:]
        c_ee = c_ell.get("ee", [])[2:]
        c_te = c_ell.get("te", [])[2:]
        c_bb = c_ell.get("bb", [])[2:]

        # CMB C_ells
        if outputs["temperature"] and outputs["polarization"]:
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

        elif outputs["temperature"]:
            result = self._prefactor_covariance(1) * np.diag(c_tt**2)

        elif outputs["polarization"]:
            if np.any(c_bb):
                result = self._prefactor_covariance(2) * block_diag(
                    np.diag(c_ee**2), np.diag(c_bb**2)
                )
            else:
                result = self._prefactor_covariance(1) * np.diag(c_ee**2)

        else:
            return NotImplemented

        final_kwargs = self._parse_covariance_kwargs(**kwargs)

        return result / final_kwargs["fsky"]
