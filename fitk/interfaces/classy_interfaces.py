"""
CLASS interfaces for FITK.

Module containing various interfaces for the CLASS (Cosmic Linear Anisotropy
System Solver) code.

##### Notes

For consistency reasons, when computing derivatives or Fisher matrices, you
should explicitly pass to `config` the fiducial values of the parameters for
which you want to compute those quantities. In other words:

The wrong way
```python
cosmo = ClassyCMBDerivative()
derivative = cosmo.derivative('signal', D(P('Omega_cdm', 0.27), 1e-3))
```

The right way
```python
cosmo = ClassyCMBDerivative(config={'output' : 'tCl', 'Omega_cdm' : 0.27})
derivative = cosmo.derivative('signal', D(P('Omega_cdm', 0.27), 1e-3))
```
"""

# for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from abc import ABC
from configparser import ConfigParser
from functools import lru_cache
from itertools import product
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
        """
        Return the current configuration used for running CLASS.

        .. versionchanged:: 0.10.0
            member is now read-only
        """
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
            "galaxy_counts": "nCl" in outputs or "dCl" in outputs,
        }

    def _run_classy(self, *args):  # pylint: disable=method-hidden
        r"""
        Run classy and returns the instance of it after computation.

        Notes
        -----
        The method is cached using <a
        href="https://docs.python.org/3/library/functools.html#functools.lru_cache"
        target="_blank" rel="noreferrer noopener">``functools.lru_cache``</a>
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
        final_kwargs["delta_ell"] = round(
            kwargs.pop("delta_ell", 2 / final_kwargs["fsky"])
        )

        return final_kwargs


class ClassyCMBDerivative(ClassyBaseDerivative):
    """
    Interface for CMB quantities.

    Interface for computing derivatives using the CMB signal and covariance
    (temperature, polarization, or both).

    .. versionadded:: 0.9.2
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
            If not specified, defaults to ``{'output' : 'tCl'}``.
            If the key ``output`` is missing, it is inserted with a default value
            ``tCl``.
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

        Notes
        -----
        The signal is composed of the following contributions, in this order:

        - temperature :math:`C_\ell` s if ``output`` contains ``tCl``
        - E-mode :math:`C_\ell` s if ``output`` contains ``pCl``
        - cross-correlations between T and E-modes if ``output`` contains ``tCl``
          and ``pCl``
        - B-mode :math:`C_\ell` s if ``output`` contains ``pCl`` *and they are non-zero*

        Note that :math:`\ell = \\{0, 1\\}` are not part of the output, i.e. we impose
        :math:`\ell_\mathrm{min} = 2`.

        The output is ordered as follows:

        .. math::
            \mathbf{S} = \\{C_\ell^{TT}, C_\ell^{EE}, C_\ell^{TE}, C_\ell^{BB} \\}
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


class ClassyGalaxyCountsDerivative(ClassyBaseDerivative):
    r"""
    Interface for galaxy number count quantities.

    Interface for computing derivatives using the galaxy number count signal
    and covariance.
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
            If not specified, defaults to `{'output' : 'nCl'}`.
            If the key `output` is missing, it is inserted with a default value
            `nCl`.
        """
        super().__init__(*args, config=config, **kwargs)
        self._config = config if config is not None else {"output": "nCl"}
        if not self._config.get("output"):
            self._config["output"] = "nCl"

    def _parse_redshifts(self):
        r"""
        Parse the redshifts from the configuration and return them.
        """
        raw_redshifts = self.config.get("selection_mean", "")

        if isinstance(raw_redshifts, (tuple, list, np.ndarray)):
            redshifts = raw_redshifts if raw_redshifts else [1.0]
        else:
            redshifts = (
                [item.strip() for item in raw_redshifts.split(",")]
                if raw_redshifts
                else [1.0]
            )

        return redshifts

    def _cross_correlations(self) -> int:
        r"""
        Return which cross-correlations the output contains.
        """
        return int(self.config.get("non_diagonal", 0))

    def _compute_angular_power_spectrum(self, *args):
        r"""
        Compute the $C_\ell$s.

        Computes the angular power spectrum (the $C_\ell$s) and returns the
        number of angular power spectra, the number of redshift bins, and the
        angular power spectra as a dictionary with keys `(ell, index_z1,
        index_z2)`
        """
        cosmo = self._run_classy(*args)

        outputs = self._parse_outputs()

        z_size = len(self._parse_redshifts())

        if outputs["galaxy_counts"]:
            # the output from CLASS
            c_ells = cosmo.density_cl()["dd"]

            # how many angular power spectra do we have
            ell_size = len(c_ells[0])

            # the angular power spectra as a dictionary
            c_ells_dict = {}

            # if we have cross-correlations, we need to handle them specially
            if self._cross_correlations() >= 1 and z_size > 1:
                for ell in range(2, ell_size):
                    counter = 0
                    for i in range(z_size):
                        for j in range(i, z_size):
                            c_ells_dict[(ell, i, j)] = c_ells_dict[
                                (ell, j, i)
                            ] = c_ells[counter][ell]
                            counter += 1
            else:
                c_ells_dict = {
                    (ell, i, i): c_ells[i][ell]
                    for i in range(z_size)
                    for ell in range(2, ell_size)
                }

            return ell_size, z_size, c_ells_dict

        return NotImplemented

    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the signal ($C_\ell$s) of galaxy number counts.

        Parameters
        ----------
        *args
            the name(s) and fiducial value(s) of the parameter(s) for which we
            want to compute the covariance

        Notes
        -----
        The coordinates used are $(z_1, z_2, \ell)$, in that increasing order.
        Note that $\ell = \\{0, 1\\}$ are not part of the output, i.e. we
        impose $\ell_\mathrm{min} = 2$.
        """
        *_, c_ells = self._compute_angular_power_spectrum(*args)

        if c_ells is NotImplemented:
            return c_ells

        return np.array([c_ells[key] for key in sorted(c_ells)])

    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        r"""
        Compute the covariance of the $C_\ell$s of galaxy number counts.

        Parameters
        ----------
        *args
            the name(s) and fiducial value(s) of the parameter(s) for which we
            want to compute the covariance

        **kwargs
            keyword arguments for the covariance. Supported values are:
            - `fsky`: the sky coverage of the survey (default: 1)
            - `delta_ell`: the bin width in multipole space (default: 2 / `fsky`)

        Notes
        -----
        The covariance is computed as:

        $$
            \mathsf{C}[(ij), (pq), \ell, \ell'] = \delta_{\ell, \ell'}
            \frac{
                C_\ell(i, p) C_\ell(j, q) + C_\ell(i, q) C_\ell(j, p)
            }
            {
                f_\mathrm{sky} \Delta \ell (2 \ell + 1)
            }
        $$

        The covariance is block diagonal in $\ell$, that is:
        $$
            \begin{pmatrix}
            \mathsf{C}[(ij), (pq), \ell = 2] & 0 & \ldots & 0\\\
            0 & \mathsf{C}[(ij), (pq), \ell = 3] & \ldots & 0\\\
            \vdots & \vdots & \ddots & \vdots\\\
            0 & 0 & \ldots & \mathsf{C}[(ij), (pq), \ell = \ell_\mathrm{max}]
            \end{pmatrix}
        $$

        Note that the covariance may be singular if the cross-correlations
        between redshift bins are not zero (i.e. if using a non-zero value for
        the `non_diagonal` parameter).
        """
        ell_size, z_size, c_ells = self._compute_angular_power_spectrum(*args)

        if c_ells is NotImplemented:
            return c_ells

        if self._cross_correlations():
            blocks = []
            # we skip ell=0 and ell=1 as CLASS sets them to zero anyway
            for ell in range(2, ell_size):
                # array containing the covariance for fixed ell
                covariance_fixed_ell = np.zeros([z_size] * 4)
                for i1, i2, j1, j2 in product(  # pylint: disable=invalid-name
                    range(z_size), repeat=4
                ):
                    covariance_fixed_ell[i1, i2, j1, j2] = (
                        c_ells[(ell, i1, j1)] * c_ells[(ell, i2, j2)]
                        + c_ells[(ell, i1, j2)] * c_ells[(ell, i2, j1)]
                    )
                blocks.append(
                    np.reshape(covariance_fixed_ell, (z_size * z_size, z_size * z_size))
                    / (2 * ell + 1)
                )

            result = block_diag(*blocks)

        else:
            result = np.diag(np.array([2 * c_ells[key] ** 2 for key in sorted(c_ells)]))

        final_kwargs = self._parse_covariance_kwargs(**kwargs)

        return result / final_kwargs["fsky"] / final_kwargs["delta_ell"]
