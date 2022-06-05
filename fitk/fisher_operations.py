# needed for compatibility with Python 3.7
from __future__ import annotations

import warnings

# standard library imports
from typing import Collection, Optional, Tuple

# third party imports
import numpy as np
from fitk.fisher_matrix import FisherMatrix
from fitk.fisher_utils import (
    MismatchingSizeError,
    MismatchingValuesError,
    process_units,
)


def bayes_factor(
    fisher_base: FisherMatrix,
    fisher_extended: FisherMatrix,
    priors: Collection[float],
    offsets: Collection[float],
):
    r"""
    Returns the Bayes factor for a nested model, defined as:
    \[
        (2 \pi)^{-p / 2} \frac{\sqrt{\mathrm{det} \mathsf{F}_B}}{\sqrt{\mathrm{det} \mathsf{F}_E}}
        \exp{\left[-\frac{1}{2} \delta \theta_\alpha \mathsf{F}_B \delta \theta_\beta\right]}
        \prod\limits_{q = 1}^{p} \Delta \theta_{n + q}
    \]
    where \(\mathsf{F}_B\) is the Fisher matrix of the base model (size \(n
    \times n\)), \(\mathsf{F}_E\) is the Fisher matrix of the extended model
    (size \(n' \times n'\), with \(n' = n + p\)), \(\delta \theta_\alpha\) is
    the offset array (size \(n\)), and \(\Delta \theta_\alpha\) is the prior
    volume (size \(p\)). For more details, see
    [arXiv:astro-ph/0703191](https://arxiv.org/abs/astro-ph/0703191), eq. (14).

    Parameters
    ----------
    fisher_base : FisherMatrix
        the Fisher matrix of the base (simpler) model

    fisher_extended : FisherMatrix
        the Fisher matrix of the extended (more complex) model

    priors : Collection[float]
        the priors for all of the parameters in the extended model

    offsets : Collection[float]
        the offsets induced in the base model by the extended model

    Returns
    -------
    float

    Raises
    ------
    `ValueError` is raised in one of the following situations:

    * `fisher_extended` does not have at least the same parameter names as `fisher_base`
    * the size of `priors` is not equal to the difference between the sizes of
    `fisher_extended` and `fisher_base`
    * the size of `offsets` is not equal to the size of `fisher_base`

    Notes
    -----
    Internally, the method first computes the logarithm of the Bayes factor to prevent
    numerical over- and underflow, and returns the exponential of that result.

    Furthermore, a warning is printed if the value of any of the offsets is
    larger than the \(1\sigma\) marginalized constraints.

    Examples
    --------
    >>> fisher_base = FisherMatrix(np.diag([1, 2, 3]))
    >>> fisher_extended = FisherMatrix(np.diag([1, 2, 3, 4, 5]))
    >>> bayes_factor(fisher_base, fisher_extended,
    ... priors=[1, 1], offsets=[0, 0, 0])
    0.035588127170858866

    """
    if not set(fisher_base.names).issubset(set(fisher_extended.names)):
        raise ValueError(
            "The extended Fisher matrix must contain at least the parameters of the base Fisher matrix"
        )

    # check dimensions
    n_base = len(fisher_base)
    n_extended = len(fisher_extended)
    n_extra = n_extended - n_base

    if len(priors) != n_extra:
        raise ValueError(
            f"The number of elements in the prior array ({len(priors)}) does not match the number of extra parameters from the extended model ({n_extra})"
        )

    if len(offsets) != n_base:
        raise ValueError(
            f"The number of elements in the offset array ({len(offsets)}) does not match the number of parameters in the base model ({n_base})"
        )

    if np.any(np.array(offsets) / fisher_base.constraints() >= 1):
        warnings.warn(
            "The Fisher matrix of the base model has offsets "
            "larger than the 1 sigma marginalized error, "
            "the obtained result may not be reliable"
        )

    return np.exp(
        -np.log(2 * np.pi) * n_extra / 2
        + np.linalg.slogdet(fisher_base.values)[-1] / 2
        - np.linalg.slogdet(fisher_extended.values)[-1] / 2
        - np.array(offsets) @ fisher_base.values @ np.array(offsets) / 2
        + np.sum(np.log(np.array(priors)))
    )


def kl_divergence(
    fisher1: FisherMatrix,
    fisher2: FisherMatrix,
    fisher_prior: Optional[FisherMatrix] = None,
    units: str = "b",
) -> Tuple[float, float, float]:
    r"""
    Computes the Kullback-Leibler divergence (or relative entropy), \(D(P_2 ||
    P_1)\), its expectation value, \(\langle D \rangle\), and the square roots
    of the variance, \(\sqrt{\sigma^2(D)}\), between two Gaussian probability
    distributions, \(P_1\) and \(P_2\). For more details, see
    [arXiv:1402.3593](https://arxiv.org/abs/1402.3593), section 3.

    Parameters
    ----------
    fisher1 : FisherMatrix
        the Fisher matrix of the first distribution

    fisher2 : FisherMatrix
        the Fisher matrix of the second distribution

    fisher_prior : Optional[FisherMatrix] = None
        the prior Fisher matrix. If not set, defaults to a zero matrix.

    units : str = 'b'
        the information units in which to output the result (default in bits).
        Can be either `'b'` (bits) or `'B'` (bytes), with an optional SI (such
        as `'MB'`) or binary (such as `'MiB'`) prefix. Please consult the table
        at the [Wikipedia article](https://en.wikipedia.org/wiki/Binary_prefix)
        for more details.

    Returns
    -------
    the result as a tuple of 3 `float`s in the requested information units

    Raises
    ------
    * `MismatchingValueError` if the parameter names of the Fisher matrices do not match
    * `ValueError` if the value of `units` cannot be parsed
    """
    if set(fisher1.names) != set(fisher2.names):
        raise MismatchingValuesError("parameter name", fisher1.names, fisher2.names)

    dimension = len(fisher1)

    if fisher_prior is not None and set(fisher1.names) != set(fisher_prior.names):
        raise MismatchingValuesError(
            "parameter name", fisher1.names, fisher_prior.names
        )

    if fisher_prior is None:
        fisher_prior = FisherMatrix(
            np.diag(np.zeros(dimension)),
            names=fisher1.names,
            fiducials=fisher1.fiducials,
        )

    f1_and_prior = (fisher1 + fisher_prior).sort().values
    f2_and_prior = (fisher2 + fisher_prior).sort().values
    f1 = fisher1.sort().values
    f2 = fisher2.sort().values

    factor = process_units(units)

    result = (
        (
            -np.linalg.slogdet(f1_and_prior)[-1]
            + np.linalg.slogdet(f2_and_prior)[-1]
            - dimension
            + np.trace(np.linalg.inv(f2_and_prior) @ f1_and_prior)
        )
        / 2
        / np.log(2)
        * factor
    )

    argument = (
        f2
        @ np.linalg.inv(f2_and_prior)
        @ f1_and_prior
        @ np.linalg.inv(f2_and_prior)
        @ (1 + f2 @ np.linalg.inv(f1_and_prior))
    )

    expectation = result + np.trace(argument) / 2 / np.log(2) * factor

    variance = np.sqrt(np.trace(argument @ argument) / 2 / np.log(2)) * factor

    return (result, expectation, variance)