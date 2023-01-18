"""
Tests for the operations submodule
"""

from __future__ import annotations

# third-party imports
import numpy as np
import pytest
from cosmicfish_pylib.fisher_matrix import fisher_matrix as CFFisherMatrix
from cosmicfish_pylib.fisher_operations import information_gain

# first party imports
from fitk.operations import bayes_factor, kl_divergence, kl_matrix
from fitk.tensors import FisherMatrix
from fitk.utilities import MismatchingValuesError


class TestFisherOperations:
    """
    Tests for any functions operating on multiple Fisher objects
    """

    def test_bayes_factor(self):
        fisher_base = FisherMatrix(np.diag([1, 2, 3]))
        fisher_extended = FisherMatrix(np.diag([1, 2, 3, 4, 5]))
        priors = [1, 1]
        offsets = [0, 0, 0, 0, 0]

        assert np.allclose(
            0.7117625434171772,
            bayes_factor(fisher_base, fisher_extended, priors=priors, offsets=offsets),
        )

        # mismatching sizes
        with pytest.raises(ValueError):
            bayes_factor(
                fisher_base,
                fisher_extended.drop(fisher_extended.names[0]),
                priors=priors,
                offsets=offsets,
            )

        # not a nested model
        with pytest.raises(ValueError):
            bayes_factor(fisher_base, fisher_base, priors=priors, offsets=offsets)

        # wrong length of prior array
        with pytest.raises(ValueError):
            bayes_factor(fisher_base, fisher_extended, priors=[1], offsets=offsets)
        # wrong length of offset array
        with pytest.raises(ValueError):
            bayes_factor(fisher_base, fisher_extended, priors=priors, offsets=[0, 0])

        # emit warning if offsets are larger than 1-sigma marg. errors
        with pytest.warns(UserWarning):
            bayes_factor(
                fisher_base, fisher_extended, priors=priors, offsets=[5, 5, 5, 5, 5]
            )

    def test_kl_divergence(self):
        fisher1 = FisherMatrix(np.diag([1, 2, 3]))
        fisher2 = FisherMatrix(np.diag([4, 5, 6]))
        fisher_prior = FisherMatrix(np.diag([1, 1, 1]))

        cf1 = CFFisherMatrix(fisher1.values)
        cf2 = CFFisherMatrix(fisher2.values)
        cf_prior = CFFisherMatrix(fisher_prior.values)

        result, expectation, _ = kl_divergence(fisher1, fisher2, fisher_prior)
        assert np.allclose(
            information_gain(cf1, cf2, cf_prior, stat=False),
            result,
        )

        assert np.allclose(
            information_gain(cf1, cf2, cf_prior, stat=True),
            expectation,
        )

        # the matrices have incompatible parameters
        with pytest.raises(MismatchingValuesError):
            kl_divergence(fisher1, fisher2.drop(fisher2.names[-1]), fisher_prior)

        # prior has incompatible parameters
        with pytest.raises(MismatchingValuesError):
            kl_divergence(fisher1, fisher2, fisher_prior.drop(fisher_prior.names[-1]))

        # testing whether we really set a zero prior
        assert np.allclose(
            kl_divergence(fisher1, fisher2),
            kl_divergence(
                fisher1, fisher2, FisherMatrix(np.diag(np.zeros(len(fisher1))))
            ),
        )

    def test_kl_matrix(self):
        fisher1 = FisherMatrix(np.diag([1, 2, 3]))
        fisher2 = FisherMatrix(np.diag([4, 5, 6]))
        fisher3 = FisherMatrix(np.diag([8, 3, 4]))

        klm = kl_matrix(fisher1, fisher2, fisher3)

        assert np.allclose(klm[0, 0], kl_divergence(fisher1, fisher1)[0])
        assert np.allclose(klm[0, 1], kl_divergence(fisher1, fisher2)[0])
        assert np.allclose(klm[1, 0], kl_divergence(fisher2, fisher1)[0])
        assert np.allclose(klm[1, 1], kl_divergence(fisher2, fisher2)[0])
