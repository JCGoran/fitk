"""
Submodule for various samplers.
See here for documentation of `FisherDefaultSampler`.
"""

# needed for compatibility with Python 3.7
from __future__ import annotations

# standard library imports
from abc import ABC, abstractmethod
from typing import Any, Optional

# third party imports
import emcee


class FisherBaseSampler(ABC):
    """
    Abstract base class for sampling of the likelihood
    """

    @abstractmethod
    def run_sampler(self, *args, **kwargs):
        """
        Abstract method for running the sampler
        """
        return NotImplemented

    @abstractmethod
    def get_samples(self, *args, **kwargs):
        """
        Abstract method for returning the samples
        """
        return NotImplemented


class FisherDefaultSampler(FisherBaseSampler):
    """
    The default sampler used, which is `emcee`.
    """

    def __init__(
        self,
        ctor_args: Optional[tuple[Any, ...]] = None,
        ctor_kwargs: Optional[dict] = None,
        run_sampler_args: Optional[tuple[Any, ...]] = None,
        run_sampler_kwargs: Optional[dict] = None,
        get_sampler_args: Optional[tuple[Any, ...]] = None,
        get_sampler_kwargs: Optional[dict] = None,
    ):
        """
        Constructor for `emcee.EnsembleSampler`.
        """
        ctor_args = () if not ctor_args else ctor_args
        ctor_kwargs = {} if not ctor_kwargs else ctor_kwargs
        self._run_sampler_args = () if not run_sampler_args else run_sampler_args
        self._run_sampler_kwargs = {} if not run_sampler_kwargs else run_sampler_kwargs
        self._get_sampler_args = () if not get_sampler_args else get_sampler_args
        self._get_sampler_kwargs = {} if not get_sampler_kwargs else get_sampler_kwargs
        self._sampler = emcee.EnsembleSampler(
            *ctor_args,
            **ctor_kwargs,
        )

    @property
    def sampler(self):
        """
        Returns the raw emcee sampler
        """
        return self._sampler

    def run_sampler(self, *args, **kwargs):
        """
        Run the emcee sampler
        """
        self.sampler.run_mcmc(
            *self._run_sampler_args,
            **self._run_sampler_kwargs,
        )

    def get_samples(self, *args, **kwargs):
        """
        Get the samples from emcee
        """
        return self.sampler.get_chain(
            *self._get_sampler_args,
            **self._get_sampler_kwargs,
        )
