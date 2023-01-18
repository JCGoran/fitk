"""
This module defines interfaces to various third-party software with which
one can compute Fisher matrices using finite differences.

### Important notice
Due to the complexities involved in distributing software that is not under
direct control of the developers/maintainers of `fitk` (version, installation,
license issues, etc.), any external, third-party software (such as cosmological
codes) is *not* bundled with `fitk` (i.e. installed automatically), and must be
installed separately by the user.

### Computation of custom derivatives
To define a new interface for computing derivatives, one should define a class
that inherits from `fitk.derivatives.FisherDerivative`, and implements
either the `signal` or the `covariance` methods (or both); below outlines the
steps to create an interface of your own using a minimal amount of code:

```python
from __future__ import annotations # required for Python 3.7
import numpy as np
from fitk import FisherDerivative

# optional: if your code has a Python interface, you should import it here
import mycode

class MyFisher(FisherDerivative):
    # define a signal function, so you can compute
    # derivatives w.r.t. it
    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        for name, value in args:
            # go through the parameters and set them
            ...
        # do the calculation using the external module, or some other means
        ...
        # the returned result _must_ be a 1-dimensional numpy array
        return np.array([...])

    # define a covariance function, so you can compute
    # derivatives w.r.t. it
    def covariance(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        for name, value in args:
            # go through the parameters and set them
            ...
        # do the computation
        ...
        # the returned result _must_ be a 2-dimensional, square, numpy array,
        # with the same number of elements as the output of the `signal`
        # method
        return np.array([...])
```

The following are some general recommendations when creating a new interface:
- for ease of use, any additional parameters specifying the configuration for
  the interface should be passed to the constructor (i.e. the `__init__`
  method)
- extra information (methods, members, custom parameters) should be documented
  accordingly
- if the external module could not be imported, rather than directly raising an
  exception, it is preferable that instantiating a class from that module
  raises an `ImportError` instead. A simple way to accomplish this is to wrap
  the import in a `try...except ImportError`, pass the success result (perhaps
  stored as a boolean) to the constructor of the class, and then only raise the
  `ImportError` there
"""
