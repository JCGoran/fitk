"""
This module defines interfaces to various third-party software with which
one can compute Fisher matrices using finite differences.

### Important notice
Due to the complexities involved in distributing software that is not
under direct control of the developers/maintainers of `fitk` (version,
installation, license issues, etc.), any third-party software (such as
cosmological codes) is *not* bundled with `fitk` (i.e. installed
automatically), and must be installed separately by the user.

### Computation of custom derivatives
To define a new interface, one should define a class that inherits from
`fitk.fisher_derivative.FisherDerivative`, and implements either the `signal`
or the `covariance` method; below outlines the steps to create an interface of
your own using a minimal amount of code:

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

It is recommended (but not required) that any additional parameters should be
stored in the constructor (i.e. `__init__` method), to prevent forgetting to
pass them when computing the derivatives.
Additionally, any extra information (members and methods) should be documented
accordingly.
Finally, if the external module could not be loaded, importing the module
should raise an `ImportError` if the user attempts to instantiate a class
with that module missing.
"""
