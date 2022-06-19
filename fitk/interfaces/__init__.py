"""
This module defines interfaces to various software with which one can compute
Fisher matrices using finite differences.

To define a new interface, one should define a class that inherits from
`FisherDerivative`, and implements at least the `signal` method; below
outlines the steps to create an interface of your own using a minimal amount
of code:

```python
from __future__ import annotations # required for Python 3.7
import numpy as np
from fitk.fisher_derivative import FisherDerivative

# optional: if your code has a Python interface, it would be beneficial to use it here
import mycode

class MyFisher(FisherDerivative):
    def signal(
        self,
        *args: tuple[str, float],
        **kwargs,
    ):
        for name, value in args:
            # go through the parameters and set them
            ...
        # do the calculation using mycode, or some other means (call using
        # `os.system` or `subprocess.Popen`)
        ...
        # the returned result _must_ be a 1-dimensional numpy array
        return np.array([...])

    # optional: define a covariance function as well, so you can compute
    # derivatives w.r.t. the covariance as well
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
"""
