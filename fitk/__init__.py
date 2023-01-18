r"""
<style>
a[target="_blank"]::after {
  content: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAQElEQVR42qXKwQkAIAxDUUdxtO6/RBQkQZvSi8I/pL4BoGw/XPkh4XigPmsUgh0626AjRsgxHTkUThsG2T/sIlzdTsp52kSS1wAAAABJRU5ErkJggg==);
  margin: 0 3px 0 5px;
}
</style>
### FITK - Fisher Information ToolKit, version $VERSION

# What is it?

FITK is a Python package for computing, manipulating, and plotting Fisher-like
objects (mostly matrices).

# Notes

Throughout the documentation, it is assumed that the modules are imported
in the following way:

>>> import numpy as np
>>> from fitk import FisherMatrix, FisherFigure1D, FisherFigure2D, D, FisherDerivative

The most important classes are:

- `fitk.tensors.FisherMatrix`: for manipulating Fisher matrices
- `fitk.graphics.FisherFigure1D` and `fitk.graphics.FisherFigure2D`: for plotting Fisher matrices
- `fitk.derivatives.FisherDerivative` and `fitk.derivatives.D`: for computing derivatives and Fisher
  matrices using finite differences

See the "Submodules" list in the sidebar for more info on the available classes
and methods.

# How do I...?

## ...make a Fisher matrix?

Use the `fitk.tensors.FisherMatrix` class for this.
There are several options you can use:

1. with default names (`p1`, ..., `pn`) and fiducials (all zeros):
>>> my_matrix = FisherMatrix(np.diag([1, 2, 3]))

2. with specified names, fiducials, and LaTeX names (they are all optional):
>>> my_matrix = FisherMatrix(
... np.diag([1, 2, 3]),
... names=['a', 'b', 'c'],
... fiducial=[-1, 0, 1],
... latex_names=['$x$', '$y$', '$z$']
... )

## ...sort a matrix according to some criterion?

Use the `fitk.tensors.FisherMatrix.sort` method, which returns a new Fisher
matrix.  It works the same way as the Python built-in function
<a class="external" href="https://docs.python.org/3/library/functions.html#sorted" target="_blank" rel="noreferrer noopener">`sorted`</a>, but
notably also has some convenience shortcuts:

1. sort according to names (alphabetically)
>>> my_matrix.sort()

2. sort according to values of fiducials
>>> my_matrix.sort(key='fiducials')

3. sort according to LaTeX names (alphabetically)
>>> my_matrix.sort(key='latex_names')

4. sort according names according to some order (invalid names will raise an
error)
>>> my_matrix.sort(key=['b', 'a', 'c'])

5. sort according to indices (0-based) of the names
>>> my_matrix.sort(key=[2, 1, 0])

## ...reverse the order of parameters in the matrix?

You can use the following code:
>>> my_matrix.sort(key=list(reversed(my_matrix.names)))

## ...delete a parameter?

Use the `fitk.tensors.FisherMatrix.drop` method, which returns the new matrix
with those parameters dropped (it does *not* modify the original matrix
in-place!), like:

>>> new_matrix = my_matrix.drop('a', 'b') # drops parameters 'a' and 'b'

If you want to drop parameters *except* the ones listed, use `invert=True`
instead:

>>> new_matrix_complement = my_matrix.drop('a', 'b', invert=True) # drops everything except 'a' and 'b'

An error is raised if a parameter does not exist, unless you pass
`ignore_errors=True`, in which case specified non-existing parameters will
simply be ignored.

## ...rename a parameter?

Use the `fitk.tensors.FisherMatrix.rename` method, which returns the new Fisher
matrix with those names:
>>> my_matrix.rename({'a' : 'q'})

or, if you want to also specify new LaTeX names and fiducials:
>>> my_matrix.rename({'a' : dict(name='q', fiducial=3, latex_name='$q$')})

Note that there shouldn't be any duplicates in the newly-constructed matrix,
otherwise an error is raised.

## ...marginalize over some parameters?

Use the `fitk.tensors.FisherMatrix.marginalize_over` method, which returns a
new Fisher matrix, with those parameters marginalized over:
>>> my_matrix.marginalize_over('a', 'b') # marginalizes over parameters 'a' and 'b'

If you want to marginalize over parameters *except* the ones listed, use
`invert=True` instead:
>>> my_matrix.marginalize_over('a', 'b', invert=True) # marginalizes over all parameters except 'a' and 'b'

## ...make a coordinate change/transformation?

Use the `fitk.tensors.FisherMatrix.reparametrize` method, which allows you to
specify the Jacobian of transformation, and returns the new Fisher matrix:
>>> my_matrix.reparametrize(my_jacobian)

You can also specify the new names, fiducials, and LaTeX names if you want by
passing the `names`, `fiducials`, and `latex_names` parameters, respectively.

Note that the Jacobian does *not* need to be a square matrix, so the new names,
fiducials, and LaTeX names should have the same dimensions as the output
matrix.

## ...verify that the Fisher matrix is valid?

Use the `fitk.tensors.FisherMatrix.is_valid` method, which returns a boolean
reporting whether the matrix is a Fisher matrix:

>>> my_matrix.is_valid()

## ...retrieve/add a value for a specific element in the matrix?

You can use the element accessor `[]`.

For instance, the below outputs the off-diagonal element corresponding to parameters 'a' and 'b':
>>> my_matrix['a', 'b']

while the below sets its value:
>>> my_matrix['a', 'b'] = 5

Note that the transpose element (in this case, `('b', 'a')`) is automatically
updated to keep the Fisher matrix symmetric.

## ...add a Gaussian prior for a specific parameter?

Since this is equivalent to adding $1 / \sigma^2$ to the diagonal element
corresponding to that parameter, see: [...retrieve/add a value for a specific
element in the
matrix?](#retrieveadd-a-value-for-a-specific-element-in-the-matrix)

## ...compute constraints?

Use the `fitk.tensors.FisherMatrix.constraints` method, which returns the
constraints as a numpy array:
>>> my_matrix.constraints()

By default, the constraints returned are the 1$\sigma$ marginalized ones; if
you want the unmarginalized ones, pass `marginalized=False` to the above.
Furthermore, if you want to compute the constraints at a certain threshold
(like a $p$-value or a $\sigma$ value), pass `p=[VALUE]` or `sigma=[VALUE]` to
the above.

## ...save a matrix to a file?

Use the `fitk.tensors.FisherMatrix.to_file` method:
>>> my_matrix.to_file('example_matrix.json')

You can optionally specify metadata by passing `metadata=[VALUE]`, where
`[VALUE]` is a Python dictionary.

## ...read a matrix from a file?

Use the `fitk.tensors.FisherMatrix.from_file` method (note that it's a <a
href="https://docs.python.org/3/library/functions.html#classmethod"
target="_blank" rel="noreferrer noopener">class method</a>, so you need to use
the name of the *class*, not the *instance of that class*):
>>> FisherMatrix.from_file('example_matrix.json')

Note that any metadata (see also: [...save a matrix to a
file?](#save-a-matrix-to-a-file)) is ignored when loading the file.

## ...add two matrices?

Assuming they have the same names and fiducials, it's as simple as:
>>> result = fm1 + fm2

In case the LaTeX names do not match, the names of the left-most matrix are
stored in the result.

## ...compute the figure of merit (FoM)?

Use the `fitk.tensors.FisherMatrix.figure_of_merit` method:
>>> my_matrix.figure_of_merit()

See the linked method for details on how the FoM is computed.

## ...numerically compute a Fisher matrix using a code?

Assuming you already have a Python module for the code, you can follow the
instructions how to define an interface at `fitk.interfaces`.

Once you've done that, and you are sure that the results check out, it should
be as simple as:
>>> interface = MyInterface()

The Fisher matrix with some parameter `a` can then be obtained using the
`fitk.derivatives.FisherDerivative.fisher_matrix` method:
>>> my_matrix = interface.fisher_matrix(D('a', fiducial=0, abs_step=1e-3))

The method takes an optional `parameter_dependence` argument, which specifies
whether the signal or the covariance are parameter-dependent (or both), the
default being `signal`.

## ...compute raw derivatives?

Assuming you already have a Python module for your code, you can follow the
instructions how to define an interface at `fitk.interfaces`.

Once you've done that, and you are sure that the results check out, it should
be as simple as:
>>> interface = MyInterface()

and the derivatives can be obtained with:
>>> derivative = interface.derivative(D('a', fiducial=0, abs_step=1e-3))

## ...make a 1D plot of marginalized parameters?

Use the `fitk.graphics.FisherFigure1D` class for that:
>>> fig = FisherFigure1D()

Then, to actually plot the matrix on the canvas, use
`fitk.graphics.FisherFigure1D.plot`:
>>> fig.plot(my_matrix)

Note that `fitk.graphics.FisherFigure1D.plot` accepts the most frequently used
arguments from <a
href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html"
target="_blank" rel="noopener noreferrer">`matplotlib.pyplot.plot`</a>, such as
`ls` (or `linestyle`), `c` (or `color`), `lw` (or `linewidth`), `label`, as
well as many others.

## ...make a triangle plot/make a 2D plot?

Use the `fitk.graphics.FisherFigure2D` class for that:
>>> fig = FisherFigure2D()

You can optionally specify `show_1d_curves=True` to show the marginalized 1D
curves.

Then, to actually plot the matrix on the canvas, use
`fitk.graphics.FisherFigure2D.plot`:
>>> fig.plot(my_matrix)

Note that `plot` accepts the most frequently used arguments from
<a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html" target="_blank" rel="noreferrer noopener">`matplotlib.pyplot.plot`</a>,
such as `ls` (or `linestyle`), `c` (or `color`), `lw` (or `linewidth`),
`label`, as well as many others.

## ...add a legend to the plot?

Use the `legend` method of `fitk.graphics.FisherFigure1D` or `fitk.graphics.FisherFigure2D`:
>>> fig.legend()

## ...add a title to the plot?

Use the `set_title` method of `fitk.graphics.FisherFigure1D` or `fitk.graphics.FisherFigure2D`:
>>> fig.set_title('Example plot')

## ...save the plot?

Use the `savefig` method of `fitk.graphics.FisherFigure1D` or `fitk.graphics.FisherFigure2D`:
>>> fig.savefig('my_figure.pdf')

It takes exactly the same arguments as
<a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html" target="_blank" rel="noreferrer noopener">`matplotlib.pyplot.savefig`</a>,
with minimal changes to defaults for nicer outputs.
"""

from pathlib import Path

__version__ = (
    (Path(__file__).resolve().parent / "VERSION.txt")
    .read_text(encoding="utf-8")
    .strip()
)

from fitk import interfaces, operations, utilities
from fitk.derivatives import D, FisherDerivative
from fitk.graphics import FisherConstraintsFigure, FisherFigure1D, FisherFigure2D
from fitk.operations import bayes_factor, kl_divergence, kl_matrix
from fitk.tensors import FisherMatrix
from fitk.utilities import math_mode
