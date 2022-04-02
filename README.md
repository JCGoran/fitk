## FITK - the Fisher Information ToolKit

Fitk is a Python package for manipulating and plotting of Fisher information matrices.


### Features (what it can do)

* intuitive and consistent interface
* OOP design
* sane defaults
* extensive documentation
* many unit tests to ensure validity of results (using [pytest](https://pytest.org/))
* future-proof and backwards compatible with Python 3.7+
* minimal dependencies (standard lib, `numpy`, `scipy`, and `matplotlib` only)
* simple installation
* rendering of matrices with LaTeX parameters in Jupyter lab/notebooks
* built-in support for arithmetic operations between matrices (`+`, `-`, `*`, `/`, `@` (matrix multiplication), `**` (power))
* ability to compute condition numbers, eigenvalues, eigenvectors, and marginalized or unmarginalized constraints on parameters at arbitrary sigma levels or p-values
* ability to drop or marginalize over any number of parameters, or sort them in any order
* saving and loading of Fisher matrices with metadata support
* plotting of collections of Fisher matrices themselves
* plotting of 1D Gaussians for collections of Fisher matrices
* plotting of 2D ellipses at any confidence level (triangle plots) with optional 1D Gaussians for collections of Fisher matrices


### What it can't do

* compute the Fisher matrix elements themselves (i.e. derivatives of `$THING` w.r.t. parameters)
* be compatible with Python 2


### Installation

The best way to install it is via `pip`:

```plaintext
pip install fitk
```

Note that on some systems you may have to replace `pip` by `pip3` to use Python 3 for the installation.
Furthermore, if you only wish to install the package for the current user (or don't have root privileges), you should supply the `--user` flag to the above command.

Alternatively, if you want to install the latest development version:

```plaintext
pip install git+https://github.com/JCGoran/fitk
```


### Usage

The simplest way to use it is to load the core functions and classes:

```python
import numpy as np
from fitk import FisherMatrix, FisherPlotter
```


### Design principles

* _never_ alter anything in other modules (change `PATH`, change matplotlib backend, etc.); instead, prefer local contexts
* nothing should be modified in-place; this allows us to pass class instances directly instead of using temporary variables which would pollute the global namespace of variables


### FAQ

#### Why can't fitk compute derivatives w.r.t. parameters?

This was considered for addition in the beginning, but was dropped for the following reasons:

* lack of maintainability - scientific codes are written in various programming languages, and since fitk was made in Python, there would either need to be many implementations in those languages with corresponding Python wrappers, or those codes would need to be ported to Python themselves. Furthermore, codes evolve, and keeping fitk up-to-date with them would be a maintenance nightmare
* performance issues - writing a naive numerical finite-difference interface for Python-compatible codes is straightforward, but it can be quite challenging to implement it so that it's stable and performant. For instance, sometimes it's much faster to code up an analytical derivative in the original code itself. Additionally, with the rise of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), many codes now implement the computation of exact derivatives, circumventing the need for the use of numerical methods altogether.

Therefore, fitk is only in charge of the analysis part, i.e. the algebraic manipulation and plotting of already constructed Fisher information matrices.

#### Why doesn't fitk validate the Fisher matrix automatically?

This was added in the initial version, but was a bit of a hassle to use; the user had to disable the automatic validation first by calling the appropriate function, then do manipulations with the Fisher matrix, and finally re-enable the validation again.
This was especially annoying since, without disabling the validation, the user couldn't set an element of the Fisher matrix using `<instance>[name1, name2] = value`, and would constantly throw errors if one forgot to disable it.
The current "sane" way to do it is to just call the `is_valid()` method to make sure that the result you ended up with is indeed a valid Fisher matrix.

#### Why are some methods (like `plot_1d`) in `FisherPlotter` and not in `FisherMatrix`?

It was a design decision that I stuck with while writing the package.

#### Why is there no support for Python 2?

Because it's reached EOL (end of life) in 2020, and I started writing this in 2021.
Furthermore, there are some features (like type hints) that fitk uses which are exclusive to Python 3.

#### Why are some methods (like `sort`) called using the syntax `<instance>.sort(<args>)` and not `sort(<instance>, <args>)`?

Two words: [method chaining](https://en.wikipedia.org/wiki/Method_chaining).
Most `FisherMatrix` methods return an instance of `FisherMatrix`, so it's possible to do wonderful stuff with a single line of code, such as:

```python
fm = FisherMatrix(np.diag([1, 2, 3]), names=list('cba'))
fm.drop('c').sort().to_file('the_one_with_ab.json')
```

Initially, there was the idea to have a similar interface to [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html), i.e. have a keyword argument `inplace` for many of the methods which would toggle whether we return a new instance of `FisherMatrix` or modify the original one in-place (such as [`pandas.drop`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)), however, even they would like to get rid of that behavior (see [this issue](https://github.com/pandas-dev/pandas/issues/16529)), so I just decided to preemptively not include it in the first place.

#### Why does addition/subtraction of `FisherMatrix` objects seemingly return the parameter names in random order?

Internally, the code calls Python's `set` built-in, which doesn't preserve ordering, and consequently the names of the resulting `FisherMatrix` may appear to be sorted randomly.
However, this is not a cause for concern, since the equality (`==`) operator is modified to not care about the ordering of the parameter names, so `m1 + m2 == m2 + m1` returns true regardless of what is the ordering of names on both sides of the equality.

Note that for all of the other binary operations implemented (`*`, `/`, and `@`), the result returned will always have the parameter names sorted according to the left-most operand.

##### Why does `<instance>.inverse()` return an object of type `FisherMatrix`?

The code doesn't differentiate between Fisher matrices and covariance matrices, mostly since we can do the exact same operations on both (and the inverse of a Fisher matrix satisfies the properties of a Fisher matrix), so it's up to the user to track whether something is a Fisher matrix or a covariance matrix.

Since Python allows attributes to be assigned to instances as well as classes themselves, something like this could be useful (albeit hacky):

```python
FisherMatrix.is_instance = True
fm = FisherMatrix(np.diag([1, 2, 3]))
fm.is_inverse # returns False
cov = fm.inverse()
cov.is_inverse = True
cov.is_inverse # returns True
```

Then all `FisherMatrix` instances will have `is_inverse=False` by default, and you can explicitly change it for covariance matrices (though this will reset to `False` every time you call a method on a covariance matrix, so YMMV).

#### Is there some way to print out the matrices as Markdown/LaTeX tables?

It was decided that this was outside the scope of this project.
The recommended way to print out one of those is using a third party package, such as [pytablewriter](https://pypi.org/project/pytablewriter/); for instance, if you'd like to print out the fiducials and the Fisher matrix as a matrix in LaTeX, this is one way to accomplish this:

```python
import pytablewriter

fid = pytablewriter.LatexTableWriter(
    headers=fm.latex_names.tolist(),
    value_matrix=[fm.fiducials.tolist()],
)

fid.write_table()

val = pytablewriter.LatexTableWriter(
    value_matrix=fm.values.tolist(),
)

val.write_table()
```

#### Can fitk go beyond the quadratic estimator and work with higher order corrections (like [this one](https://arxiv.org/abs/1506.04866))?

Not currently, but if there is enough interest I may consider implementing it.


## LICENSE

GNU GPLv3
