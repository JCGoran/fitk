## FITK - the Fisher Information ToolKit
[![codecov](https://codecov.io/gh/JCGoran/fisher-information/branch/master/graph/badge.svg?token=NX9WRX89SI)](https://codecov.io/gh/JCGoran/fisher-information)
[![CircleCI](https://circleci.com/gh/JCGoran/fisher-information/tree/master.svg?style=svg&circle-token=7f9dcec28ca0b548c7a7f01c1e5cbfb6129f513a)](https://circleci.com/gh/JCGoran/fisher-information/tree/master)

Fitk is a Python package for manipulating and plotting of Fisher information matrices.

### Installation

The best way to install the stable version is via `pip`:

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
from fitk import FisherMatrix, FisherFigure1D, FisherFigure2D
```

For the (extensive) documentation, refer to [this website](https://jcgoran.github.io/fisher-information/fitk.html).

### Development

Development of fitk is done using Poetry for Python.
If you have it installed, all you need to do is:

```plaintext
poetry install
```

and you are good to go.

To make sure nothing was broken during development, run:

```plaintext
python -m pytest
```

Should any of the tests fail, you need to fix them first.
If implementing new features, please create corresponding tests as to keep the code coverage approximately constant.

### License

MIT
