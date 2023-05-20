# Changelog

## Unreleased

## 0.10.3

### Changed

- Changed documentation template so inherited methods are not duplicated
- Renamed `_FisherBaseFigure` to `FisherBaseFigure`
- Renamed `_FisherMultipleAxesFigure` to `FisherMultipleAxesFigure`
- Updated documentation so it's more compatible with ReST

## 0.10.2

### Fixed

- Fixed shading of contours when calling `plot` method of `FisherFigure`s

## 0.10.1

### Fixed

- Fixed plotting order when calling `FisherBarFigure` methods with `kind='barh'`

## 0.10.0

### Added

- Added `use_pinv` kwarg to `FisherDerivative.fisher_matrix` and `FisherMatrix.inverse` methods (as well as any associated methods that indirectly use those) for using the Moore-Penrose pseudoinverse

### Fixed

- Fixed prefactor for covariance of CLASS CMB interface
- Improved docstrings for "Quickstart" section
- Improved docstrings for CLASS interfaces

### Changed

- Made `config` member of `ClassyBaseDerivative` a read-only property

## 0.9.3

- No changes, the previous release to PyPI contained extra files

## 0.9.2

### Added

- Added CLASS interface for CMB quantities
- Added COFFE interface for redshift-averaged multipoles
- Added ability to use evolution bias in `CoffeMultipolesBiasDerivative`
- Added ability to install third-party interfaces (COFFE and CLASS) via pip
- Added "Quickstart" section and info about installing interfaces
- Added `software_names`, `version`, `authors`, and `urls` members to `FisherDerivative`

### Removed

- Removed `__software_name__`, `__url__`, `__version__`, and `__maintainers__` members of `CoffeMultipolesDerivative`

## 0.9.1

### Added

- Added `pydocstyle` package to dev dependencies

### Changed

- Updated docstrings

## 0.9.0

### Added

- Added `reparametrize_symbolic` method to `FisherMatrix` for performing symbolic reparametrizations (using SymPy under the hood)
- Added SymPy to main dependencies

## 0.8.0

### Added

- Added `P` class for specifying Fisher parameters

### Changed

- Removed `name`, `fiducial`, and `latex_name` members from `D`, and replaced them with `parameter` instead
- Updated docstrings

## 0.7.2

### Fixed

- Fixed ordering of parameters when using `FisherBarFigure.plot`

## 0.7.1

### Fixed

- Refactored plotting to reduce code complexity

## 0.7.0

### Added

- Added README to PyPI release

### Changed

- Updated docstrings

## 0.6.6

### Fixed

- Fixed bug with parsing of `contour_levels_2d` in `FisherFigure2D`

## 0.6.5

### Fixed

- Fixed bug in parsing of `key` argument to `sort` method of `FisherMatrix`; when `key` was set to a list of names, it did not work correctly
- Fixed bug in `FisherBaseFigure` which resulted in a `KeyError` occasionally being raised in the CI

### Changed

- Updated docstrings

## 0.6.4

### Added

- Added `fiducial`, `set_fiducial`, `latex_name`, and `set_latex_name` methods to `FisherMatrix`

## 0.6.3

### Fixed

- Fixed parsing of `options` kwarg to `FisherBaseFigure`

## 0.6.2

### Changed

- Replaced `**kwargs` of `fisher_matrix` method by `kwargs_signal` and `kwargs_covariance`
- Updated examples

## 0.6.1

### Added

- Added `contour_levels_1d` and `contour_levels_2d` to `FisherFigure2D`

## 0.6.0

### Added

- added `FisherBarFigure` class for bar-like plotting of constraints
- added `mark_fiducials` kwarg to `FisherFigure1D` and `FisherFigure2D`

### Changed

- Renamed modules:
    - `fisher_derivative` to `derivatives`
    - `fisher_matrix` to `tensors`
    - `fisher_plotter` to `graphics`
    - `fisher_utils` to `utilities`
    - `fisher_operations` to `operations`
- Updated docstrings
