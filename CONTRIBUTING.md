## General

Contributions are very welcome! The below outlines how the development of FITK is done.

### Submitting contributions

In order to submit your contribution, I strongly suggest that you:

- fork this repository
- `git clone` the fork to your local machine
- create a new branch
- make the necessary code changes on that branch, and commit them
- push the branch with the changes
- create a [pull request](https://github.com/JCGoran/fitk/pulls)

### Development

Development of fitk is done using [Poetry](https://github.com/python-poetry/poetry/).
If you have it installed, all you need to do is:

```plaintext
python3 -m poetry install
```

and you should be good to go.

#### Linting

In brief, running:

```plaintext
./check_syntax.sh
```

should not report any errors (indicated by a non-zero exit code).

If the script reports formatting errors, you may fix them by running:

```plaintext
./fix_formatting.sh
```

Note that any programming logic or syntax errors reported by the syntax checker must be fixed manually.

#### Testing

To make sure the current functionality has not been affected in some way by the new changes, run:

```plaintext
./check_coverage.sh
```

In addition to running the tests, the command above creates a code coverage report in HTML format, available at `./htmlcov/index.html`.

In case you modified the `graphics` module, if you would like to check whether your changes differ compared to the benchmark images, you may run:

```plaintext
./check_coverage.sh --images
```

instead.

Should any of the tests fail, you need to fix them first.

If implementing new features, please create corresponding tests as to keep the code coverage approximately constant.
If you are implementing features in the `graphics` module, you can generate the new benchmark images using:

```plaintext
./generate_test_images.sh
```

##### Testing interfaces

The tests for the interfaces to third-party software are not currently part of the code coverage, and are ran separately.
In order to run those locally, use:

```plaintext
poetry run pytest -v tests/test_[INTERFACE]_interfaces.py
```

where `[INTERFACE]` is one of the following:

- `classy`: for tests of the CLASS code
- `coffe`: for tests of the COFFE code

Note that the corresponding third-party code(s) must be installed beforehand for the tests to run properly.

#### Generating documentation

Documentation which can be browsed locally can be generated using:

```plaintext
./generate_docs.sh
```

#### Versioning

You can change the version of the package consistently using:

```plaintext
./change_version.py
```

and follow the prompts.
If you require further info on the capabilities of the script, run the above with the `-h` flag.

**Note**: version changes are only performed by the maintainer(s).
