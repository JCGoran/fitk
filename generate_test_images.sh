#!/usr/bin/env sh

# script for generating baseline images using pytest-mpl

set -eu

image_directory='tests/data_input/'
modules='tests/test_graphics.py'

python3 -m poetry run pytest --mpl-generate-path="${image_directory}" ${modules}

set +eu
