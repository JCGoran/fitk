#!/usr/bin/env sh

# script for generating baseline images using pytest-mpl

set -eux

image_directory='tests/data_input/'
modules="${1:-tests/test_graphics.py}"

python -m pytest --mpl-generate-path="${image_directory}" "${modules}"

set +eux
