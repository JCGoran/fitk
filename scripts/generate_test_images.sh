#!/usr/bin/env sh

# script for generating baseline images using pytest-mpl

set -eux

this_dir="$(cd "$(dirname "$0")"; pwd -P)"
cd "${this_dir}/.."
image_directory='tests/data_input/'
modules="${1:-tests/test_graphics.py}"

python -m pytest --mpl-generate-path="${image_directory}" "${modules}"

cd - > /dev/null

set +eux
