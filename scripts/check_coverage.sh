#!/usr/bin/env sh

# script for generating a code coverage report for fitk

set -eux

check_coverage(){
    this_dir="$(cd "$(dirname "$0")"; pwd -P)"
    excluded='not __init__.py and not interfaces'
    cd "${this_dir}/.."
    if [ "${1:-}" = '-i' ] || [ "${1:-}" = '--images' ]
    then
        python -m pytest --mpl --doctest-modules --mpl-generate-summary=html -k "${excluded}" --cov=python/fitk/ --cov=tests/ --cov-report=xml tests/ python/fitk/
    else
        python -m pytest --doctest-modules -k "${excluded}" --cov=python/fitk/ --cov=tests/ --cov-report=xml tests/ python/fitk/
    fi
    python -m coverage html
    python -m coverage report
    cd - > /dev/null
}

check_coverage "${1:-}"

set +eux
