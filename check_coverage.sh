#!/usr/bin/env sh

# script for generating a code coverage report for fitk

set -eux

check_coverage(){
    excluded='not __init__.py and not interfaces'
    if [ "${1:-}" = '-i' ] || [ "${1:-}" = '--images' ]
    then
        poetry run pytest --mpl --doctest-modules --mpl-generate-summary=html -k "${excluded}" --cov=fitk/ --cov=tests/ --cov-report=xml tests/ fitk/
    else
        poetry run pytest --doctest-modules -k "${excluded}" --cov=fitk/ --cov=tests/ --cov-report=xml tests/ fitk/
    fi
    poetry run coverage html
    poetry run coverage report
}

check_coverage "${1:-}"

set +eux
