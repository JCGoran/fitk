#!/usr/bin/env sh

# script for generating a code coverage report for fitk

set -eux

check_coverage(){
    if [ "${1:-}" = '-i' ] || [ "${1:-}" = '--images' ]
    then
        poetry run pytest --mpl --doctest-modules --mpl-generate-summary=html -k 'not test_interfaces and not __init__.py and not interfaces' --cov=fitk/ --cov=tests/ --cov-report=xml tests/ fitk/
    else
        poetry run pytest --doctest-modules -k 'not test_interfaces and not __init__.py and not interfaces' --cov=fitk/ --cov=tests/ --cov-report=xml tests/ fitk/
    fi
    poetry run coverage html
    poetry run coverage report
}

check_coverage "${1:-}"

set +eux
