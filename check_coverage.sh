#!/usr/bin/env sh

# script for generating a code coverage report for fitk

set -eu

check_coverage(){
    if [ "${1:-}" = '-i' ] || [ "${1:-}" = '--images' ]
    then
        command="python3 -m poetry run pytest --mpl --doctest-modules --mpl-generate-summary=html -k 'not test_interfaces and not __init__.py and not interfaces' --cov=./ --cov-report=html tests/ fitk/"
    else
        command="python3 -m poetry run pytest --doctest-modules -k 'not test_interfaces and not __init__.py and not interfaces' --cov=./ --cov-report=html tests/ fitk/"
    fi
    if ! eval "${command}"
    then
        printf 'Tests failed!\n'
        return 1
    fi
    printf 'Tests successful, you can find the HTML coverage report in the file below:\n'
    printf '\t%s/htmlcov/index.html\n' "${PWD}"
    return 0
}

check_coverage "${1:-}"

set +eu
