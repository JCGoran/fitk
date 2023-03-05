#!/usr/bin/env sh

set -eux

validate(){
    modules='fitk/ tests/'
    if ! python3 -m poetry > /dev/null 2>&1
    then
        launcher=''
    else
        launcher='poetry run'
    fi

    python3 -m ${launcher} black --check ${modules}
    # the E0611 is because Pylint detects some native modules (like
    # `collections.abc.Collection`) as missing on some machines for one reason
    # or another
    python3 -m ${launcher} pylint --disable=W,R,C,E0611 ${modules}
}

validate

set +eux
