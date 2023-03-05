#!/usr/bin/env sh

set -eux

fix(){
    modules='fitk/ tests/'
    if ! python3 -m poetry > /dev/null 2>&1
    then
        launcher=''
    else
        launcher='poetry run'
    fi

    python3 -m ${launcher} black ${modules}
}

fix

set +eux
