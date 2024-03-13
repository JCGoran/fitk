#!/usr/bin/env sh

set -eux

fix(){
    modules='python/fitk/ tests/'
    this_dir="$(cd "$(dirname "$0")"; pwd -P)"
    cd "${this_dir}/.."
    python -m black ${modules}
    cd - > /dev/null
}

fix

set +eux
