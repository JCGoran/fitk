#!/usr/bin/env sh

set -eux

fix(){
    modules='python/fitk/ tests/'
    python -m black ${modules}
}

fix

set +eux
