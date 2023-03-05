#!/usr/bin/env sh

set -eux

fix(){
    modules='fitk/ tests/'
    poetry run black ${modules}
}

fix

set +eux
