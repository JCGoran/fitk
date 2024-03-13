#!/usr/bin/env sh

set -eux

validate(){
    modules='python/fitk/ tests/'
    this_dir="$(cd "$(dirname "$0")"; pwd -P)"
    cd "${this_dir}/.."
    python -m black --check ${modules}
    # the E0611 is because Pylint detects some native modules (like
    # `collections.abc.Collection`) as missing on some machines for one reason
    # or another
    python -m pylint --disable=W,R,C,E0611,E1126 ${modules}
    cd - > /dev/null
}

validate

set +eux
