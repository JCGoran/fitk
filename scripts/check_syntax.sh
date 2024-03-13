#!/usr/bin/env sh

set -eux

validate(){
    modules='python/fitk/ tests/'
    python -m black --check ${modules}
    # the E0611 is because Pylint detects some native modules (like
    # `collections.abc.Collection`) as missing on some machines for one reason
    # or another
    python -m pylint --disable=W,R,C,E0611 ${modules}
}

validate

set +eux
