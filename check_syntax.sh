#!/usr/bin/env sh

set -eux

validate(){
    modules='fitk/ tests/'
    poetry run black --check ${modules}
    # the E0611 is because Pylint detects some native modules (like
    # `collections.abc.Collection`) as missing on some machines for one reason
    # or another
    poetry run pylint --disable=W,R,C,E0611 ${modules}
}

validate

set +eux
