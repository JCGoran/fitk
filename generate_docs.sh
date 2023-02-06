#!/usr/bin/env sh

# script for generating docs for the current module using the pdoc module

set -eu

parse_docs(){
    # need to substitute the version in the init file
    init_path="fitk/__init__.py"
    init_content="$(cat "${init_path}")"
    package_version="$(cat "fitk/VERSION.txt")"
    sed -i 's/\$VERSION/'"${package_version}"'/g' "${init_path}"

    module="fitk/graphics.py"
    file_content="$(cat "${module}")"
    TEMP_IMAGE_DIR="$(mktemp -d -p .)"
    export TEMP_IMAGE_DIR

    # delete the image dir in any case
    # restore the module file
    # restore the init file
    trap 'rm -fr ${TEMP_IMAGE_DIR}; printf "%s\n" "${file_content}" > "${module}"; printf "%s\n" "${init_content}" > "${init_path}"' EXIT INT

    if ! python3 -m poetry > /dev/null 2>&1
    then
        launcher=''
    else
        launcher='poetry run'
    fi

    python3 -m ${launcher} pytest --doctest-modules "${module}"
    for image in ${TEMP_IMAGE_DIR}/*
    do
        base_image="$(basename ${image})"
        sed -i 's?\$IMAGE_PATH\/'${base_image}'?'"data:image/png;base64,$(base64 -w 0 ${image})"'?g' "${module}"
    done
    if [ "${1:-}" = '-i' ]
    then
        python3 -m ${launcher} pdoc -h 0.0.0.0 --docformat numpy --math -t templates/ ./fitk
    else
        python3 -m ${launcher} pdoc --docformat numpy --math -o docs/ -t templates/ ./fitk
    fi

    printf 'The documentation can be found under: \n'
    printf '\t%s\n' "${PWD}/docs/index.html"

    return 0
}

parse_docs "$@"

set +eu
