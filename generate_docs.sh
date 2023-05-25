#!/usr/bin/env bash

# script for generating docs for the current module using the pdoc module

set -eu

# the name of the package
PACKAGE='fitk'
export PACKAGE

# all of the (versioned!) tags for the package
all_tags="$(git tag --sort=committerdate | awk -F. '$1 > 0 || ($1 == 0 && $2 >= 6)' | sort -V -r)"
all_tags="$(printf '%s\n%s' "master" "${all_tags}")"
LATEST_TAG="$(git tag --sort=committerdate | tail -1)"

# all versions which should be documented by Jinja
VERSIONS="$(echo "${all_tags}" | xargs)"
export VERSIONS

parse_docs(){
    # need to substitute the version in the init file
    init_path="${PACKAGE}/__init__.py"
    init_content="$(cat "${init_path}")"
    package_version="$(cat "${PACKAGE}/VERSION.txt")"
    sed -i 's/\$VERSION/'"${package_version}"'/g' "${init_path}"

    module="${PACKAGE}/graphics.py"
    file_content="$(cat "${module}")"
    TEMP_IMAGE_DIR="$(mktemp -d -p .)"
    export TEMP_IMAGE_DIR

    # delete the image dir in any case
    # restore the module file
    # restore the init file
    trap 'rm -fr ${TEMP_IMAGE_DIR}; printf "%s\n" "${file_content}" > "${module}"; printf "%s\n" "${init_content}" > "${init_path}"' EXIT INT RETURN

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

    docpath="${1:-}"

    python3 -m ${launcher} pdoc --docformat numpy --math -o "${docpath}" -t templates/ ./${PACKAGE}

    printf 'The documentation can be found under: \n'
    printf '\t%s\n' "${docpath}/index.html"

    return 0
}

if [ "${1:-}" = '-t' ]
then
    # every tag that will be documented
    for tag in ${all_tags}
    do
        git reset -- ${PACKAGE}/
        git checkout -- ${PACKAGE}/
        git checkout "${tag}" -- ${PACKAGE}/
        parse_docs "./docs/${tag}/"
    done
    # create an index.html pointing to the latest release
    echo "<meta http-equiv=\"Refresh\" content=\"0; url='/${PACKAGE}/${LATEST_TAG}/'\" />" > ./docs/index.html
else
    parse_docs ./docs
fi

set +eu
