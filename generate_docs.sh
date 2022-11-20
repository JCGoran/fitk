#!/usr/bin/env bash

parse_docs(){
    # need to substitute the version in the init file
    init_path="fitk/__init__.py"
    init_content="$(cat "${init_path}")"
    package_version="$(cat "fitk/VERSION.txt")"
    sed -i 's/\$VERSION/'"${package_version}"'/g' "${init_path}"

    logo_path="./assets/project_logo.png"

    module="fitk/fisher_plotter.py"
    file_content="$(cat "${module}")"
    TEMP_IMAGE_DIR="$(mktemp -d -p .)"
    export TEMP_IMAGE_DIR
    python3 -m doctest "${module}"
    for image in ${TEMP_IMAGE_DIR}/*
    do
        base_image="$(basename ${image})"
        sed -i 's?\$IMAGE_PATH\/'${base_image}'?'"data:image/png;base64,$(base64 -w 0 ${image})"'?g' "${module}"
    done
    if [ "$1" = '-i' ]
    then
        python3 -m pdoc \
            -h 0.0.0.0 \
            --logo "data:image/png;base64,$(base64 -w 0 "${logo_path}")" \
            --favicon "data:image/png;base64,$(convert -background none -gravity center ${logo_path} -resize 400x400 -extent 400x400 png:- | base64 -w 0)" \
            --docformat numpy --math fitk
    else
        python3 -m pdoc \
            --logo "data:image/png;base64,$(base64 -w 0 "${logo_path}")" \
            --favicon "data:image/png;base64,$(convert -background none -gravity center ${logo_path} -resize 400x400 -extent 400x400 png:- | base64 -w 0)" \
            --docformat numpy --math -o docs/ fitk
    fi

    # restore the module file
    printf '%s\n' "${file_content}" > "fitk/fisher_plotter.py"

    # restore the init file
    printf '%s\n' "${init_content}" > "${init_path}"

    # delete the image dir
    trap 'rm -fr ${TEMP_IMAGE_DIR}' EXIT

    return 0
}

parse_docs "$@"
