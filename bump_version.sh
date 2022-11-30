#!/usr/bin/env sh

# script for bumping the version of a Python package (major, minor, or patch)
# to keep both poetry and the file containing the version number in sync

# requirements:
# - Python 3 with venv and Poetry
# - any POSIX-compatible shell
# - awk


show_usage(){
    cat << EOM
USAGE: ./bump_version.sh [-M|-m|-p]
OPTIONS:
    -M  bump the major version
    -m  bump the minor version
    -p  bump the patch version
EOM
}


update_version(){
    if [ -z "${VIRTUAL_ENV}" ]
    then
        printf "ERROR: you are not in a virtual environment, aborting...\n"
        return 1
    fi

    # relative path of the file containing the version
    VERSION_PATH="$(python3 -m poetry version --no-ansi | awk '{print $1}')/VERSION.txt"

    package_version="$(python3 -m poetry version --short --no-ansi)"

    major="$(printf '%s' "${package_version}" | awk -F'.' '{print $1}')"
    minor="$(printf '%s' "${package_version}" | awk -F'.' '{print $2}')"
    patch="$(printf '%s' "${package_version}" | awk -F'.' '{print $3}')"

    current_version="${major}.${minor}.${patch}"

    printf "Current version: %s\n" "${current_version}"

    case $1 in
        '-M')
            major=$(( major + 1 ))
            minor='0'
            patch='0'
            ;;
        '-m')
            minor=$(( minor + 1 ))
            patch='0'
            ;;
        '-p')
            patch=$(( patch + 1 ))
            ;;
        *)
            show_usage
            return 255
            ;;
    esac

    version="${major}.${minor}.${patch}"

    printf "Proposed change: %s -> %s, are you sure? " "${current_version}" "${version}"
    read -r REPLY

    if [ "${REPLY}" = 'y' ] || [ "${REPLY}" = 'Y' ]
    then
        if [ -z "$(git diff --staged)" ]
        then
            printf "%s\n" "${version}" > "${VERSION_PATH}"
            python3 -m poetry version "${version}"
            git add --update -- pyproject.toml "${VERSION_PATH}" && \
            git commit --message 'Bumped version' &&
            git tag "${version}"
        else
            printf "ERROR: you have uncommited changes in the staging area, please remove or commit them before proceeding\n"
            return 3
        fi
        return 0
    fi

    printf "ERROR: response not 'y' or 'Y', aborting...\n"
    return 2
}


update_version "$@"
