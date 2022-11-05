#!/usr/bin/env bash

# script for bumping the version of FITK (major, minor, or patch) to keep both
# poetry and the file containing the version number in sync


# relative path of the file containing the version
VERSION_PATH="fitk/VERSION.txt"


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
        return 2
    fi
    package_version="$(python3 -m poetry version --short)"

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

    read -p "$(printf "Proposed change: %s -> %s, are you sure? " "${current_version}" "${version}")" -r

    if [ "${REPLY}" = 'y' ] || [ "${REPLY}" = 'Y' ]
    then

        printf "%s\n" "${version}" > "${VERSION_PATH}"
        python3 -m poetry version "${version}"
        return 0
    fi

    printf "ERROR: response not either 'y' or 'Y', aborting...\n"
    return 3
}


update_version "$@"
