#!/usr/bin/env bash

# script for finding a git tag in some format, and switching to it
set -eu

# find the latest tag in the current branch
TAG="$(git describe --tags --abbrev=0)"

# check whether the tag matches a versioning scheme
if [[ "${TAG}" =~ [0-9]+\.[0-9]+\.[0-9]+$ ]]
then
        printf 'Checking out tag %s\n' "${TAG}"
        # force check out the tag
        git checkout --force "${TAG}"
else
        printf 'Nothing to checkout\n'
fi

set +eu
