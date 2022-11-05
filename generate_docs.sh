#!/usr/bin/env bash

# need to substitute the version in the init file
init_path="fitk/__init__.py"
init_content="$(cat "${init_path}")"
package_version="$(python3 -m poetry version --short)"
sed -i 's/\$VERSION/'"${package_version}"'/g' "${init_path}"
python3 -m pdoc --docformat numpy --math -o docs/ fitk
# restore the init file
printf '%s' "${init_content}" > "${init_path}"
