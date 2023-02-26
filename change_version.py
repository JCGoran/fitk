#!/usr/bin/env python3

"""
Custom script for changing the version of the current package
"""

import argparse
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Union

from packaging.version import Version


def change_version(version_file: Union[str, Path]):
    """
    Function used to change the version number of the current package

    Parameters
    ----------
    version_file : str or Path
        relative path to the file containing the package version
    """
    current_version = Version(Path(version_file).read_text(encoding="utf-8").strip())

    parser = argparse.ArgumentParser(
        description="Change the version of a package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  To change the version to X.Y.Z while automatically commiting the changes,
  without prompts, run:

  ./change_version.py --git --yes X.Y.Z""",
    )

    parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help="The version we wish to change the package to",
    )

    parser.add_argument(
        "--git",
        action="store_true",
        help="Automatically commit and tag changes to the version",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt anything, assume the answer is always positive",
    )

    parsed_args = parser.parse_args()

    if not parsed_args.version:
        print(f"Current version: {current_version}")

        new_version_input = input("Enter the new version:")
    else:
        new_version_input = str(parsed_args.version)

    new_version = Version(new_version_input)

    print(f"Proposed change: {current_version} -> {new_version}")

    if new_version < current_version:
        warnings.warn(
            "The new version appears to be lower than the current one "
            "(for details of the comparison, see PEP 440)"
        )

    if not parsed_args.yes:
        response = input("Do you want to proceed? [Y/n]")
    else:
        response = "y"

    valid_response_yes = ["y", "yes"]
    valid_response_no = ["n", "no"]

    if response.lower() not in valid_response_yes + valid_response_no:
        print("Unable to understand input, exiting...", file=sys.stderr)
        return 1

    if response.lower() in valid_response_yes:
        Path(version_file).write_text(f"{new_version}\n", encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "poetry",
                "version",
                str(new_version),
            ],
            check=True,
        )

        if parsed_args.git:
            git_location = shutil.which("git")
            git_location = (
                git_location if git_location is not None else "/usr/bin/python3"
            )

            # the script should fail if the working tree is not clean
            is_clean = subprocess.run(
                [
                    git_location,
                    "diff",
                    "--staged",
                ],
                check=True,
                capture_output=True,
            )
            if is_clean.stdout:
                raise Exception(
                    "The current git staging area is not empty, "
                    "please unstage any changes using `git reset .` before continuing"
                )

            subprocess.run(
                [
                    git_location,
                    "add",
                    "--update",
                    "--",
                    "pyproject.toml",
                    str(version_file),
                ],
                check=True,
            )

            subprocess.run(
                [
                    git_location,
                    "commit",
                    "--message",
                    "Bumped version",
                ],
                check=True,
            )
            subprocess.run(
                [
                    git_location,
                    "tag",
                    str(new_version),
                ],
                check=True,
            )
    else:
        print("Nothing to do")

    return 0


if __name__ == "__main__":
    sys.exit(change_version("fitk/VERSION.txt"))
