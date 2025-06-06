"""Script that checks project requirements for pre-release versions."""

from pathlib import Path

import tomllib
from packaging.requirements import Requirement


def get_dev_dependencies(dependency_list):
    """Return list of dependencies with prerelease specifiers."""
    prereleases = []
    for dependency in dependency_list:
        requirement = Requirement(dependency)
        if requirement.specifier.prereleases:
            prereleases.append(dependency)

    return prereleases


if __name__ == '__main__':
    folder = Path(__file__).parent
    toml_path = folder.joinpath('..', 'pyproject.toml')

    with open(toml_path, 'rb') as f:
        pyproject = tomllib.load(f)

    dev_deps = get_dev_dependencies(pyproject['project']['dependencies'])

    if dev_deps:
        raise RuntimeError(f'Found dev dependencies: {", ".join(dev_deps)}')
