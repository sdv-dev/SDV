import glob
import inspect
import operator
import os
import shutil
import stat
import sys
from pathlib import Path

import tomli
from invoke import task
from packaging.requirements import Requirement
from packaging.version import Version

COMPARISONS = {'>=': operator.ge, '>': operator.gt, '<': operator.lt, '<=': operator.le}
EXTERNAL_DEPENDENCY_CAPS = {
    'torch': '2.9.0'
}

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


@task
def check_dependencies(c):
    c.run('python -m pip check')


@task
def unit(c):
    c.run('python -m pytest ./tests/unit --cov=sdv --cov-report=xml:./unit_cov.xml')


@task
def integration(c):
    c.run('python -m pytest ./tests/integration --reruns 3 --cov=sdv --cov-report=xml:./integration_cov.xml')


@task
def benchmark_dtypes(c):
    c.run('python -m pytest ./tests/benchmark/supported_dtypes_benchmark.py')


def _get_minimum_versions(dependencies, python_version):
    min_versions = {}
    for dependency in dependencies:
        if '@' in dependency:
            name, url = dependency.split(' @ ')
            min_versions[name] = f'{url}#egg={name}'
            continue

        req = Requirement(dependency)
        if ';' in dependency:
            marker = req.marker
            if marker and not marker.evaluate({'python_version': python_version}):
                continue  # Skip this dependency if the marker does not apply to the current Python version

        if req.name not in min_versions:
            min_version = next(
                (spec.version for spec in req.specifier if spec.operator in ('>=', '==')), None
            )
            if min_version:
                min_versions[req.name] = f'{req.name}=={min_version}'

        elif '@' not in min_versions[req.name]:
            existing_version = Version(min_versions[req.name].split('==')[1])
            new_version = next(
                (spec.version for spec in req.specifier if spec.operator in ('>=', '==')),
                existing_version,
            )
            if new_version > existing_version:
                min_versions[req.name] = (
                    f'{req.name}=={new_version}'  # Change when a valid newer version is found
                )

    return list(min_versions.values())


@task
def install_minimum(c):
    with open('pyproject.toml', 'rb') as pyproject_file:
        pyproject_data = tomli.load(pyproject_file)

    dependencies = pyproject_data.get('project', {}).get('dependencies', [])
    python_version = '.'.join(map(str, sys.version_info[:2]))
    minimum_versions = _get_minimum_versions(dependencies, python_version)

    if minimum_versions:
        install_deps = ' '.join(minimum_versions)
        c.run(f'python -m pip install {install_deps}')
        for dep, cap in EXTERNAL_DEPENDENCY_CAPS.values():
            c.run(f'python -m pip install "{dep}<{cap}"')


@task
def minimum(c):
    install_minimum(c)
    check_dependencies(c)
    unit(c)
    integration(c)


@task
def readme(c):
    test_path = Path('tests/readme_test')
    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)

    cwd = os.getcwd()
    os.makedirs(test_path, exist_ok=True)
    shutil.copy('README.md', test_path / 'README.md')
    os.chdir(test_path)
    c.run('rundoc run --single-session python3 -t python3 README.md')
    os.chdir(cwd)
    shutil.rmtree(test_path)


@task
def tutorials(c):
    for ipynb_file in glob.glob('tutorials/*.ipynb') + glob.glob('tutorials/**/*.ipynb'):
        if '.ipynb_checkpoints' not in ipynb_file:
            c.run(
                (
                    'jupyter nbconvert --execute --ExecutePreprocessor.timeout=3600 '
                    f'--to=html --stdout {ipynb_file}'
                ),
                hide='out',
            )


@task
def lint(c):
    check_dependencies(c)
    c.run('ruff check .')
    c.run('ruff format --check --diff .')


@task
def fix_lint(c):
    check_dependencies(c)
    c.run('ruff check --fix .')
    c.run('ruff format .')


def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


@task
def rmdir(c, path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
    except PermissionError:
        pass
