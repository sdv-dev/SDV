import glob
import os
import re
import shutil
import stat
from pathlib import Path

from invoke import task


@task
def pytest(c):
    c.run('python -m pytest --reruns 5 --cov=sdv')


@task
def install_minimum(c):
    with open('setup.py', 'r') as setup_py:
        lines = setup_py.read().splitlines()

    versions = []
    started = False
    for line in lines:
        if started:
            if line == ']':
                break

            line = line.strip()
            line = re.sub(r',?<=?[\d.]*,?', '', line)
            line = re.sub(r'>=?', '==', line)
            line = re.sub(r"""['",]""", '', line)
            versions.append(line)

        elif line.startswith('install_requires = ['):
            started = True

    c.run(f'python -m pip install {" ".join(versions)}')


@task
def minimum(c):
    install_minimum(c)
    c.run('python -m pip check')
    c.run('python -m pytest')


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
            c.run((
                'jupyter nbconvert --execute --ExecutePreprocessor.timeout=3600 '
                f'--to=html --stdout {ipynb_file}'
            ), hide='out')


@task
def lint(c):
    c.run('flake8 sdv')
    c.run('flake8 tests --ignore=D,SFS2')
    c.run('isort -c --recursive sdv tests')
    c.run('pydocstyle sdv')
    c.run('pylint rdt --rcfile=setup.cfg')


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)


@task
def rmdir(c, path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
    except PermissionError:
        pass
