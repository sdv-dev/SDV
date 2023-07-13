#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()


install_requires = [
    'Faker>=10,<15',
    'graphviz>=0.13.2,<1',
    "numpy>=1.20.0,<1.25.0;python_version<'3.10'",
    "numpy>=1.23.3,<1.25.0;python_version>='3.10'",
    "pandas>=1.1.3;python_version<'3.10'",
    "pandas>=1.5.0;python_version>='3.10'",
    'tqdm>=4.15,<5',
    'copulas>=0.9.0,<0.10',
    'ctgan>=0.7.2,<0.8',
    'deepecho>=0.4.1,<0.5',
    'rdt>=1.5.0,<2',
    'sdmetrics>=0.10.0,<0.11',
    'cloudpickle>=2.1.0,<3.0',
    'boto3>=1.15.0,<2',
    'botocore>=1.18,<2'
]

pomegranate_requires = [
    "pomegranate>=0.14.3,<0.15",
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'pytest-rerunfailures>10',
    'jupyter>=1.0.0,<2',
    'rundoc>=0.4.3,<0.5',
]

development_requires = [
    # general
    'bumpversion>=0.5.3,<0.6',
    'pip>=9.0.1',
    'watchdog>=0.8.3,<0.9',

    # docs
    'docutils>=0.12,<0.18',
    'm2r2>=0.2.5,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'sphinx_toolbox>=2.5,<3',
    'Sphinx>=3,<3.3',
    'pydata-sphinx-theme<0.5',
    'markupsafe<2.1.0',

    # Jinja2>=3 makes the sphinx theme fail
    'Jinja2>=2,<3',

    # style check
    'flake8>=3.7.7,<4',
    'flake8-absolute-import>=1.0,<2',
    'flake8-builtins>=1.5.3,<1.6',
    'flake8-comprehensions>=3.6.1,<3.7',
    'flake8-debugger>=4.0.0,<4.1',
    'flake8-docstrings>=1.5.0,<2',
    'flake8-eradicate>=1.1.0,<1.2',
    'flake8-fixme>=1.1.1,<1.2',
    'flake8-mock>=0.3,<0.4',
    'flake8-multiline-containers>=0.0.18,<0.1',
    'flake8-mutable>=1.2.0,<1.3',
    'flake8-expression-complexity>=0.0.9,<0.1',
    'flake8-print>=4.0.0,<4.1',
    'flake8-pytest-style>=1.5.0,<2',
    'flake8-quotes>=3.3.0,<4',
    'flake8-sfs>=0.0.3,<0.1',
    'flake8-variables-names>=0.0.4,<0.1',
    'dlint>=0.11.0,<0.12',
    'isort>=4.3.4,<5',
    'pandas-vet>=0.2.3,<0.3',
    'pep8-naming>=0.12.1,<0.13',
    'pydocstyle>=6.1.1,<6.2',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<1.6',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',
    'invoke'
]

setup(
    author='DataCebo, Inc.',
    author_email='info@sdv.dev',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: Free for non-commercial use',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description='Generate synthetic data for single table, multi table and sequential data',
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
        'pomegranate': pomegranate_requires,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='sdv synthetic-data synhtetic-data-generation timeseries single-table multi-table',
    license='BSL-1.1',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='sdv',
    packages=find_packages(include=['sdv', 'sdv.*']),
    python_requires='>=3.8,<3.11',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/SDV',
    version='1.2.1',
    zip_safe=False,
)
