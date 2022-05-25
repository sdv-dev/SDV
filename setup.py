#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    'Faker>=3.0.0,<10',
    'graphviz>=0.13.2,<1',
    "numpy>=1.18.0,<1.20.0;python_version<'3.7'",
    "numpy>=1.20.0,<2;python_version>='3.7'",
    'pandas>=1.1.3,<2',
    'tqdm>=4.15,<5',
    'copulas>=0.7.0,<0.8',
    'ctgan>=0.5.1,<0.6',
    'deepecho>=0.3.0.post1,<0.4',
    'rdt>=0.6.2,<0.7',
    'sdmetrics>=0.5.0,<0.6',
]

pomegranate_requires = [
    "pomegranate>=0.13.4,<0.14.2;python_version<'3.7'",
    "pomegranate>=0.14.1,<0.15;python_version>='3.7'",
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
    'Sphinx>=3,<3.3',
    'pydata-sphinx-theme<0.5',
    'markupsafe<2.1.0',

    # Jinja2>=3 makes the sphinx theme fail
    'Jinja2>=2,<3',

    # style check
    'flake8>=3.7.7,<4',
    'flake8-absolute-import>=1.0,<2',
    'flake8-docstrings>=1.5.0,<2',
    'flake8-sfs>=0.0.3,<0.1',
    'isort>=4.3.4,<5',

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
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description='Synthetic Data Generation for tabular, relational and time series data.',
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
        'pomegranate': pomegranate_requires,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='sdv synthetic-data synhtetic-data-generation timeseries single-table multi-table',
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='sdv',
    packages=find_packages(include=['sdv', 'sdv.*']),
    python_requires='>=3.6,<3.10',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/SDV',
    version='0.15.0.dev1',
    zip_safe=False,
)
