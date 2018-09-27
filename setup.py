#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'boto3>=1.7.47',
    'exrex==0.10.5',
    'numpy==1.13.1',
    'pandas==0.22.0',
    'scipy==0.19.1',
    'scikit-learn==0.19.1',
    'copulas==0.2.0',
    'rdt==0.1.0'
]

setup_requires = ['pytest-runner', ]

test_require = [
    'pytest>=3.4.2',
    'coverage>=4.5.1',
    'tox>=2.9.1',
    'flake8>=3.5.0',
    'isort>=4.3.4',
]

development_requires = [
    'bumpversion>=0.5.3',
    'watchdog>=0.8.3',
    'Sphinx>=1.7.1',
    'recommonmark>=0.4.0',
    'sphinx_rtd_theme>=0.2.4',
    'flake8>=3.5.0',
    'isort>=4.3.4',
    'autoflake>=1.1',
    'autopep8>=1.3.5',
    'twine>=1.10.0',
    'wheel>=0.30.0',
    'm2r==0.2.0'
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Automated generative modeling and sampling",
    extras_require={
        'test': test_require,
        'dev': test_require + development_requires
    },
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='sdv',
    name='sdv',
    packages=find_packages(include=['sdv', 'sdv.*']),
    python_requires='>=3.5',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=test_require,
    url='https://github.com/HDI-Project/sdv',
    version='0.1.0',
    zip_safe=False,
)
