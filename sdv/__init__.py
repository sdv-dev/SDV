# -*- coding: utf-8 -*-
# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.python-guide.org/en/latest/writing/logging/

"""Top-level package for SDV."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '1.0.1.dev0'

import warnings
from importlib.metadata import entry_points as get_entry_points


def _add_version():
    try:
        entry_points = get_entry_points(name='version', group='sdv_modules')
    except TypeError:
        entry_points = [
            entry_point for entry_point in get_entry_points().get('sdv_modules', [])
            if entry_point.name == 'version'
        ]

    for entry_point in entry_points:
        try:
            module = entry_point.load()
        except Exception:
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.module}". '
            warnings.warn(msg)
            continue

        if 'version' not in globals():
            globals()['version'] = module


_add_version()
