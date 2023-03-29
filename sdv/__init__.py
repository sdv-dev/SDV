# -*- coding: utf-8 -*-
# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.python-guide.org/en/latest/writing/logging/

"""Top-level package for SDV."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '1.0.1.dev0'

import warnings

try:
    from importlib_metadata import entry_points
except ImportError:
    from importlib.metadata import entry_points


def _add_version():
    for entry_point in entry_points(name='version', group='sdv_modules'):
        try:
            module = entry_point.load()
        except Exception:
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.module}". '
            warnings.warn(msg)
            continue

        if 'version' not in globals():
            globals()['version'] = module


_add_version()
