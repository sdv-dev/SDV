# -*- coding: utf-8 -*-
# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.python-guide.org/en/latest/writing/logging/

"""Top-level package for SDV."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '1.2.1'


import sys
import warnings
from operator import attrgetter

from pkg_resources import iter_entry_points

from sdv import (
    constraints, data_processing, datasets, evaluation, lite, metadata, metrics, multi_table,
    sampling, sequential, single_table)

__all__ = [
    'constraints',
    'data_processing',
    'datasets',
    'evaluation',
    'lite',
    'metadata',
    'metrics',
    'multi_table',
    'sampling',
    'sequential',
    'single_table'
]


def _get_addon_target(addon_path_name):
    """Find the target object for the add-on.

    Args:
        addon_path_name (str):
            The add-on's name. The add-on's name should be the full path of valid Python
            identifiers (i.e. importable.module:object.attr).

    Returns:
        tuple:
            * object:
                The base module or object the add-on should be added to.
            * str:
                The name the add-on should be added to under the module or object.
    """
    module_path, _, object_path = addon_path_name.partition(':')
    module_path = module_path.split('.')

    if module_path[0] != __name__:
        msg = f"expected base module to be '{__name__}', found '{module_path[0]}'"
        raise AttributeError(msg)

    target_base = sys.modules[__name__]
    for submodule in module_path[1:-1]:
        target_base = getattr(target_base, submodule)

    addon_name = module_path[-1]
    if object_path:
        if len(module_path) > 1 and not hasattr(target_base, module_path[-1]):
            msg = f"cannot add '{object_path}' to unknown submodule '{'.'.join(module_path)}'"
            raise AttributeError(msg)

        if len(module_path) > 1:
            target_base = getattr(target_base, module_path[-1])

        split_object = object_path.split('.')
        addon_name = split_object[-1]

        if len(split_object) > 1:
            target_base = attrgetter('.'.join(split_object[:-1]))(target_base)

    return target_base, addon_name


def _find_addons():
    """Find and load all sdv add-ons."""
    group = 'sdv_modules'
    for entry_point in iter_entry_points(group=group):
        try:
            addon = entry_point.load()
        except Exception:  # pylint: disable=broad-exception-caught
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.module_name}".'
            warnings.warn(msg)
            continue

        try:
            addon_target, addon_name = _get_addon_target(entry_point.name)
        except AttributeError as error:
            msg = f"Failed to set '{entry_point.name}': {error}."
            warnings.warn(msg)
            continue

        setattr(addon_target, addon_name, addon)


_find_addons()
