# -*- coding: utf-8 -*-
# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.python-guide.org/en/latest/writing/logging/

"""Top-level package for SDV."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '1.0.1'

from sdv._addons import _find_addons

_find_addons('sdv_modules', globals())
