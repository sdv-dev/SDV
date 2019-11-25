# -*- coding: utf-8 -*-
# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.pthon-guide.org/en/latest/writing/logging/

"""Top-level package for SDV."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.1'

from sdv.demo import load_demo
from sdv.metadata import Metadata
from sdv.sdv import SDV

__all__ = (
    'load_demo',
    'Metadata',
    'SDV',
)
