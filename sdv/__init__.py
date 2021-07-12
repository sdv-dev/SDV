# -*- coding: utf-8 -*-
# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.pthon-guide.org/en/latest/writing/logging/

"""Top-level package for SDV."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.11.0'

from sdv import constraints, evaluation, metadata, relational, tabular
from sdv.demo import get_available_demos, load_demo
from sdv.metadata import Metadata, Table
from sdv.sdv import SDV

__all__ = (
    'demo',
    'constraints',
    'evaluation',
    'metadata',
    'relational',
    'tabular',
    'get_available_demos',
    'load_demo',
    'Metadata',
    'Table',
    'SDV',
)
