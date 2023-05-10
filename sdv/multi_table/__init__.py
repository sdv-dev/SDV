"""Synthesizers for Multi Table data."""

from sdv._addons import _find_addons
from sdv.multi_table.hma import HMASynthesizer

__all__ = (
    'HMASynthesizer',
)

_find_addons('sdv.multi_table_modules', globals(), add_all=True)
