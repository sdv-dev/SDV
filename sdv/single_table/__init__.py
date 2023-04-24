"""Synthesizers for Single Table data."""

from sdv import _find_addons
from sdv.single_table.copulagan import CopulaGANSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.ctgan import CTGANSynthesizer, TVAESynthesizer

__all__ = (
    'GaussianCopulaSynthesizer',
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'CopulaGANSynthesizer',
)

_find_addons('sdv.single_table_modules', globals(), add_all=True)
