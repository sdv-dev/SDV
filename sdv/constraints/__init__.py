"""SDV Constraints module."""
from sdv._addons import _find_addons
from sdv.constraints.base import Constraint
from sdv.constraints.tabular import (
    FixedCombinations, FixedIncrements, Inequality, Negative, OneHotEncoding, Positive, Range,
    ScalarInequality, ScalarRange, Unique, create_custom_constraint_class)

__all__ = [
    'create_custom_constraint_class',
    'Constraint',
    'Inequality',
    'ScalarInequality',
    'FixedCombinations',
    'FixedIncrements',
    'Range',
    'ScalarRange',
    'Negative',
    'Positive',
    'OneHotEncoding',
    'Unique'
]

_find_addons('sdv.constraints_modules', globals(), add_all=True)
