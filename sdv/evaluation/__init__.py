"""Module to compare real and synthetic data."""

from sdv.evaluation.evaluation import (
    evaluate_quality,
    run_diagnostic,
)
from sdv.evaluation.utils import print_referential_integrity


__all__ = (
    'evaluate_quality',
    'run_diagnostic',
    'print_referential_integrity',
)
