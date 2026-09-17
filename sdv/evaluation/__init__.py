"""Module to compare real and synthetic data."""

from sdv.evaluation.evaluation import (
    evaluate_quality,
    run_diagnostic,
)
from sdv.evaluation._visualization import (
    get_cardinality_plot,
    get_column_pair_plot,
    get_column_plot,
)


__all__ = (
    'evaluate_quality',
    'run_diagnostic',
    'get_cardinality_plot',
    'get_column_pair_plot',
    'get_column_plot',
)
