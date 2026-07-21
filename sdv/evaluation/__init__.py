"""Module to compare real and synthetic data."""

from sdv.evaluation.evaluation import (
    evaluate_quality,
    get_cardinality_plot,
    get_column_pair_plot,
    get_column_plot,
    run_diagnostic,
)


__all__ = (
    'evaluate_quality',
    'get_cardinality_plot',
    'get_column_pair_plot',
    'get_column_plot',
    'run_diagnostic',
)
