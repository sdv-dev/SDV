"""Methods to compare the real and synthetic data."""

import pandas as pd
from sdmetrics.reports import DiagnosticReport, QualityReport

from sdv.metadata.metadata import Metadata

DEFAULT_SINGLE_TABLE_NAME = 'table'
ALLOWED_TYPES = (pd.DataFrame, dict)


def _validate_data_type(data, argument_name):
    if not isinstance(data, ALLOWED_TYPES):
        raise TypeError(
            f'{argument_name} must be a pandas DataFrame or dictionary, got {type(data).__name__}.'
        )


def _validate_data(real_data, synthetic_data):
    _validate_data_type(real_data, 'real_data')
    _validate_data_type(synthetic_data, 'synthetic_data')

    if type(real_data) is not type(synthetic_data):
        raise TypeError(
            'real_data and synthetic_data must have the same type. '
            f'Got {type(real_data).__name__} and '
            f'{type(synthetic_data).__name__}.'
        )


def _handle_single_table(real_data, synthetic_data, metadata):
    if isinstance(real_data, pd.DataFrame) and isinstance(synthetic_data, pd.DataFrame):
        table_name = DEFAULT_SINGLE_TABLE_NAME
        if isinstance(metadata, Metadata):
            table_name = metadata._get_single_table_name() or table_name
        else:
            metadata = Metadata.load_from_dict(
                metadata.to_dict(), single_table_name=DEFAULT_SINGLE_TABLE_NAME
            )

        real_data = {table_name: real_data}
        synthetic_data = {table_name: synthetic_data}

    return real_data, synthetic_data, metadata


def evaluate_quality(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (Metadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        QualityReport:
            Single table quality report object.
    """
    _validate_data(real_data, synthetic_data)
    real_data, synthetic_data, metadata = _handle_single_table(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
    )
    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return quality_report


def run_diagnostic(real_data, synthetic_data, metadata, verbose=True):
    """Run diagnostic report for the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (Metadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        DiagnosticReport:
            Single table diagnostic report object.
    """
    _validate_data(real_data, synthetic_data)
    real_data, synthetic_data, metadata = _handle_single_table(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
    )
    diagnostic_report = DiagnosticReport()
    diagnostic_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return diagnostic_report
