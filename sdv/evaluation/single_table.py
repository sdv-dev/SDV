"""Methods to compare the real and synthetic data for single-table."""

import sdmetrics.reports.utils as report
from sdmetrics.reports.single_table.diagnostic_report import DiagnosticReport
from sdmetrics.reports.single_table.quality_report import QualityReport


def evaluate_quality(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (SingleTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        QualityReport:
            Single table quality report object.
    """
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
        metadata (SingleTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        DiagnosticReport:
            Single table diagnostic report object.
    """
    diagnostic_report = DiagnosticReport()
    diagnostic_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return diagnostic_report


def get_column_plot(real_data, synthetic_data, metadata, column_name):
    """Get a plot of the real and synthetic data for a given column.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_data (pandas.DataFrame):
            The synthetic table data.
        metadata (SingleTableMetadata):
            The table metadata.
        column_name (str):
            The name of the column.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    return report.get_column_plot(real_data, synthetic_data, column_name, metadata.to_dict())


def get_column_pair_plot(real_data, synthetic_data, metadata, column_names):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        metadata (SingleTableMetadata):
            The table metadata.
        column_names (list[string]):
            The names of the two columns to plot.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.
    """
    return report.get_column_pair_plot(real_data, synthetic_data, column_names, metadata.to_dict())
