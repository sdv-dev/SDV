"""Methods to compare the real and synthetic data for multi-table."""
import sdmetrics.reports.utils as report
from sdmetrics.reports.multi_table.diagnostic_report import DiagnosticReport
from sdmetrics.reports.multi_table.quality_report import QualityReport


def evaluate_quality(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.

    Args:
        real_data (dict):
            Dictionary containing the real table data.
        synthetic_column (dict):
            Dictionary containing the synthetic table data.
        metadata (MultiTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        QualityReport:
            Multi table quality report object.
    """
    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return quality_report


def run_diagnostic(real_data, synthetic_data, metadata, verbose=True):
    """Run diagnostic report for the synthetic data.

    Args:
        real_data (dict):
            Dictionary containing the real table data.
        synthetic_column (dict):
            Dictionary containing the synthetic table data.
        metadata (MultiTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        DiagnosticReport:
            Multi table diagnostic report object.
    """
    diagnostic_report = DiagnosticReport()
    diagnostic_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return diagnostic_report


def get_column_plot(real_data, synthetic_data, metadata, table_name, column_name):
    """Get a plot of the real and synthetic data for a given column.

    Args:
        real_data (dict):
            Dictionary containing the real table data.
        synthetic_column (dict):
            Dictionary containing the synthetic table data.
        metadata (MultiTableMetadata):
            Metadata describing the data.
        table_name (str):
            The name of the table.
        column_name (str):
            The name of the column.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    metadata = metadata.to_dict()['tables'][table_name]
    real_data = real_data[table_name]
    synthetic_data = synthetic_data[table_name]
    return report.get_column_plot(real_data, synthetic_data, column_name, metadata)


def get_column_pair_plot(real_data, synthetic_data, metadata, table_name, column_names):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (dict):
            Dictionary containing the real table data.
        synthetic_column (dict):
            Dictionary containing the synthetic table data.
        metadata (MultiTableMetadata):
            Metadata describing the data.
        table_name (str):
            The name of the table.
        column_names (list[string]):
            The names of the two columns to plot.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.
    """
    metadata = metadata.to_dict()['tables'][table_name]
    real_data = real_data[table_name]
    synthetic_data = synthetic_data[table_name]
    return report.get_column_pair_plot(real_data, synthetic_data, column_names, metadata)
