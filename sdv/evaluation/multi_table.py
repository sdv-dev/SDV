"""Methods to compare the real and synthetic data for multi-table."""

from sdmetrics import visualization
from sdmetrics.reports.multi_table.diagnostic_report import DiagnosticReport
from sdmetrics.reports.multi_table.quality_report import QualityReport

import sdv.evaluation.single_table as single_table_visualization


def evaluate_quality(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.

    Args:
        real_data (dict):
            Dictionary containing the real table data.
        synthetic_column (dict):
            Dictionary containing the synthetic table data.
        metadata (Metadata):
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
        metadata (Metadata):
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


def get_column_plot(real_data, synthetic_data, metadata, table_name, column_name, plot_type=None):
    """Get a plot of the real and synthetic data for a given column.

    Args:
        real_data (dict):
            Dictionary containing the real table data.
        synthetic_column (dict):
            Dictionary containing the synthetic table data.
        metadata (Metadata):
            Metadata describing the data.
        table_name (str):
            The name of the table.
        column_name (str):
            The name of the column.
        plot_type (str or None):
            The plot type to use to plot the cardinality. Must be either 'bar' or 'distplot'. If
            ``None``, select between 'bar' or displot depending on the data.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    metadata = metadata.tables[table_name]
    real_data = real_data[table_name] if real_data else None
    synthetic_data = synthetic_data[table_name] if synthetic_data else None
    return single_table_visualization.get_column_plot(
        real_data,
        synthetic_data,
        metadata,
        column_name,
        plot_type,
    )


def get_column_pair_plot(
    real_data, synthetic_data, metadata, table_name, column_names, plot_type=None, sample_size=None
):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (dict or None):
            Dictionary containing the real table data.
        synthetic_column (dict or None):
            Dictionary containing the synthetic table data.
        metadata (Metadata):
            Metadata describing the data.
        table_name (str):
            The name of the table.
        column_names (list[string]):
            The names of the two columns to plot.
        plot_type (str or None):
            The plot to be used. Can choose between ``box``, ``heatmap``, ``scatter``, ``violin``
            or ``None``. If ``None` select between ``box``, ``heatmap`` or ``scatter`` depending
            on the data that the column contains, ``scatter`` used for datetime and numerical
            values, ``heatmap`` for categorical and ``box`` for a mix of both. Defaults to
            ``None``.
        sample_size (int or None):
            The number of samples to plot. If ``None``, all samples are plotted.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.
    """
    metadata = metadata.tables[table_name]
    real_data = real_data[table_name] if real_data else None
    synthetic_data = synthetic_data[table_name] if synthetic_data else None
    return single_table_visualization.get_column_pair_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_names=column_names,
        plot_type=plot_type,
        sample_size=sample_size,
    )


def get_cardinality_plot(
    real_data,
    synthetic_data,
    child_table_name,
    parent_table_name,
    child_foreign_key,
    metadata,
    plot_type='bar',
):
    """Get a plot of the cardinality of the parent-child relationship.

    Args:
        real_data (dict):
            The real data.
        synthetic_data (dict):
            The synthetic data.
        child_table_name (string):
            The name of the child table.
        parent_table_name (string):
            The name of the parent table.
        child_foreign_key (string):
            The name of the foreign key column in the child table.
        metadata (Metadata):
            Metadata describing the data.
        plot_type (str):
            The plot type to use to plot the cardinality. Must be either 'bar' or 'distplot'.
            Defaults to 'bar'.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    parent_primary_key = metadata.tables[parent_table_name].primary_key
    return visualization.get_cardinality_plot(
        real_data,
        synthetic_data,
        child_table_name,
        parent_table_name,
        child_foreign_key,
        parent_primary_key,
        plot_type,
    )
