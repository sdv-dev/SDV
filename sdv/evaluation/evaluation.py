"""Methods to compare the real and synthetic data for single-table."""

import pandas as pd
from sdmetrics import visualization
from sdmetrics.reports import DiagnosticReport, QualityReport

from sdv.errors import VisualizationUnavailableError
from sdv.evaluation._utils import _prepare_data_visualization
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


def get_column_plot(
    real_data, synthetic_data, metadata, column_name, table_name=None, plot_type=None
):
    """Get a plot of the real and synthetic data for a given column.

    Args:
        real_data (pandas.DataFrame or None):
            The real table data.
        synthetic_data (pandas.DataFrame or None):
            The synthetic table data.
        metadata (Metadata):
            The table metadata.
        column_name (str):
            The name of the column.
        plot_type (str or None):
            The plot to be used. Can choose between ``distplot``, ``bar`` or ``None``. If ``None`
            select between ``distplot`` or ``bar`` depending on the data that the column contains,
            ``distplot`` for datetime and numerical values and ``bar`` for categorical.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    if isinstance(real_data, dict) and len(real_data) > 1 and table_name is None:
        raise TypeError('For multi-table please provide a table_name.')

    _validate_data(real_data, synthetic_data)
    real_data, synthetic_data, metadata = _handle_single_table(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
    )
    if table_name is None:
        table_name = metadata._get_single_table_name()

    table_metadata = metadata.tables[table_name]
    real_data = real_data[table_name]
    synthetic_data = synthetic_data[table_name]

    sdtype = table_metadata.columns.get(column_name)['sdtype']
    if plot_type is None:
        if sdtype in ['datetime', 'numerical']:
            plot_type = 'distplot'
        elif sdtype in ['categorical', 'boolean']:
            plot_type = 'bar'

        else:
            raise VisualizationUnavailableError(
                f"The column '{column_name}' has sdtype '{sdtype}', which does not have a "
                'supported visualization. To visualize this data anyways, please add a '
                "'plot_type'."
            )

    real_data = _prepare_data_visualization(real_data, table_metadata, column_name, None)
    synthetic_data = _prepare_data_visualization(synthetic_data, table_metadata, column_name, None)

    return visualization.get_column_plot(
        real_data, synthetic_data, column_name, plot_type=plot_type
    )


def get_column_pair_plot(
    real_data,
    synthetic_data,
    metadata,
    column_names,
    table_name=None,
    plot_type=None,
    sample_size=None,
):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        metadata (Metadata):
            The table metadata.
        column_names (list[string]):
            The names of the two columns to plot.
        plot_type (str or None):
            The plot to be used. Can choose between ``box``, ``heatmap``, ``scatter``, ``violin``
            or ``None``. If ``None` select between ``box``, ``heatmap`` or ``scatter`` depending
            on the data that the column contains, ``scatter`` used for datetime and numerical
            values, ``heatmap`` for categorical and ``box`` for a mix of both. Defaults to
            ``None``.
        table_name (str or None):
            The name of the table. If single table, the parameter is optional.
        sample_size (int or None):
            The number of samples to use for the plot. If ``None`` use the whole dataset.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.
    """
    if isinstance(real_data, dict) and len(real_data) > 1 and table_name is None:
        raise TypeError('For multi-table datasets please provide the `table_name` parameter.')

    _validate_data(real_data, synthetic_data)
    real_data, synthetic_data, metadata = _handle_single_table(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
    )
    if table_name is None:
        table_name = metadata._get_single_table_name()

    table_metadata = metadata.tables[table_name]
    real_data = real_data[table_name]
    synthetic_data = synthetic_data[table_name]

    if plot_type is None:
        plot_type = []
        for column_name in column_names:
            sdtype = table_metadata.columns.get(column_name)['sdtype']
            if sdtype in ['numerical', 'datetime']:
                plot_type.append('scatter')
            elif sdtype in ['categorical', 'boolean']:
                plot_type.append('heatmap')
            else:
                raise VisualizationUnavailableError(
                    f"The column '{column_name}' has sdtype '{sdtype}', which does not have a "
                    'supported visualization. To visualize this data anyways, please add a '
                    "'plot_type'."
                )

        if len(set(plot_type)) > 1:
            plot_type = 'box'
        else:
            plot_type = plot_type.pop()

    real_data = _prepare_data_visualization(real_data, table_metadata, column_names, sample_size)
    synthetic_data = _prepare_data_visualization(
        synthetic_data, table_metadata, column_names, sample_size
    )

    return visualization.get_column_pair_plot(real_data, synthetic_data, column_names, plot_type)


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
