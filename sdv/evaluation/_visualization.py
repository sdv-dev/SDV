import pandas as pd
from sdmetrics import visualization

from sdv.errors import VisualizationUnavailableError
from sdv.metadata.metadata import Metadata


def _prepare_data_visualization(data, metadata, column_names, sample_size):
    """Prepare the data for a plot.

    Args:
        data (pd.DataFrame or None):
            The data to be prepared.
        metadata (Metadata):
            The metadata of the data.
        column_names (str or list[str]):
            The column names to plot.
        sample_size (int or None):
            The number of samples to plot. If ``None``, use the whole dataset.

    Returns:
        pd.DataFrame or None:
            The prepared data.
    """
    if data is None:
        return None

    col_names = column_names if isinstance(column_names, list) else [column_names]
    data = data.copy()
    for column_name in col_names:
        sdtype = metadata.columns[column_name]['sdtype']
        if sdtype == 'datetime':
            datetime_format = metadata.columns[column_name].get('datetime_format')
            data[column_name] = pd.to_datetime(data[column_name], format=datetime_format)

    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size)

    return data


def _get_column_plot(real_data, synthetic_data, metadata, column_name, plot_type=None):
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
            ``distplot`` for datetime and numerical values and ``bar`` for categorical and ordinal.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    sdtype = metadata.columns.get(column_name)['sdtype']
    if plot_type is None:
        if sdtype in ['datetime', 'numerical']:
            plot_type = 'distplot'
        elif sdtype in ['categorical', 'ordinal', 'boolean']:
            plot_type = 'bar'

        else:
            raise VisualizationUnavailableError(
                f"The column '{column_name}' has sdtype '{sdtype}', which does not have a "
                'supported visualization. To visualize this data anyways, please add a '
                "'plot_type'."
            )

    real_data = _prepare_data_visualization(real_data, metadata, column_name, None)
    synthetic_data = _prepare_data_visualization(synthetic_data, metadata, column_name, None)

    return visualization.get_column_plot(
        real_data, synthetic_data, column_name, plot_type=plot_type
    )


def _get_column_pair_plot(
    real_data, synthetic_data, metadata, column_names, plot_type=None, sample_size=None
):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_data (pandas.Dataframe):
            The synthetic table data.
        metadata (Metadata):
            The table metadata.
        column_names (list[string]):
            The names of the two columns to plot.
        plot_type (str or None):
            The plot to be used. Can choose between ``box``, ``heatmap``, ``scatter``, ``violin``
            or ``None``. If ``None` select between ``box``, ``heatmap`` or ``scatter`` depending
            on the data that the column contains, ``scatter`` used for datetime and numerical
            values, ``heatmap`` for categorical and ordinal, and ``box`` for a mix of both. Defaults
            to ``None``.
        sample_size (int or None):
            The number of samples to use for the plot. If ``None`` use the whole dataset.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.
    """
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    if plot_type is None:
        plot_type = []
        for column_name in column_names:
            sdtype = metadata.columns.get(column_name)['sdtype']
            if sdtype in ['numerical', 'datetime']:
                plot_type.append('scatter')
            elif sdtype in ['categorical', 'ordinal', 'boolean']:
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

    real_data = _prepare_data_visualization(real_data, metadata, column_names, sample_size)
    synthetic_data = _prepare_data_visualization(
        synthetic_data, metadata, column_names, sample_size
    )

    return visualization.get_column_pair_plot(real_data, synthetic_data, column_names, plot_type)


def get_column_plot(real_data, synthetic_data, metadata, table_name, column_name, plot_type=None):
    """Get a plot of the real and synthetic data for a given column.

    Args:
        real_data (dict):
            Dictionary containing the real table data.
        synthetic_data (dict):
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
    return _get_column_plot(
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
            values, ``heatmap`` for categorical, and ordinal, and ``box`` for a mix of both.
            Defaults to ``None``.
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
    return _get_column_pair_plot(
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
