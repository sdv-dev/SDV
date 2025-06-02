"""Methods to compare the real and synthetic data for single-table."""

from sdmetrics import visualization
from sdmetrics.reports.single_table.diagnostic_report import DiagnosticReport
from sdmetrics.reports.single_table.quality_report import QualityReport

from sdv.errors import VisualizationUnavailableError
from sdv.evaluation._utils import _prepare_data_visualization
from sdv.metadata.metadata import Metadata


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
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

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
    diagnostic_report = DiagnosticReport()
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    diagnostic_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return diagnostic_report


def get_column_plot(real_data, synthetic_data, metadata, column_name, plot_type=None):
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
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    sdtype = metadata.columns.get(column_name)['sdtype']
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

    real_data = _prepare_data_visualization(real_data, metadata, column_name, None)
    synthetic_data = _prepare_data_visualization(synthetic_data, metadata, column_name, None)

    return visualization.get_column_plot(
        real_data, synthetic_data, column_name, plot_type=plot_type
    )


def get_column_pair_plot(
    real_data, synthetic_data, metadata, column_names, plot_type=None, sample_size=None
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

    real_data = _prepare_data_visualization(real_data, metadata, column_names, sample_size)
    synthetic_data = _prepare_data_visualization(
        synthetic_data, metadata, column_names, sample_size
    )

    return visualization.get_column_pair_plot(real_data, synthetic_data, column_names, plot_type)
