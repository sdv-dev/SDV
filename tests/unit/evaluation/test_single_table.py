import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdv.errors import VisualizationUnavailableError
from sdv.evaluation.single_table import (
    DiagnosticReport,
    QualityReport,
    evaluate_quality,
    get_column_pair_plot,
    get_column_plot,
    run_diagnostic,
)
from sdv.metadata.metadata import Metadata


def test_evaluate_quality():
    """Test ``generate`` is called for the ``QualityReport`` object."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')
    QualityReport.generate = Mock()

    # Run
    evaluate_quality(data1, data2, metadata)

    # Assert
    QualityReport.generate.assert_called_once_with(
        data1, data2, metadata._convert_to_single_table().to_dict(), True
    )


def test_evaluate_quality_metadata():
    """Test ``generate`` is called for the ``QualityReport`` object with Metadata."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata_dict = {'columns': {'col': {'sdtype': 'numerical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    QualityReport.generate = Mock()

    # Run
    evaluate_quality(data1, data2, metadata)

    # Assert
    expected_metadata = metadata.tables['table'].to_dict()
    QualityReport.generate.assert_called_once_with(data1, data2, expected_metadata, True)


def test_run_diagnostic():
    """Test ``generate`` is called for the ``DiagnosticReport`` object."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')
    DiagnosticReport.generate = Mock(return_value=123)

    # Run
    run_diagnostic(data1, data2, metadata)

    # Assert
    DiagnosticReport.generate.assert_called_once_with(
        data1, data2, metadata._convert_to_single_table().to_dict(), True
    )


def test_run_diagnostic_metadata():
    """Test ``generate`` is called for the ``DiagnosticReport`` object with Metadata."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata_dict = {'columns': {'col': {'sdtype': 'numerical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    DiagnosticReport.generate = Mock(return_value=123)

    # Run
    run_diagnostic(data1, data2, metadata)

    # Assert
    expected_metadata = metadata.tables['table'].to_dict()
    DiagnosticReport.generate.assert_called_once_with(data1, data2, expected_metadata, True)


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_continuous_data(mock_get_plot):
    """Test the ``get_column_plot`` with continuous data.

    Test that when we call ``get_column_plot`` with continuous data (datetime or numerical)
    this will choose to use the ``distplot`` as ``plot_type``.
    """
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_continuous_data_metadata(mock_get_plot):
    """Test the ``get_column_plot`` with continuous data.

    Test that when we call ``get_column_plot`` with continuous data (datetime or numerical)
    this will choose to use the ``distplot`` as ``plot_type``. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata_dict = {'columns': {'col': {'sdtype': 'numerical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_discrete_data(mock_get_plot):
    """Test the ``get_column_plot`` with discrete data.

    Test that when we call ``get_column_plot`` with discrete data (categorical or boolean)
    this will choose to use the ``bar`` as ``plot_type``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='categorical')

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_discrete_data_metadata(mock_get_plot):
    """Test the ``get_column_plot`` with discrete data.

    Test that when we call ``get_column_plot`` with discrete data (categorical or boolean)
    this will choose to use the ``bar`` as ``plot_type``. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'categorical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_discrete_data_with_distplot(mock_get_plot):
    """Test the ``get_column_plot`` with discrete data.

    Test that when we call ``get_column_plot`` with discrete data (categorical or boolean)
    and pass in the ``distplot`` it will call the ``sdmetrics.visualization.get_column_plot``
    with it and not switch to ``bar``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='categorical')

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col', plot_type='distplot')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_discrete_data_with_distplot_metadata(mock_get_plot):
    """Test the ``get_column_plot`` with discrete data.

    Test that when we call ``get_column_plot`` with discrete data (categorical or boolean)
    and pass in the ``distplot`` it will call the ``sdmetrics.visualization.get_column_plot``
    with it and not switch to ``bar``. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'categorical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col', plot_type='distplot')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_invalid_sdtype(mock_get_plot):
    """Test the ``get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``get_column_plot`` with an sdtype that can't be plotted, this raises
    an error.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='id')

    # Run and Assert
    error_msg = re.escape(
        "The column 'col' has sdtype 'id', which does not have a "
        "supported visualization. To visualize this data anyways, please add a 'plot_type'."
    )
    with pytest.raises(VisualizationUnavailableError, match=error_msg):
        get_column_plot(data1, data2, metadata, 'col')


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_invalid_sdtype_metadata(mock_get_plot):
    """Test the ``get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``get_column_plot`` with an sdtype that can't be plotted, this raises
    an error. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'id'}}}
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run and Assert
    error_msg = re.escape(
        "The column 'col' has sdtype 'id', which does not have a "
        "supported visualization. To visualize this data anyways, please add a 'plot_type'."
    )
    with pytest.raises(VisualizationUnavailableError, match=error_msg):
        get_column_plot(data1, data2, metadata, 'col')


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_invalid_sdtype_with_plot_type(mock_get_plot):
    """Test the ``get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``get_column_plot`` with an sdtype that can't be plotted, but passing
    ``plot_type`` it will attempt to plot it using the ``sdmetrics.visualization.get_column_plot``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='id')

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col', plot_type='bar')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_invalid_sdtype_with_plot_type_metadata(mock_get_plot):
    """Test the ``get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``get_column_plot`` with an sdtype that can't be plotted, but passing
    ``plot_type`` it will attempt to plot it using the ``sdmetrics.visualization.get_column_plot``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'id'}}}
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col', plot_type='bar')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test_get_column_plot_with_datetime_sdtype(mock_get_plot):
    """Test the ``get_column_plot`` with datetime sdtype.

    Test that when we call ``get_column_plot`` with ``datetime`` this will parse it using the
    datetime format provided in the metadata and it will cast it to ``datetime64``.
    """
    # Setup
    real_data = pd.DataFrame({'datetime': ['2021-02-01', '2021-12-01']})
    synthetic_data = pd.DataFrame({'datetime': ['2023-02-21', '2022-12-13']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('datetime', 'table', sdtype='datetime', datetime_format='%Y-%m-%d')

    # Run
    plot = get_column_plot(real_data, synthetic_data, metadata, 'datetime')

    # Assert
    expected_real_data = pd.DataFrame({'datetime': pd.to_datetime(['2021-02-01', '2021-12-01'])})
    expected_synth_data = pd.DataFrame({'datetime': pd.to_datetime(['2023-02-21', '2022-12-13'])})

    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], expected_real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], expected_synth_data)
    assert mock_get_plot.call_args[0][2] == 'datetime'
    assert mock_get_plot.call_args[1]['plot_type'] == 'distplot'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_continous_data(mock_get_plot):
    """Test ``get_column_pair_plot`` with continuous data.

    Test that when we call ``get_column_pair_plot`` with ``continuous`` data, this will
    automatically choose to use the ``scatter`` plot instead of the ``heatmap``.
    """
    # Setup
    columns = ['amount', 'date']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'date': ['2021-01-01', '2022-01-01', '2023-01-01'],
    })
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'date': ['2021-01-01', '2022-01-01', '2023-01-01'],
    })
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('date', 'table', sdtype='datetime')

    # Run
    plot = get_column_pair_plot(real_data, synthetic_data, metadata, columns)

    # Assert
    expected_real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    })
    expected_synth_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    })
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], expected_real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], expected_synth_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'scatter'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_discrete_data(mock_get_plot):
    """Test the ``get_column_pair_plot`` when using discrete data.

    Test that the ``get_column_pair_plot`` will automatically use ``heatmap`` if the data
    provided is discrete.
    """
    # Setup
    columns = ['name', 'subscriber']
    real_data = pd.DataFrame({'name': ['John', 'Emily'], 'subscriber': [True, False]})
    synthetic_data = pd.DataFrame({'name': ['John', 'Johanna'], 'subscriber': [False, False]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('name', 'table', sdtype='categorical')
    metadata.add_column('subscriber', 'table', sdtype='boolean')

    # Run
    plot = get_column_pair_plot(real_data, synthetic_data, metadata, columns)

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'heatmap'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_mixed_data(mock_get_plot):
    """Test the ``get_column_pair_plot`` with mixed data types.

    Test that when using both discrete and continuous data, we will be using automatically the
    ``box`` plot.
    """
    # Setup
    columns = ['name', 'counts']
    real_data = pd.DataFrame({'name': ['John', 'Emily'], 'counts': [1, 2]})
    synthetic_data = pd.DataFrame({'name': ['John', 'Johanna'], 'counts': [3, 1]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('name', 'table', sdtype='categorical')
    metadata.add_column('counts', 'table', sdtype='numerical')

    # Run
    plot = get_column_pair_plot(real_data, synthetic_data, metadata, columns)

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'box'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_forced_plot_type(mock_get_plot):
    """Test the ``get_column_pair_plot`` with continuous data and fixed plot type.

    Test that when using ``continuous`` data but asking to plot as ``heatmap`` this will still
    force the ``sdmetrics.visualization.get_column_pair_plot`` to use this.
    """
    # Setup
    columns = ['amount', 'date']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'date': ['2021-01-01', '2022-01-01', '2023-01-01'],
    })
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'date': ['2021-01-01', '2022-01-01', '2023-01-01'],
    })
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('date', 'table', sdtype='datetime')

    # Run
    plot = get_column_pair_plot(real_data, synthetic_data, metadata, columns, plot_type='heatmap')

    # Assert
    expected_real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    })
    expected_synth_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    })

    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], expected_real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], expected_synth_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'heatmap'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_invalid_sdtype(mock_get_plot):
    """Test the ``get_column_pair_plot`` with sdtype that can't be plotted.

    Test that when we call ``get_column_pair_plot`` with an sdtype that can't be plotted,
    this raises an error.
    """
    # Setup
    columns = ['amount', 'id']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'id': [1, 2, 3],
    })
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'id': [1, 2, 3],
    })
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('id', 'table', sdtype='id')

    # Run and Assert
    error_msg = re.escape(
        "The column 'id' has sdtype 'id', which does not have a "
        "supported visualization. To visualize this data anyways, please add a 'plot_type'."
    )
    with pytest.raises(VisualizationUnavailableError, match=error_msg):
        get_column_pair_plot(real_data, synthetic_data, metadata, columns)


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_invalid_sdtype_and_plot_type(mock_get_plot):
    """Test the ``get_column_pair_plot`` with sdtype that can't be plotted but providing plot type.

    Test that when providing the ``plot_type`` for an sdtype that can't be plotted, this will be
    plotted.
    """
    # Setup
    columns = ['amount', 'id']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'id': [1, 2, 3],
    })
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'id': [1, 2, 3],
    })
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('id', 'table', sdtype='id')

    # Run
    plot = get_column_pair_plot(real_data, synthetic_data, metadata, columns, plot_type='heatmap')

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'heatmap'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_sample_size(mock_get_plot):
    """Test ``get_column_pair_plot`` with ``sample_size`` parameter."""
    # Setup
    columns = ['amount', 'price']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'price': [10, 20, 30],
    })
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'price': [11.0, 22.0, 33.0],
    })
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('price', 'table', sdtype='numerical')

    # Run
    get_column_pair_plot(real_data, synthetic_data, metadata, columns, sample_size=2)

    # Assert
    real_subsample = mock_get_plot.call_args[0][0]
    synthetic_subsample = mock_get_plot.call_args[0][1]
    assert len(real_subsample) == 2
    assert len(synthetic_subsample) == 2
    assert real_subsample.isin(real_data).all().all()
    assert synthetic_subsample.isin(synthetic_data).all().all()


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_sample_size_metadata(mock_get_plot):
    """Test ``get_column_pair_plot`` with ``sample_size`` parameter with Metadata."""
    # Setup
    columns = ['amount', 'price']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'price': [10, 20, 30],
    })
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'price': [11.0, 22.0, 33.0],
    })
    metadata_dict = {
        'columns': {
            'amount': {'sdtype': 'numerical'},
            'price': {'sdtype': 'numerical'},
        }
    }
    metadata = Metadata.load_from_dict(metadata_dict)

    # Run
    get_column_pair_plot(real_data, synthetic_data, metadata, columns, sample_size=2)

    # Assert
    real_subsample = mock_get_plot.call_args[0][0]
    synthetic_subsample = mock_get_plot.call_args[0][1]
    assert len(real_subsample) == 2
    assert len(synthetic_subsample) == 2
    assert real_subsample.isin(real_data).all().all()
    assert synthetic_subsample.isin(synthetic_data).all().all()


@patch('sdmetrics.visualization.get_column_pair_plot')
def test_get_column_pair_plot_with_sample_size_too_big(mock_get_plot):
    """Test ``get_column_pair_plot`` when ``sample_size`` is bigger than the length of the data."""
    # Setup
    columns = ['amount', 'price']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'price': [10, 20, 30],
    })
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'price': [11.0, 22.0, 33.0],
    })
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('price', 'table', sdtype='numerical')

    # Run
    plot = get_column_pair_plot(real_data, synthetic_data, metadata, columns, sample_size=10)

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'scatter'
    assert plot == mock_get_plot.return_value
