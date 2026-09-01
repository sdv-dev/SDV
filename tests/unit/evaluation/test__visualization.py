import re
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.errors import VisualizationUnavailableError
from sdv.evaluation._visualization import (
    _get_column_pair_plot,
    _get_column_plot,
    _prepare_data_visualization,
    get_cardinality_plot,
    get_column_pair_plot,
    get_column_plot,
)
from sdv.metadata._single_table import _SingleTableMetadata
from sdv.metadata.metadata import Metadata


def test__prepare_data_visualization():
    """Test ``_prepare_data_visualization``."""
    # Setup
    np.random.seed(0)
    metadata = _SingleTableMetadata.load_from_dict({
        'columns': {
            'col1': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'col2': {'sdtype': 'numerical'},
        }
    })
    column_names = ['col1', 'col2']
    sample_size = 2
    data = pd.DataFrame({
        'col1': ['2021-01-01', '2021-02-01', '2021-03-01'],
        'col2': [4, 5, 6],
    })

    # Run
    result = _prepare_data_visualization(data, metadata, column_names, sample_size)

    # Assert
    expected_result = pd.DataFrame(
        {
            'col1': pd.to_datetime(['2021-03-01', '2021-02-01']),
            'col2': [6, 5],
        },
        index=[2, 1],
    )
    pd.testing.assert_frame_equal(result, expected_result)


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_continuous_data(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with continuous data.

    Test that when we call ``_get_column_plot`` with continuous data (datetime or numerical)
    this will choose to use the ``distplot`` as ``plot_type``.
    """
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_continuous_data_metadata(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with continuous data.

    Test that when we call ``_get_column_plot`` with continuous data (datetime or numerical)
    this will choose to use the ``distplot`` as ``plot_type``. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata_dict = {'columns': {'col': {'sdtype': 'numerical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_discrete_data(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with discrete data.

    Test that when we call ``_get_column_plot`` with discrete data (categorical or boolean)
    this will choose to use the ``bar`` as ``plot_type``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='categorical')
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_discrete_data_metadata(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with discrete data.

    Test that when we call ``_get_column_plot`` with discrete data (categorical or boolean)
    this will choose to use the ``bar`` as ``plot_type``. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'categorical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_discrete_data_with_distplot(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with discrete data.

    Test that when we call ``_get_column_plot`` with discrete data (categorical or boolean)
    and pass in the ``distplot`` it will call the ``sdmetrics.visualization.get_column_plot``
    with it and not switch to ``bar``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='categorical')
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col', plot_type='distplot')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_discrete_data_with_distplot_metadata(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with discrete data.

    Test that when we call ``_get_column_plot`` with discrete data (categorical or boolean)
    and pass in the ``distplot`` it will call the ``sdmetrics.visualization.get_column_plot``
    with it and not switch to ``bar``. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'categorical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col', plot_type='distplot')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='distplot')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_invalid_sdtype(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``_get_column_plot`` with an sdtype that can't be plotted, this raises
    an error.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='id')
    mock_prepare.side_effect = [data1, data2]

    # Run and Assert
    error_msg = re.escape(
        "The column 'col' has sdtype 'id', which does not have a "
        "supported visualization. To visualize this data anyways, please add a 'plot_type'."
    )
    with pytest.raises(VisualizationUnavailableError, match=error_msg):
        _get_column_plot(data1, data2, metadata, 'col')


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_invalid_sdtype_metadata(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``_get_column_plot`` with an sdtype that can't be plotted, this raises
    an error. Uses Metadata.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'id'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    mock_prepare.side_effect = [data1, data2]

    # Run and Assert
    error_msg = re.escape(
        "The column 'col' has sdtype 'id', which does not have a "
        "supported visualization. To visualize this data anyways, please add a 'plot_type'."
    )
    with pytest.raises(VisualizationUnavailableError, match=error_msg):
        _get_column_plot(data1, data2, metadata, 'col')


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_invalid_sdtype_with_plot_type(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``_get_column_plot`` with an sdtype that can't be plotted, but passing
    ``plot_type`` it will attempt to plot it using the ``sdmetrics.visualization.get_column_plot``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='id')
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col', plot_type='bar')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_plot_invalid_sdtype_with_plot_type_metadata(mock_prepare, mock_get_plot):
    """Test the ``_get_column_plot`` with sdtype that can't be plotted.

    Test that when we call ``_get_column_plot`` with an sdtype that can't be plotted, but passing
    ``plot_type`` it will attempt to plot it using the ``sdmetrics.visualization.get_column_plot``.
    """
    # Setup
    data1 = pd.DataFrame({'col': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'col': ['a', 'b', 'c']})
    metadata_dict = {'columns': {'col': {'sdtype': 'id'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    mock_prepare.side_effect = [data1, data2]

    # Run
    plot = _get_column_plot(data1, data2, metadata, 'col', plot_type='bar')

    # Assert
    mock_get_plot.assert_called_once_with(data1, data2, 'col', plot_type='bar')
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test__get_column_plot_real_data_none(mock_get_plot):
    """Test ``_get_column_plot`` when ``real_data`` is None."""
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')

    # Run
    plot = _get_column_plot(None, data, metadata, 'col')

    # Assert
    mock_get_plot.call_args[0][1].equals(data)
    assert mock_get_plot.call_args[0][0] is None
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test__get_column_plot_synthetic_data_none(mock_get_plot):
    """Test ``_get_column_plot`` when ``synthetic_data`` is None."""
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')

    # Run
    plot = _get_column_plot(data, None, metadata, 'col')

    # Assert
    mock_get_plot.call_args[0][0].equals(data)
    assert mock_get_plot.call_args[0][1] is None
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_plot')
def test__get_column_plot_with_datetime_sdtype(mock_get_plot):
    """Test the ``_get_column_plot`` with datetime sdtype.

    Test that when we call ``_get_column_plot`` with ``datetime`` this will parse it using the
    datetime format provided in the metadata and it will cast it to ``datetime64``.
    """
    # Setup
    real_data = pd.DataFrame({'datetime': ['2021-02-01', '2021-12-01']})
    synthetic_data = pd.DataFrame({'datetime': ['2023-02-21', '2022-12-13']})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('datetime', 'table', sdtype='datetime', datetime_format='%Y-%m-%d')

    # Run
    plot = _get_column_plot(real_data, synthetic_data, metadata, 'datetime')

    # Assert
    expected_real_data = pd.DataFrame({'datetime': pd.to_datetime(['2021-02-01', '2021-12-01'])})
    expected_synth_data = pd.DataFrame({'datetime': pd.to_datetime(['2023-02-21', '2022-12-13'])})

    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], expected_real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], expected_synth_data)
    assert mock_get_plot.call_args[0][2] == 'datetime'
    assert mock_get_plot.call_args[1]['plot_type'] == 'distplot'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test__get_column_pair_plot_with_continous_data(mock_get_plot):
    """Test ``_get_column_pair_plot`` with continuous data.

    Test that when we call ``_get_column_pair_plot`` with ``continuous`` data, this will
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
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns)

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
@patch('sdv.evaluation._visualization._prepare_data_visualization')
def test__get_column_pair_plot_with_discrete_data(mock_prepare, mock_get_plot):
    """Test the ``_get_column_pair_plot`` when using discrete data.

    Test that the ``_get_column_pair_plot`` will automatically use ``heatmap`` if the data
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
    mock_prepare.side_effect = [real_data, synthetic_data]

    # Run
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns)

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'heatmap'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test__get_column_pair_plot_with_mixed_data(mock_get_plot):
    """Test the ``_get_column_pair_plot`` with mixed data types.

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
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns)

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'box'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test__get_column_pair_plot_with_forced_plot_type(mock_get_plot):
    """Test the ``_get_column_pair_plot`` with continuous data and fixed plot type.

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
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns, plot_type='heatmap')

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
def test__get_column_pair_plot_with_invalid_sdtype(mock_get_plot):
    """Test the ``_get_column_pair_plot`` with sdtype that can't be plotted.

    Test that when we call ``_get_column_pair_plot`` with an sdtype that can't be plotted,
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
        _get_column_pair_plot(real_data, synthetic_data, metadata, columns)


@patch('sdmetrics.visualization.get_column_pair_plot')
def test__get_column_pair_plot_with_invalid_sdtype_and_plot_type(mock_get_plot):
    """Test the ``_get_column_pair_plot`` with sdtype that can't be plotted but providing plot type.

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
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns, plot_type='heatmap')

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'heatmap'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test__get_column_pair_plot_with_sample_size(mock_get_plot):
    """Test ``_get_column_pair_plot`` with ``sample_size`` parameter."""
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
    _get_column_pair_plot(real_data, synthetic_data, metadata, columns, sample_size=2)

    # Assert
    real_subsample = mock_get_plot.call_args[0][0]
    synthetic_subsample = mock_get_plot.call_args[0][1]
    assert len(real_subsample) == 2
    assert len(synthetic_subsample) == 2
    assert real_subsample.isin(real_data).all().all()
    assert synthetic_subsample.isin(synthetic_data).all().all()


@patch('sdmetrics.visualization.get_column_pair_plot')
def test__get_column_pair_plot_with_sample_size_metadata(mock_get_plot):
    """Test ``_get_column_pair_plot`` with ``sample_size`` parameter with Metadata."""
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
    _get_column_pair_plot(real_data, synthetic_data, metadata, columns, sample_size=2)

    # Assert
    real_subsample = mock_get_plot.call_args[0][0]
    synthetic_subsample = mock_get_plot.call_args[0][1]
    assert len(real_subsample) == 2
    assert len(synthetic_subsample) == 2
    assert real_subsample.isin(real_data).all().all()
    assert synthetic_subsample.isin(synthetic_data).all().all()


@patch('sdmetrics.visualization.get_column_pair_plot')
def test__get_column_pair_plot_with_sample_size_too_big(mock_get_plot):
    """Test ``_get_column_pair_plot`` when ``sample_size`` is bigger than the length of the data."""
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
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns, sample_size=10)

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'scatter'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test___get_column_pair_plot_with_real_data_none(mock_get_plot):
    """Test ``_get_column_pair_plot`` when ``real_data`` is None."""
    # Setup
    columns = ['amount', 'price']
    real_data = None
    synthetic_data = pd.DataFrame({
        'amount': [1.0, 2.0, 3.0],
        'price': [11.0, 22.0, 33.0],
    })
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('price', 'table', sdtype='numerical')

    # Run
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns)

    # Assert
    assert mock_get_plot.call_args[0][0] is None
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][1], synthetic_data)
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'scatter'
    assert plot == mock_get_plot.return_value


@patch('sdmetrics.visualization.get_column_pair_plot')
def test___get_column_pair_plot_with_synthetic_data_none(mock_get_plot):
    """Test ``_get_column_pair_plot`` when ``synthetic_data`` is None."""
    # Setup
    columns = ['amount', 'price']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'price': [10, 20, 30],
    })
    synthetic_data = None
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('amount', 'table', sdtype='numerical')
    metadata.add_column('price', 'table', sdtype='numerical')

    # Run
    plot = _get_column_pair_plot(real_data, synthetic_data, metadata, columns)

    # Assert
    pd.testing.assert_frame_equal(mock_get_plot.call_args[0][0], real_data)
    assert mock_get_plot.call_args[0][1] is None
    assert mock_get_plot.call_args[0][2] == columns
    assert mock_get_plot.call_args[0][3] == 'scatter'
    assert plot == mock_get_plot.return_value


@patch('sdv.evaluation._visualization._get_column_plot')
def test_get_column_plot(mock_plot):
    """Test the ``get_column_plot``.

    Ensure that the ``get_column_plot`` is being called with the ``_SingleTableMetadata`` object
    and the expected table.
    """
    # Setup
    table1 = pd.DataFrame({'col': [1, 2, 3]})
    table2 = pd.DataFrame({'col': [2, 1, 3]})
    data1 = {'table': table1}
    data2 = {'table': table2}
    metadata = Metadata()
    metadata.detect_table_from_dataframe('table', table1)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_plot(data1, data2, metadata, 'table', 'col')

    # Assert
    call_metadata = metadata.tables['table']
    mock_plot.assert_called_once_with(table1, table2, call_metadata, 'col', None)
    assert plot == 'plot'


@patch('sdv.evaluation._visualization._get_column_plot')
def test_get_column_plot_only_real_or_synthetic(mock_plot):
    """Test that ``get_column_plot`` works when only real or synthetic data is provided."""
    # Setup
    table1 = pd.DataFrame({'col': [1, 2, 3]})
    data1 = {'table': table1}
    metadata = Metadata()
    metadata.detect_table_from_dataframe('table', table1)
    mock_plot.return_value = 'plot'

    # Run
    get_column_plot(data1, None, metadata, 'table', 'col')
    get_column_plot(None, data1, metadata, 'table', 'col')

    # Assert
    call_metadata = metadata.tables['table']
    mock_plot.assert_has_calls([
        ((table1, None, call_metadata, 'col', None), {}),
        ((None, table1, call_metadata, 'col', None), {}),
    ])


@patch('sdv.evaluation._visualization._get_column_pair_plot')
def test_get_column_pair_plot(mock_plot):
    """Test that ``get_column_pair_plot`` is being called with the expected objects."""
    # Setup
    table1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]})
    table2 = pd.DataFrame({'col1': [2, 1, 3], 'col2': [1, 2, 3]})
    data1 = {'table': table1}
    data2 = {'table': table2}
    metadata = Metadata()
    metadata.detect_table_from_dataframe('table', table1)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_pair_plot(data1, data2, metadata, 'table', ['col1', 'col2'], sample_size=2)

    # Assert
    call_metadata = metadata.tables['table']
    mock_plot.assert_called_once_with(
        real_data=table1,
        synthetic_data=table2,
        metadata=call_metadata,
        column_names=['col1', 'col2'],
        plot_type=None,
        sample_size=2,
    )
    assert plot == 'plot'


@patch('sdv.evaluation._visualization._get_column_pair_plot')
def test_get_column_pair_plot_only_real_or_synthetic(mock_plot):
    """Test that ``get_column_pair_plot`` works when only real or synthetic data is provided."""
    # Setup
    table1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]})
    data1 = {'table': table1}
    metadata = Metadata()
    metadata.detect_table_from_dataframe('table', table1)
    mock_plot.return_value = 'plot'

    # Run
    get_column_pair_plot(data1, None, metadata, 'table', ['col1', 'col2'], sample_size=2)
    get_column_pair_plot(None, data1, metadata, 'table', ['col1', 'col2'], sample_size=2)

    # Assert
    call_metadata = metadata.tables['table']
    call1 = call(
        real_data=table1,
        synthetic_data=None,
        metadata=call_metadata,
        column_names=['col1', 'col2'],
        plot_type=None,
        sample_size=2,
    )
    call2 = call(
        real_data=None,
        synthetic_data=table1,
        metadata=call_metadata,
        column_names=['col1', 'col2'],
        plot_type=None,
        sample_size=2,
    )
    mock_plot.assert_has_calls([call1, call2])


@patch('sdmetrics.visualization.get_cardinality_plot')
def test_get_cardinality_plot(mock_plot):
    """Test it calls ``get_column_cardinality_plot`` in sdmetrics with the parent primary key."""
    # Setup
    data1 = {
        'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]}),
        'table2': pd.DataFrame({'col1': [2, 2, 3], 'col2': [6, 7, 8]}),
    }
    data2 = {
        'table1': pd.DataFrame({'col1': [2, 1, 3], 'col2': [1, 2, 3]}),
        'table2': pd.DataFrame({'col1': [2, 2, 3], 'col2': [6, 7, 8]}),
    }
    metadata_dict = {
        'tables': {
            'table1': {
                'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}},
                'primary_key': 'col1',
            },
            'table2': {
                'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}
            },
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'col1',
                'child_foreign_key': 'col1',
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1',
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_cardinality_plot(data1, data2, 'table2', 'table1', 'col1', metadata)

    # Assert
    mock_plot.assert_called_once_with(data1, data2, 'table2', 'table1', 'col1', 'col1', 'bar')
    assert plot == 'plot'


@patch('sdmetrics.visualization.get_cardinality_plot')
def test_get_cardinality_plot_plot_type(mock_plot):
    """Test it calls ``get_column_cardinality_plot`` with different ``plot_type``."""
    # Setup
    data1 = {
        'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]}),
        'table2': pd.DataFrame({'col1': [2, 2, 3], 'col2': [6, 7, 8]}),
    }
    data2 = {
        'table1': pd.DataFrame({'col1': [2, 1, 3], 'col2': [1, 2, 3]}),
        'table2': pd.DataFrame({'col1': [2, 2, 3], 'col2': [6, 7, 8]}),
    }
    metadata_dict = {
        'tables': {
            'table1': {
                'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}},
                'primary_key': 'col1',
            },
            'table2': {
                'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}
            },
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'col1',
                'child_foreign_key': 'col1',
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1',
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_cardinality_plot(
        data1, data2, 'table2', 'table1', 'col1', metadata, plot_type='distplot'
    )

    # Assert
    mock_plot.assert_called_once_with(data1, data2, 'table2', 'table1', 'col1', 'col1', 'distplot')
    assert plot == 'plot'
