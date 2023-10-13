
from unittest.mock import Mock, patch

import pandas as pd

from sdv.evaluation.multi_table import (
    DiagnosticReport, QualityReport, evaluate_quality, get_cardinality_plot, get_column_pair_plot,
    get_column_plot, run_diagnostic)
from sdv.metadata.multi_table import MultiTableMetadata


def test_evaluate_quality():
    """Test ``generate`` is called for the ``QualityReport`` object."""
    # Setup
    table = pd.DataFrame({'col': [1, 2, 3]})
    data1 = {'table': table}
    data2 = {'table': pd.DataFrame({'col': [2, 1, 3]})}
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', table)
    QualityReport.generate = Mock()

    # Run
    evaluate_quality(data1, data2, metadata)

    # Assert
    QualityReport.generate.assert_called_once_with(data1, data2, metadata.to_dict(), True)


def test_run_diagnostic():
    """Test ``generate`` is called for the ``DiagnosticReport`` object."""
    # Setup
    table = pd.DataFrame({'col': [1, 2, 3]})
    data1 = {'table': table}
    data2 = {'table': pd.DataFrame({'col': [2, 1, 3]})}
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', table)
    DiagnosticReport.generate = Mock()

    # Run
    run_diagnostic(data1, data2, metadata)

    # Assert
    DiagnosticReport.generate.assert_called_once_with(data1, data2, metadata.to_dict(), True)


@patch('sdmetrics.reports.utils.get_column_plot')
def test_get_column_plot(mock_plot):
    """Test it calls ``get_column_plot`` in sdmetrics."""
    # Setup
    table1 = pd.DataFrame({'col': [1, 2, 3]})
    table2 = pd.DataFrame({'col': [2, 1, 3]})
    data1 = {'table': table1}
    data2 = {'table': table2}
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', table1)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_plot(data1, data2, metadata, 'table', 'col')

    # Assert
    call_metadata = {'columns': {'col': {'sdtype': 'numerical'}}}
    mock_plot.assert_called_once_with(table1, table2, 'col', call_metadata)
    assert plot == 'plot'


@patch('sdmetrics.reports.utils.get_column_pair_plot')
def test_get_column_pair_plot(mock_plot):
    """Test it calls ``get_column_pair_plot`` in sdmetrics."""
    # Setup
    table1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]})
    table2 = pd.DataFrame({'col1': [2, 1, 3], 'col2': [1, 2, 3]})
    data1 = {'table': table1}
    data2 = {'table': table2}
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', table1)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_pair_plot(data1, data2, metadata, 'table', ['col1', 'col2'])

    # Assert
    call_metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}}
    mock_plot.assert_called_once_with(table1, table2, ['col1', 'col2'], call_metadata)
    assert plot == 'plot'


@patch('sdmetrics.reports.utils.get_cardinality_plot')
def test_get_cardinality_plot(mock_plot):
    """Test it calls ``get_column_cardinality_plot`` in sdmetrics."""
    # Setup
    data1 = {
        'table1': pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]}),
        'table2': pd.DataFrame({'col1': [2, 2, 3], 'col2': [6, 7, 8]})
    }
    data2 = {
        'table1': pd.DataFrame({'col1': [2, 1, 3], 'col2': [1, 2, 3]}),
        'table2': pd.DataFrame({'col1': [2, 2, 3], 'col2': [6, 7, 8]})
    }
    metadata_dict = {
        'tables': {
            'table1': {
                'columns': {
                    'col1': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'numerical'}
                },
                'primary_key': 'col1'
            },
            'table2': {
                'columns': {
                    'col1': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'numerical'}
                }
            }
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'col1',
                'child_foreign_key': 'col1'
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }
    metadata = MultiTableMetadata.load_from_dict(metadata_dict)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_cardinality_plot(data1, data2, 'table2', 'table1', 'col1', metadata)

    # Assert
    mock_plot.assert_called_once_with(data1, data2, 'table2', 'table1', 'col1', metadata_dict)
    assert plot == 'plot'
