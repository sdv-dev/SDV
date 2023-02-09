
from unittest.mock import Mock, patch

import pandas as pd

from sdv.evaluation.single_table import (
    DiagnosticReport, QualityReport, evaluate_quality, get_column_pair_plot, get_column_plot,
    run_diagnostic)
from sdv.metadata.single_table import SingleTableMetadata


def test_evaluate_quality():
    """Test ``generate`` is called for the ``QualityReport`` object."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    QualityReport.generate = Mock()

    # Run
    evaluate_quality(data1, data2, metadata)

    # Assert
    QualityReport.generate.assert_called_once_with(data1, data2, metadata.to_dict(), True)


def test_run_diagnostic():
    """Test ``generate`` is called for the ``DiagnosticReport`` object."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    DiagnosticReport.generate = Mock(return_value=123)

    # Run
    run_diagnostic(data1, data2, metadata)

    # Assert
    DiagnosticReport.generate.assert_called_once_with(data1, data2, metadata.to_dict(), True)


@patch('sdmetrics.reports.utils.get_column_plot')
def test_get_column_plot(mock_plot):
    """Test it calls ``get_column_plot`` in sdmetrics."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_plot(data1, data2, metadata, 'col')

    # Assert
    mock_plot.assert_called_once_with(data1, data2, 'col', metadata.to_dict())
    assert plot == 'plot'


@patch('sdmetrics.reports.utils.get_column_pair_plot')
def test_get_column_pair_plot(mock_plot):
    """Test it calls ``get_column_pair_plot`` in sdmetrics."""
    # Setup
    data1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]})
    data2 = pd.DataFrame({'col1': [2, 1, 3], 'col2': [1, 2, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col1', sdtype='numerical')
    metadata.add_column('col2', sdtype='numerical')
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_pair_plot(data1, data2, metadata, ['col1', 'col2'])

    # Assert
    mock_plot.assert_called_once_with(data1, data2, ['col1', 'col2'], metadata.to_dict())
    assert plot == 'plot'
