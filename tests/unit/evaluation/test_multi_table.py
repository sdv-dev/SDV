
from unittest.mock import Mock, patch
import pandas as pd

from sdv.evaluation.multi_table import evaluate_quality, run_diagnostic, get_column_pair_plot, get_column_plot, QualityReport, DiagnosticReport
from sdv.metadata.multi_table import MultiTableMetadata


def test_evaluate_quality():
    """Test the correct score is returned."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', data1)
    QualityReport.generate = Mock()
    QualityReport.get_score = Mock(return_value=123)

    # Run
    score = evaluate_quality(data1, data2, metadata) 

    # Assert
    QualityReport.generate.assert_called_once_with(data1, data2, metadata.to_dict(), True)
    QualityReport.get_score.assert_called_once_with()
    assert score == 123


def test_run_diagnostic():
    """Test the correct diagnostic is returned."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', data1)
    DiagnosticReport.generate = Mock()
    DiagnosticReport.get_results = Mock(return_value={'err_type': 'str'})

    # Run
    diagnostic = run_diagnostic(data1, data2, metadata) 
    
    # Assert
    DiagnosticReport.generate.assert_called_once_with(data1, data2, metadata.to_dict(), True)
    DiagnosticReport.get_results.assert_called_once_with()
    assert diagnostic == {'err_type': 'str'}


@patch('sdmetrics.reports.utils.get_column_plot')
def test_get_column_plot(mock_plot):
    """Test it calls ``get_column_plot`` in sdmetrics."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', data1)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_plot(data1, data2, metadata, 'table', 'col')
    
    # Assert
    mock_plot.assert_called_once_with(data1, data2, 'col', metadata.to_dict())
    assert plot == 'plot'


@patch('sdmetrics.reports.utils.get_column_pair_plot')
def test_get_column_pair_plot(mock_plot):
    """Test it calls ``get_column_pair_plot`` in sdmetrics."""
    # Setup
    data1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 2, 1]})
    data2 = pd.DataFrame({'col1': [2, 1, 3], 'col2': [1, 2, 3]})
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', data1)
    mock_plot.return_value = 'plot'

    # Run
    plot = get_column_pair_plot(data1, data2, metadata, 'table', ['col1', 'col2']) 
    
    # Assert
    mock_plot.assert_called_once_with(data1, data2, ['col1', 'col2'], metadata.to_dict())
    assert plot == 'plot'
