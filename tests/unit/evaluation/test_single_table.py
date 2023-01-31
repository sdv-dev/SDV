
from unittest.mock import patch
import pandas as pd

from sdv.evaluation.single_table import evaluate_quality, run_diagnostic, get_column_pair_plot, get_column_plot
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


@patch('sdmetrics.reports.single_table.quality_report.QualityReport')
def test_evaluate_quality(mock_report):
    """Test the correct score is returned."""
    # Setup
    data1 = pd.DataFrame({'col': [1, 2, 3]})
    data2 = pd.DataFrame({'col': [2, 1, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    mock_report.get_score.return_value = 123

    # Run
    score = evaluate_quality(data1, data2, metadata) 
    
    # Assert
    mock_report.generate.assert_called_once_with(data1, data2, metadata, True)
    mock_report.get_score.assert_called_once_with()
    assert score == 123

def test_run_diagnostic():
    """Test ."""
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    synthesizer = GaussianCopulaSynthesizer(metadata)

    # Run
    synthesizer.fit(data)
    samples = synthesizer.sample(10)
    diagnostic = run_diagnostic(data, samples, metadata)

def test_get_column_plot():
    """Test."""
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    samples = synthesizer.sample(10)
    get_column_plot(data, samples, metadata, 'col')  # method produces non uniform for [1,2,3] data, maybe bugged? 

def test_get_column_pair_plot():
    """Test."""
    data = pd.DataFrame({'col': [1, 2, 3], 'col2': [1, 2, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    metadata.add_column('col2', sdtype='numerical')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    samples = synthesizer.sample(10)
    get_column_pair_plot(data, samples, metadata, ['col', 'col2'])
