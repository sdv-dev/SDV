
import pandas as pd

from sdv.evaluation.multi_table import evaluate_quality, run_diagnostic
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.multi_table.hma import HMASynthesizer


def test_evaluation():
    """Test ``evaluate_quality`` and ``run_diagnostic``."""
    # Setup
    table = pd.DataFrame({'col': [1, 2, 3]})
    data = {'table': table}
    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('table', table)
    synthesizer = HMASynthesizer(metadata)

    # Run and Assert
    synthesizer.fit(data)
    samples = synthesizer.sample()
    score = evaluate_quality(data, samples, metadata).get_score()
    assert score == 0.6666666666666667

    diagnostic = run_diagnostic(data, samples, metadata).get_results()
    assert diagnostic == {
        'DANGER': ['More than 50% of the synthetic rows are copies of the real data'],
        'SUCCESS': [
            'The synthetic data covers over 90% of the numerical ranges present in the real data',
            'The synthetic data follows over 90% of the min/max boundaries set by the real data'
        ],
        'WARNING': []
    }
