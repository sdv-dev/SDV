
import pandas as pd

from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


def test_evaluation():
    """Test ``evaluate_quality`` and ``run_diagnostic``."""
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    synthesizer = GaussianCopulaSynthesizer(metadata)

    # Run and Assert
    synthesizer.fit(data)
    samples = synthesizer.sample(10)
    score = evaluate_quality(data, samples, metadata).get_score()
    assert score == 0.8333333333333334

    diagnostic = run_diagnostic(data, samples, metadata).get_results()
    assert diagnostic == {
        'DANGER': ['More than 50% of the synthetic rows are copies of the real data'],
        'SUCCESS': [
            'The synthetic data covers over 90% of the numerical ranges present in the real data',
            'The synthetic data follows over 90% of the min/max boundaries set by the real data'
        ],
        'WARNING': []
    }
