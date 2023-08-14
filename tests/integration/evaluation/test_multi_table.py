
import pandas as pd

from sdv.evaluation.multi_table import evaluate_quality, run_diagnostic
from sdv.metadata.multi_table import MultiTableMetadata


def test_evaluation():
    """Test ``evaluate_quality`` and ``run_diagnostic``."""
    # Setup
    table = pd.DataFrame({'id': [0, 1, 2, 3], 'col': [1, 2, 3, 4]})
    slightly_different_table = pd.DataFrame({'id': [0, 1, 2, 3], 'col': [1, 2, 3, 3.5]})
    data = {
        'table1': table,
        'table2': table,
    }
    samples = {
        'table1': table,
        'table2': slightly_different_table,
    }
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'col': {'sdtype': 'numerical'},
                },
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'col': {'sdtype': 'numerical'},
                },
            }
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'parent_primary_key': 'id',
                'child_table_name': 'table2',
                'child_foreign_key': 'id'
            }
        ]
    })

    # Run and Assert
    score = evaluate_quality(data, samples, metadata).get_score()
    assert score == .9375

    diagnostic = run_diagnostic(data, samples, metadata).get_results()
    assert diagnostic == {
        'DANGER': ['More than 50% of the synthetic rows are copies of the real data'],
        'SUCCESS': [
            'The synthetic data covers over 90% of the numerical ranges present in the real data',
            'The synthetic data follows over 90% of the min/max boundaries set by the real data'
        ],
        'WARNING': []
    }
