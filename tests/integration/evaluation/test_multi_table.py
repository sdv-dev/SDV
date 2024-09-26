import pandas as pd

from sdv.evaluation.multi_table import evaluate_quality, run_diagnostic
from sdv.metadata.multi_table import MultiTableMetadata


def test_evaluation():
    """Test ``evaluate_quality`` and ``run_diagnostic``."""
    # Setup
    table = pd.DataFrame({'id': [0, 1, 2, 3], 'col': [1, 2, 3, 4.0]})
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
            },
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'parent_primary_key': 'id',
                'child_table_name': 'table2',
                'child_foreign_key': 'id',
            }
        ],
    })

    # Run and Assert
    score = evaluate_quality(data, samples, metadata).get_score()
    assert score == 0.9566297110928815

    report = run_diagnostic(data, samples, metadata)
    assert report.get_score() == 1
    pd.testing.assert_frame_equal(
        report.get_properties(),
        pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure', 'Relationship Validity'],
            'Score': [1.0, 1.0, 1.0],
        }),
    )
