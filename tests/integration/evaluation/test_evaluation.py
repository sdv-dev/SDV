import pandas as pd
import pytest

from sdv.evaluation import evaluate_quality, run_diagnostic
from sdv.metadata.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


def _get_single_table_data():
    """Return single-table real data, synthetic data and metadata."""
    real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [0, 1, 2]})
    real_data = {'table': real_data}

    metadata = Metadata()
    metadata.add_table('table1')
    metadata.add_column('col1', 'table1', sdtype='numerical')
    metadata.add_column('col2', 'table1', sdtype='numerical')

    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        default_distribution='truncnorm',
    )
    synthesizer.fit(real_data)
    synthetic_data = {'table': synthesizer.sample(10)}

    return real_data, synthetic_data, metadata


def _get_multi_table_data():
    """Return multi-table real data, synthetic data and metadata."""
    table = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'col1': [1, 2, 3, 4.0],
        'col2': [0, 1, 2, 3],
    })
    table2 = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'col': [1, 2, 3, 4.0],
    })
    slightly_different_table2 = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'col': [1, 2, 3, 3.5],
    })

    real_data = {
        'table1': table,
        'table2': table2,
    }
    synthetic_data = {
        'table1': table,
        'table2': slightly_different_table2,
    }

    metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'col1': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'numerical'},
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
            },
        ],
    })

    return real_data, synthetic_data, metadata


def test_evaluation():
    """Test `evaluate_quality` and `run_diagnostic` with a dictionary of DataFrames."""
    # Setup
    real_data, synthetic_data, metadata = _get_multi_table_data()
    constraints = [
        {
            'class_name': 'Inequality',
            'parameters': {
                'table_name': 'table1',
                'high_column_name': 'col1',
                'low_column_name': 'col2',
            },
        }
    ]

    expected_quality_score = 0.9778889499863278
    expected_properties = pd.DataFrame({
        'Property': [
            'Data Validity',
            'Data Structure',
            'Relationship Validity',
            'Constraint Validity',
        ],
        'Score': [1.0, 1.0, 1.0, 1.0],
    })

    # Run
    quality_report = evaluate_quality(real_data, synthetic_data, metadata, verbose=False)
    diagnostic_report = run_diagnostic(
        real_data, synthetic_data, metadata, constraints, verbose=False
    )

    # Assert
    assert quality_report.get_score() == pytest.approx(expected_quality_score)
    assert diagnostic_report.get_score() == 1
    pd.testing.assert_frame_equal(
        diagnostic_report.get_properties(),
        expected_properties,
    )
