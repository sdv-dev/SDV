import pandas as pd
import pytest

from sdv.evaluation import evaluate_quality, run_diagnostic
from sdv.metadata.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


def _get_single_table_data():
    """Return single-table real data, synthetic data and metadata."""
    real_data = pd.DataFrame({'col': [1, 2, 3]})

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')

    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        default_distribution='truncnorm',
    )
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(10)

    return real_data, synthetic_data, metadata


def _get_multi_table_data():
    """Return multi-table real data, synthetic data and metadata."""
    table = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'col': [1, 2, 3, 4.0],
    })
    slightly_different_table = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'col': [1, 2, 3, 3.5],
    })

    real_data = {
        'table1': table,
        'table2': table,
    }
    synthetic_data = {
        'table1': table,
        'table2': slightly_different_table,
    }

    metadata = Metadata.load_from_dict({
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
            },
        ],
    })

    return real_data, synthetic_data, metadata


@pytest.mark.parametrize(
    (
        'data_function',
        'expected_quality_score',
        'expected_properties',
    ),
    [
        pytest.param(
            _get_single_table_data,
            0.8666666666666667,
            pd.DataFrame({
                'Property': ['Data Validity', 'Data Structure'],
                'Score': [1.0, 1.0],
            }),
            id='dataframe',
        ),
        pytest.param(
            _get_multi_table_data,
            0.9566297110928815,
            pd.DataFrame({
                'Property': [
                    'Data Validity',
                    'Data Structure',
                    'Relationship Validity',
                ],
                'Score': [1.0, 1.0, 1.0],
            }),
            id='dictionary',
        ),
    ],
)
def test_evaluation(
    data_function,
    expected_quality_score,
    expected_properties,
):
    """Test `evaluate_quality` and `run_diagnostic` with DataFrames and dictionaries."""
    # Setup
    real_data, synthetic_data, metadata = data_function()

    # Run
    quality_report = evaluate_quality(real_data, synthetic_data, metadata, verbose=False)
    diagnostic_report = run_diagnostic(real_data, synthetic_data, metadata, verbose=False)

    # Assert
    assert quality_report.get_score() == pytest.approx(expected_quality_score)
    assert diagnostic_report.get_score() == 1
    pd.testing.assert_frame_equal(
        diagnostic_report.get_properties(),
        expected_properties,
    )
