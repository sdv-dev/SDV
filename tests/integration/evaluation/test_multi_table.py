import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.evaluation.multi_table import evaluate_quality, get_column_pair_plot, run_diagnostic
from sdv.metadata.metadata import Metadata


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
    metadata = Metadata().load_from_dict({
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


def test_evaluation_metadata():
    """Test ``evaluate_quality`` and ``run_diagnostic`` with Metadata."""
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
    metadata = Metadata().load_from_dict({
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


@pytest.mark.parametrize('plot_type', ['box', 'heatmap', 'scatter', 'violin'])
def test_column_pair_plot_different_plot_types(plot_type):
    """Test the method with all supported plot types."""
    # Setup
    real_data, metadata = download_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    fig = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=real_data.copy(),
        table_name='guests',
        column_names=['room_rate', 'amenities_fee'],
        metadata=metadata,
        sample_size=40,
        plot_type=plot_type,
    )

    # Assert
    assert len(fig.data[0].x) == 40
    assert len(fig.data[1].x) == 40
