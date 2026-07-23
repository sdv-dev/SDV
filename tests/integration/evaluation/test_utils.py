import pandas as pd
import pytest

from sdv.evaluation import print_referential_integrity
from sdv.metadata import Metadata


@pytest.fixture()
def data_metadata_parent_child():
    """A parent-child dataset with a primary to foreign key relationship between 1 column."""
    data = {
        'parent': pd.DataFrame({'parent_id': [0, 1, 2, 3], 'col': [1, 2, 3, 4]}),
        'child': pd.DataFrame({'child_id': ['A', 'B', 'C', 'D'], 'parent_id': [0, 1, 1, 4]}),
    }
    metadata = Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {
                    'parent_id': {'sdtype': 'id'},
                    'col': {'sdtype': 'numerical'},
                },
                'primary_key': 'parent_id',
            },
            'child': {
                'columns': {
                    'child_id': {'sdtype': 'id'},
                    'parent_id': {'sdtype': 'id'},
                },
                'primary_key': 'child_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'parent_id',
                'child_table_name': 'child',
                'child_foreign_key': 'parent_id',
            }
        ],
    })
    return data, metadata


@pytest.fixture()
def data_metadata_parent_child_composite_keys():
    data = {
        'parent': pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': ['X', 'Y', 'X', 'Z'],
        }),
        'child': pd.DataFrame({
            'A': [1, 1, 2, 3, 1, 1, 2, 3, 4, 2],
            'B': ['X', 'Y', 'X', 'Z', 'X', 'Z', 'Y', 'X', 'W', 'Z'],
        }),
    }
    metadata = Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {
                    'A': {'sdtype': 'id'},
                    'B': {'sdtype': 'id'},
                },
                'primary_key': ['A', 'B'],
            },
            'child': {
                'columns': {
                    'A': {'sdtype': 'id'},
                    'B': {'sdtype': 'id'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': ['A', 'B'],
                'child_table_name': 'child',
                'child_foreign_key': ['A', 'B'],
            },
        ],
    })
    return data, metadata


def test_print_referential_integrity(data_metadata_parent_child, capsys):
    """Test `print_referential_integrity` with a simple parent-child dataset."""
    # Setup
    synthetic_data, metadata = data_metadata_parent_child

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id')

    # Assert
    captured = capsys.readouterr().out
    assert 'Picking random child row: A' in captured
    assert '✅ Found parent row! parent_id: 0' in captured
    assert 'Picking random child row: C' in captured
    assert '✅ Found parent row! parent_id: 1' in captured
    assert 'Picking random child row: D' in captured
    assert '❌ Unable to find the linked parent row' in captured
    assert 'Picking random child row: B' in captured
    assert '✅ Found parent row! parent_id: 1' in captured


def test_print_referential_integrity_composite_keys(
    data_metadata_parent_child_composite_keys, capsys
):
    """Test `print_referential_integrity` with a parent-child dataset with composite keys."""
    # Setup
    synthetic_data, metadata = data_metadata_parent_child_composite_keys()

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', ['A', 'B'])

    # Assert
    captured = capsys.readouterr().out
