import pandas as pd
import pytest

from sdv.evaluation.utils import print_referential_integrity
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
    """A parent-child dataset whose primary and foreign keys span two columns."""
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
    """Test ``print_referential_integrity`` with a simple parent-child dataset."""
    # Setup
    synthetic_data, metadata = data_metadata_parent_child

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=4)

    # Assert
    captured = capsys.readouterr().out
    assert 'Picking random child row: A' in captured
    assert 'Picking random child row: B' in captured
    assert 'Picking random child row: C' in captured
    assert 'Picking random child row: D' in captured
    assert captured.count('✅ Found parent row! parent_id: 0') == 1
    assert captured.count('✅ Found parent row! parent_id: 1') == 2
    assert captured.count('❌ Unable to find the linked parent row') == 1


def test_print_referential_integrity_composite_keys(
    data_metadata_parent_child_composite_keys, capsys
):
    """Test ``print_referential_integrity`` with a parent-child dataset with composite keys.

    The child table has no primary key, so no key value is printed in the heading. Two of its
    ten rows reference a combination that is missing from the parent table.
    """
    # Setup
    synthetic_data, metadata = data_metadata_parent_child_composite_keys

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', ('A', 'B'), num_rows=10)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('Picking random child row\n') == 10
    assert captured.count('✅ Found parent row! A: 1, B: X') == 2
    assert captured.count('✅ Found parent row! A: 1, B: Y') == 1
    assert captured.count('✅ Found parent row! A: 2, B: X') == 1
    assert captured.count('✅ Found parent row! A: 3, B: Z') == 1
    assert captured.count('❌ Unable to find the linked parent row') == 5


def test_print_referential_integrity_with_null_foreign_key(data_metadata_parent_child, capsys):
    """Test that a null foreign key is reported as expected rather than as a broken link."""
    # Setup
    synthetic_data, metadata = data_metadata_parent_child
    synthetic_data['child'].loc[0, 'parent_id'] = None

    # Run
    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=4)

    # Assert
    captured = capsys.readouterr().out
    assert captured.count('✅ Foreign key is null; no linked parent row expected') == 1
