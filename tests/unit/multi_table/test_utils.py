from collections import defaultdict
from unittest.mock import Mock

import pandas as pd

from sdv.multi_table import HMASynthesizer
from sdv.multi_table.utils import (
    _get_relationship_for_child, _get_relationship_for_parent, _get_rows_to_drop)


def test__get_root_tables():
    """Test the ``_get_root_tables`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]

    # Run
    result = HMASynthesizer._get_root_tables(relationships)

    # Assert
    assert result == {'parent'}


def test__get_relationship_for_child():
    """Test the ``_get_relationship_for_child`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]

    # Run
    result = _get_relationship_for_child(relationships, 'grandchild')

    # Assert
    expected_result = [
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]
    assert result == expected_result


def test__get_relationship_for_parent():
    """Test the ``_get_relationship_for_parent`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]

    # Run
    result = _get_relationship_for_parent(relationships, 'parent')

    # Assert
    expected_result = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]
    assert result == expected_result


def test__get_rows_to_drop():
    """Test the ``_get_rows_to_drop`` method.

    In the child table:
        - Index 4 is removed because its foreign key doesn't match any primary key in the parent
        table.

    In the grandchild table:
        - Index 2 is removed because its foreign key doesn't match any primary key in the child
        table.
        - Index 4 is removed due to its foreign key not aligning with any parent table primary key.
        - Index 0 is removed following the removal of index 4 from the child table, which
        invalidates the foreign key set to 9 in the grandchild table.
    """
    # Setup
    relationships = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key'
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key'
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key'
        }
    ]

    metadata = Mock()
    metadata.relationships = relationships
    metadata.tables = {
        'parent': Mock(primary_key='id_parent'),
        'child': Mock(primary_key='id_child'),
        'grandchild': Mock(primary_key='id_grandchild')
    }

    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5],
            'id_child': [5, 6, 7, 8, 9],
            'B': ['Yes', 'No', 'No', 'No', 'No']
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6],
            'child_foreign_key': [9, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No']
        })
    }

    # Run
    result = _get_rows_to_drop(metadata, data)

    # Assert
    expected_result = defaultdict(set, {
        'child': {4},
        'grandchild': {0, 2, 4},
        'parent': set()
    })
    assert result == expected_result
