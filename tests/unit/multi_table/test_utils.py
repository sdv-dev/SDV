import re
from collections import defaultdict
from copy import deepcopy
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.errors import InvalidDataError, SamplingError
from sdv.metadata.metadata import Metadata
from sdv.multi_table.utils import (
    _drop_rows,
    _get_all_descendant_per_root_at_order_n,
    _get_ancestors,
    _get_columns_to_drop_child,
    _get_disconnected_roots_from_table,
    _get_n_order_descendants,
    _get_nan_fk_indices_table,
    _get_num_column_to_drop,
    _get_primary_keys_referenced,
    _get_relationships_for_child,
    _get_relationships_for_parent,
    _get_rows_to_drop,
    _get_total_estimated_columns,
    _print_simplified_schema_summary,
    _print_subsample_summary,
    _simplify_child,
    _simplify_children,
    _simplify_data,
    _simplify_grandchildren,
    _simplify_metadata,
    _simplify_relationships_and_tables,
    _subsample_ancestors,
    _subsample_data,
    _subsample_disconnected_roots,
    _subsample_parent,
    _subsample_table_and_descendants,
)


def test__get_relationships_for_child():
    """Test the ``_get_relationships_for_child`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'},
    ]

    # Run
    result = _get_relationships_for_child(relationships, 'grandchild')

    # Assert
    expected_result = [
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'},
    ]
    assert result == expected_result


def test__get_relationships_for_parent():
    """Test the ``_get_relationships_for_parent`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'},
    ]

    # Run
    result = _get_relationships_for_parent(relationships, 'parent')

    # Assert
    expected_result = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'},
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
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
    ]

    metadata = Mock()
    metadata.relationships = relationships
    metadata.tables = {
        'parent': Mock(primary_key='id_parent'),
        'child': Mock(primary_key='id_child'),
        'grandchild': Mock(primary_key='id_grandchild'),
    }

    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5],
            'id_child': [5, 6, 7, 8, 9],
            'B': ['Yes', 'No', 'No', 'No', 'No'],
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6],
            'child_foreign_key': [9, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No'],
        }),
    }

    # Run
    result = _get_rows_to_drop(data, metadata)

    # Assert
    expected_result = defaultdict(set, {'child': {4}, 'grandchild': {0, 2, 4}, 'parent': set()})
    assert result == expected_result


def test__get_nan_fk_indices_table():
    """Test the ``_get_nan_fk_indices_table`` method."""
    # Setup
    relationships = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
    ]
    data = {
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [np.nan, 1, 2, 2, np.nan],
            'child_foreign_key': [9, np.nan, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No'],
        })
    }

    # Run
    result = _get_nan_fk_indices_table(data, relationships, 'grandchild')

    # Assert
    assert result == {0, 1, 4}


@patch('sdv.multi_table.utils._get_rows_to_drop')
def test__drop_rows(mock_get_rows_to_drop):
    """Test the ``_drop_rows`` method."""
    # Setup
    relationships = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
    ]

    metadata = Mock()
    metadata.relationships = relationships
    metadata.tables = {
        'parent': Mock(primary_key='id_parent'),
        'child': Mock(primary_key='id_child'),
        'grandchild': Mock(primary_key='id_grandchild'),
    }
    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5],
            'id_child': [5, 6, 7, 8, 9],
            'B': ['Yes', 'No', 'No', 'No', 'No'],
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6],
            'child_foreign_key': [9, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No'],
        }),
    }

    mock_get_rows_to_drop.return_value = defaultdict(set, {'child': {4}, 'grandchild': {0, 2, 4}})

    # Run
    _drop_rows(data, metadata, False)

    # Assert
    mock_get_rows_to_drop.assert_called_once_with(data, metadata)
    expected_result = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame(
            {
                'parent_foreign_key': [0, 1, 2, 2],
                'id_child': [5, 6, 7, 8],
                'B': ['Yes', 'No', 'No', 'No'],
            },
            index=[0, 1, 2, 3],
        ),
        'grandchild': pd.DataFrame(
            {'parent_foreign_key': [1, 2], 'child_foreign_key': [5, 6], 'C': ['No', 'No']},
            index=[1, 3],
        ),
    }
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, expected_result[table_name])


@patch('sdv.multi_table.utils._get_rows_to_drop')
def test_drop_unknown_references_with_nan(mock_get_rows_to_drop):
    """Test ``drop_unknown_references`` whith NaNs and drop_missing_values True."""
    # Setup
    relationships = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
    ]

    metadata = Mock()
    metadata.relationships = relationships
    metadata.tables = {'parent', 'child', 'grandchild'}

    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5, None],
            'id_child': [5, 6, 7, 8, 9, 10],
            'B': ['Yes', 'No', 'No', 'No', 'No', 'No'],
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6, 4],
            'child_foreign_key': [9, np.nan, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No', 'No'],
        }),
    }
    mock_get_rows_to_drop.return_value = defaultdict(set, {'child': {4}, 'grandchild': {0, 3, 4}})

    # Run
    _drop_rows(data, metadata, True)

    # Assert
    mock_get_rows_to_drop.assert_called_once()
    expected_result = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame(
            {
                'parent_foreign_key': [0.0, 1.0, 2.0, 2.0],
                'id_child': [5, 6, 7, 8],
                'B': ['Yes', 'No', 'No', 'No'],
            },
            index=[0, 1, 2, 3],
        ),
        'grandchild': pd.DataFrame(
            {'parent_foreign_key': [2, 4], 'child_foreign_key': [5.0, 4.0], 'C': ['No', 'No']},
            index=[2, 5],
        ),
    }
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, expected_result[table_name])


@patch('sdv.multi_table.utils._get_rows_to_drop')
def test_drop_unknown_references_drop_missing_values_false(mock_get_rows_to_drop):
    """Test ``drop_unknown_references`` with NaNs and drop_missing_values False."""
    # Setup
    relationships = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
    ]

    metadata = Mock()
    metadata.relationships = relationships
    metadata.tables = {'parent', 'child', 'grandchild'}
    metadata.validate_data.side_effect = InvalidDataError('Invalid data')

    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5, None],
            'id_child': [5, 6, 7, 8, 9, 10],
            'B': ['Yes', 'No', 'No', 'No', 'No', 'No'],
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6, 4],
            'child_foreign_key': [9, np.nan, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No', 'No'],
        }),
    }
    mock_get_rows_to_drop.return_value = defaultdict(set, {'child': {4}, 'grandchild': {0, 3, 4}})

    # Run
    _drop_rows(data, metadata, drop_missing_values=False)

    # Assert
    mock_get_rows_to_drop.assert_called_once()
    expected_result = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame(
            {
                'parent_foreign_key': [0.0, 1.0, 2.0, 2.0, None],
                'id_child': [5, 6, 7, 8, 10],
                'B': ['Yes', 'No', 'No', 'No', 'No'],
            },
            index=[0, 1, 2, 3, 5],
        ),
        'grandchild': pd.DataFrame(
            {
                'parent_foreign_key': [1, 2, 4],
                'child_foreign_key': [np.nan, 5, 4.0],
                'C': ['No', 'No', 'No'],
            },
            index=[1, 2, 5],
        ),
    }
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, expected_result[table_name])


@patch('sdv.multi_table.utils._get_rows_to_drop')
def test_drop_unknown_references_drop_all_rows(mock_get_rows_to_drop):
    """Test ``drop_unknown_references`` when all rows are dropped."""
    # Setup
    relationships = [
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'grandchild',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
    ]

    metadata = Mock()
    metadata.relationships = relationships
    metadata.tables = {'parent', 'child', 'grandchild'}
    metadata.validate_data.side_effect = InvalidDataError('Invalid data')

    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5],
            'id_child': [5, 6, 7, 8, 9],
            'B': ['Yes', 'No', 'No', 'No', 'No'],
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6],
            'child_foreign_key': [9, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No'],
        }),
    }

    mock_get_rows_to_drop.return_value = defaultdict(set, {'child': {0, 1, 2, 3, 4}})

    # Run and Assert
    expected_message = re.escape(
        'The provided data does not match the metadata:\n'
        "All references in table 'child' are unknown and must be dropped. "
        'Try providing different data for this table.'
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        _drop_rows(data, metadata, False)


def test__get_n_order_descendants():
    """Test the ``_get_n_order_descendants`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
    ]

    # Run
    grandparent_order_1 = _get_n_order_descendants(relationships, 'grandparent', 1)
    grandparent_order_2 = _get_n_order_descendants(relationships, 'grandparent', 2)
    grandparent_order_3 = _get_n_order_descendants(relationships, 'grandparent', 3)
    other_order_2 = _get_n_order_descendants(relationships, 'other_table', 2)

    # Assert
    expected_gp_order_1 = {
        'order_1': ['parent', 'other_table'],
    }
    expected_gp_order_2 = {'order_1': ['parent', 'other_table'], 'order_2': ['child']}
    expected_gp_order_3 = {
        'order_1': ['parent', 'other_table'],
        'order_2': ['child'],
        'order_3': ['grandchild'],
    }
    expected_other_order_2 = {'order_1': [], 'order_2': []}
    assert grandparent_order_1 == expected_gp_order_1
    assert grandparent_order_2 == expected_gp_order_2
    assert grandparent_order_3 == expected_gp_order_3
    assert other_order_2 == expected_other_order_2


def test__get_all_descendant_per_root_at_order_n():
    """Test the ``_get_all_descendant_per_root_at_order_n`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
        {'parent_table_name': 'other_root', 'child_table_name': 'child'},
    ]

    # Run
    result = _get_all_descendant_per_root_at_order_n(relationships, 3)

    # Assert
    expected_result = {
        'other_root': {
            'order_1': ['child'],
            'order_2': ['grandchild'],
            'order_3': [],
            'num_descendants': 2,
        },
        'grandparent': {
            'order_1': ['parent', 'other_table'],
            'order_2': ['child'],
            'order_3': ['grandchild'],
            'num_descendants': 4,
        },
    }
    assert result == expected_result


@pytest.mark.parametrize(
    ('table_name', 'expected_result'),
    [
        ('grandchild', {'child', 'parent', 'grandparent', 'other_root'}),
        ('child', {'parent', 'grandparent', 'other_root'}),
        ('parent', {'grandparent'}),
        ('other_table', {'grandparent'}),
        ('grandparent', set()),
        ('other_root', set()),
    ],
)
def test__get_ancestors(table_name, expected_result):
    """Test the ``_get_ancestors`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
        {'parent_table_name': 'other_root', 'child_table_name': 'child'},
    ]

    # Run
    result = _get_ancestors(relationships, table_name)

    # Assert
    assert result == expected_result


@pytest.mark.parametrize(
    ('table_name', 'expected_result'),
    [
        ('grandchild', {'disconnected_root'}),
        ('child', {'disconnected_root'}),
        ('parent', {'disconnected_root'}),
        ('other_table', {'disconnected_root', 'other_root'}),
        ('grandparent', {'disconnected_root'}),
        ('other_root', {'disconnected_root'}),
        ('disconnected_root', {'grandparent', 'other_root'}),
        ('disconnect_child', {'grandparent', 'other_root'}),
    ],
)
def test__get_disconnected_roots_from_table(table_name, expected_result):
    """Test the ``_get_disconnected_roots_from_table`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
        {'parent_table_name': 'other_root', 'child_table_name': 'child'},
        {'parent_table_name': 'disconnected_root', 'child_table_name': 'disconnect_child'},
    ]

    # Run
    result = _get_disconnected_roots_from_table(relationships, table_name)

    # Assert
    assert result == expected_result


def test__simplify_relationships_and_tables():
    """Test the ``_simplify_relationships`` method."""
    # Setup
    relationship_extras = {'parent_primary_key': 'pk', 'child_foreign_key': 'fk'}
    metadata = Metadata().load_from_dict({
        'tables': {
            'grandparent': {'columns': {'col_1': {'sdtype': 'numerical'}}},
            'parent': {'columns': {'col_2': {'sdtype': 'numerical'}}},
            'child': {'columns': {'col_3': {'sdtype': 'numerical'}}},
            'grandchild': {'columns': {'col_4': {'sdtype': 'numerical'}}},
            'other_table': {'columns': {'col_5': {'sdtype': 'numerical'}}},
            'other_root': {'columns': {'col_6': {'sdtype': 'numerical'}}},
        },
        'relationships': [
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'parent',
                **relationship_extras
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                **relationship_extras
            },
            {
                'parent_table_name': 'child',
                'child_table_name': 'grandchild',
                **relationship_extras
            },
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'other_table',
                **relationship_extras
            },
            {
                'parent_table_name': 'other_root',
                'child_table_name': 'child',
                **relationship_extras
            },
        ],
    })
    tables_to_drop = {'grandchild', 'other_root'}

    # Run
    _simplify_relationships_and_tables(metadata, tables_to_drop)

    # Assert
    expected_relationships = [
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent', **relationship_extras},
        {'parent_table_name': 'parent', 'child_table_name': 'child', **relationship_extras},
        {'parent_table_name': 'grandparent', 'child_table_name': 'other_table', **relationship_extras},
    ]
    expected_tables = {
        'grandparent': {'columns': {'col_1': {'sdtype': 'numerical'}}},
        'parent': {'columns': {'col_2': {'sdtype': 'numerical'}}},
        'child': {'columns': {'col_3': {'sdtype': 'numerical'}}},
        'other_table': {'columns': {'col_5': {'sdtype': 'numerical'}}},
    }
    assert metadata.relationships == expected_relationships
    assert metadata.to_dict()['tables'] == expected_tables


def test__simplify_grandchildren():
    """Test the ``_simplify_grandchildren`` method."""
    # Setup
    metadata = Metadata().load_from_dict({
        'tables': {
            'grandparent': {'columns': {'col_1': {'sdtype': 'numerical'}}},
            'parent': {'columns': {'col_2': {'sdtype': 'numerical'}}},
            'child_1': {
                'columns': {
                    'col_3': {'sdtype': 'numerical'},
                    'col_4': {'sdtype': 'categorical'},
                    'col_5': {'sdtype': 'boolean'},
                    'col_7': {'sdtype': 'email'},
                    'col_8': {'sdtype': 'id'},
                    'col_9': {'sdtype': 'unknown'},
                }
            },
            'child_2': {
                'columns': {
                    'col_10': {'sdtype': 'id'},
                    'col_11': {'sdtype': 'phone_number'},
                    'col_12': {'sdtype': 'categorical'},
                }
            },
        }
    })
    grandchildren = {'child_1', 'child_2'}

    # Run
    _simplify_grandchildren(metadata, grandchildren)

    # Assert
    expected_child_1 = {
        'col_7': {'sdtype': 'email'},
        'col_9': {'sdtype': 'unknown'},
    }
    expected_child_2 = {
        'col_11': {'sdtype': 'phone_number'},
    }
    assert metadata.tables['child_1'].columns == expected_child_1
    assert metadata.tables['child_2'].columns == expected_child_2


def test__get_num_column_to_drop():
    """Test the ``_get_num_column_to_drop`` method."""
    # Setup
    metadata = Mock()
    categorical_columns = {f'col_{i}': {'sdtype': 'categorical'} for i in range(300)}
    numerical_columns = {f'col_{i}': {'sdtype': 'numerical'} for i in range(300, 600)}
    datetime_columns = {f'col_{i}': {'sdtype': 'datetime'} for i in range(600, 900)}
    id_columns = {f'col_{i}': {'sdtype': 'id'} for i in range(900, 910)}
    email_columns = {f'col_{i}': {'sdtype': 'email'} for i in range(910, 920)}
    metadata = Metadata().load_from_dict({
        'tables': {
            'child': {
                'columns': {
                    **categorical_columns,
                    **numerical_columns,
                    **datetime_columns,
                    **id_columns,
                    **email_columns,
                }
            }
        }
    })

    child_table = 'child'
    max_col_per_relationship = 500
    num_modelable_columnn = len(metadata.tables[child_table].columns) - 20

    # Run
    num_col_to_drop, modelable_columns = _get_num_column_to_drop(
        metadata, child_table, max_col_per_relationship
    )

    # Assert
    actual_num_modelable_columnn = sum([len(value) for value in modelable_columns.values()])
    assert num_col_to_drop == 873
    assert actual_num_modelable_columnn == num_modelable_columnn


@patch('sdv.multi_table.utils._get_num_column_to_drop')
def test__get_columns_to_drop_child_drop_all_modelable_columns(mock_get_num_column_to_drop):
    """Test the ``_get_columns_to_drop_child`` when all modelable columns must be droped."""
    # Setup
    metadata = Mock()
    max_col_per_relationship = 10
    modelable_column = {
        'numerical': ['col_1', 'col_3', 'col_3'],
        'categorical': ['col_4', 'col_5'],
    }
    mock_get_num_column_to_drop.return_value = (10, modelable_column)

    # Run
    columns_to_drop = _get_columns_to_drop_child(metadata, 'child', max_col_per_relationship)

    # Assert
    mock_get_num_column_to_drop.assert_called_once_with(metadata, 'child', max_col_per_relationship)
    assert columns_to_drop == ['col_1', 'col_3', 'col_3', 'col_4', 'col_5']


@patch('sdv.multi_table.utils._get_num_column_to_drop')
def test__get_columns_to_drop_child_only_one_sdtyoe(mock_get_num_column_to_drop):
    """Test ``_get_columns_to_drop_child`` when all modelable columns are from the same sdtype."""
    # Setup
    metadata = Mock()
    max_col_per_relationship = 10
    modelable_column = {'numerical': ['col_1', 'col_2', 'col_3'], 'categorical': []}
    mock_get_num_column_to_drop.return_value = (2, modelable_column)

    # Run
    columns_to_drop = _get_columns_to_drop_child(metadata, 'child', max_col_per_relationship)

    # Assert
    mock_get_num_column_to_drop.assert_called_once_with(metadata, 'child', max_col_per_relationship)
    assert set(columns_to_drop).issubset({'col_1', 'col_2', 'col_3'})
    assert len(set(columns_to_drop)) == len(columns_to_drop) == 2


@patch('sdv.multi_table.utils._get_num_column_to_drop')
def test__get_column_to_drop_child(mock_get_num_column_to_drop):
    """Test the ``_get_columns_to_drop_child`` when all modelable columns are from the same sdtype.

    Here we check that the sdtype proportion are preserved when dropping columns.
    We drop 5 columns in total, so to preserve the proportion we should drop 3 numerical columns
    and 2 categorical columns.
    """
    # Setup
    metadata = Mock()
    max_col_per_relationship = 10
    modelable_column = {
        'numerical': ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6'],
        'categorical': ['col_7', 'col_8', 'col_9', 'col_10'],
    }
    mock_get_num_column_to_drop.return_value = (5, modelable_column)

    # Run
    columns_to_drop = _get_columns_to_drop_child(metadata, 'child', max_col_per_relationship)

    # Assert
    mock_get_num_column_to_drop.assert_called_once_with(metadata, 'child', max_col_per_relationship)
    numerical_set = {'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6'}
    categorical_set = {'col_7', 'col_8', 'col_9', 'col_10'}
    output_set = set(columns_to_drop)
    assert len(output_set) == len(columns_to_drop) == 5
    assert len(output_set.intersection(numerical_set)) == 3
    assert len(output_set.intersection(categorical_set)) == 2


@patch('sdv.multi_table.utils._get_columns_to_drop_child')
def test__simplify_child(mock_get_columns_to_drop_child):
    """Test the ``_simplify_child`` method."""
    # Setup
    metadata = Mock()
    metadata.tables = {
        'child': Mock(),
    }
    metadata.tables['child'].columns = {
        'col_1': {'sdtype': 'numerical'},
        'col_2': {'sdtype': 'categorical'},
        'col_3': {'sdtype': 'numerical'},
        'col_4': {'sdtype': 'categorical'},
        'col_5': {'sdtype': 'numerical'},
        'col_6': {'sdtype': 'categorical'},
    }
    max_col_per_relationship = 3
    mock_get_columns_to_drop_child.return_value = ['col_1', 'col_2', 'col_3']

    # Run
    _simplify_child(metadata, 'child', max_col_per_relationship)

    # Assert
    mock_get_columns_to_drop_child.assert_called_once_with(
        metadata, 'child', max_col_per_relationship
    )
    assert metadata.tables['child'].columns == {
        'col_4': {'sdtype': 'categorical'},
        'col_5': {'sdtype': 'numerical'},
        'col_6': {'sdtype': 'categorical'},
    }


@patch('sdv.multi_table.utils.HMASynthesizer')
def test__simplify_children_valid_children(mock_hma):
    """Test the ``_simplify_children`` when there is no column to drop in the children."""
    # Setup
    children = ['child_1', 'child_2']
    num_data_column = 3
    metadata = Mock()
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child_1'},
        {'parent_table_name': 'parent', 'child_table_name': 'child_2'},
    ]
    metadata.relationships = relationships
    metadata._get_foreign_keys.return_value = ['fk']
    mock_hma._get_num_extended_columns.side_effect = [250, 499]

    # Run
    _simplify_children(metadata, children, 'parent', num_data_column)

    # Assert
    mock_hma._get_num_extended_columns.assert_has_calls([
        call(metadata, 'child_1', 'parent', 3),
        call(metadata, 'child_2', 'parent', 3),
    ])


@patch('sdv.multi_table.utils.HMASynthesizer')
@patch('sdv.multi_table.utils._get_columns_to_drop_child')
def test__simplify_children(mock_get_columns_to_drop_child, mock_hma):
    """Test the ``_simplify_children`` method."""
    # Setup
    children = ['child_1', 'child_2']
    num_data_column = 3
    relatioships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child_1'},
        {'parent_table_name': 'parent', 'child_table_name': 'child_2'},
    ]
    child_1 = {
        'columns': {
            'col_1': {'sdtype': 'numerical'},
            'col_2': {'sdtype': 'categorical'},
            'col_3': {'sdtype': 'numerical'},
        }
    }
    child_2 = {
        'columns': {
            'col_5': {'sdtype': 'numerical'},
            'col_6': {'sdtype': 'categorical'},
            'col_7': {'sdtype': 'numerical'},
        }
    }
    child_1_before_simplify = deepcopy(child_1)
    child_1_before_simplify['columns']['col_4'] = {'sdtype': 'categorical'}
    child_2_before_simplify = deepcopy(child_2)
    child_2_before_simplify['columns']['col_8'] = {'sdtype': 'categorical'}
    metadata = Metadata().load_from_dict({
        'relationships': relatioships,
        'tables': {'child_1': child_1_before_simplify, 'child_2': child_2_before_simplify},
    })
    metadata_after_simplify_2 = Metadata().load_from_dict({
        'relationships': relatioships,
        'tables': {'child_1': child_1, 'child_2': child_2},
    })
    mock_hma._get_num_extended_columns.side_effect = [800, 700]
    mock_get_columns_to_drop_child.side_effect = [['col_4'], ['col_8']]

    # Run
    _simplify_children(metadata, children, 'parent', num_data_column)

    # Assert
    assert metadata.to_dict()['tables'] == metadata_after_simplify_2.to_dict()['tables']
    mock_hma._get_num_extended_columns.assert_has_calls([
        call(metadata, 'child_1', 'parent', 3),
        call(metadata, 'child_2', 'parent', 3),
    ])
    mock_get_columns_to_drop_child.assert_has_calls([
        call(metadata, 'child_1', 500),
        call(metadata, 'child_2', 500),
    ])


@patch('sdv.multi_table.utils.HMASynthesizer')
def test__get_total_estimated_columns(mock_hma):
    """Test the ``_get_total_estimated_columns`` method."""
    # Setup
    mock_hma._estimate_num_columns.return_value = {'child_1': 500, 'child_2': 700}
    metadata = Mock()

    # Run
    result = _get_total_estimated_columns(metadata)

    # Assert
    mock_hma._estimate_num_columns.assert_called_once_with(metadata)
    assert result == 1200


@patch('sdv.multi_table.utils.HMASynthesizer')
def test__simplify_metadata_no_child_simplification(mock_hma):
    """Test the ``_simplify_metadata`` method.

    Here we test the case where no child simplification is needed.
    """
    # Setup
    relationships = [
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'parent',
            'child_foreign_key': 'fk',
            'parent_primary_key': 'pk',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'child_foreign_key': 'fk',
            'parent_primary_key': 'pk',
        },
        {
            'parent_table_name': 'child',
            'child_table_name': 'grandchild',
            'child_foreign_key': 'fk',
            'parent_primary_key': 'pk',
        },
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'other_table',
            'child_foreign_key': 'fk',
            'parent_primary_key': 'pk'
        },
        {'parent_table_name': 'other_root', 'child_table_name': 'child', 'child_foreign_key': 'fk'},
    ]
    tables = {
        'grandparent': {'columns': {'col_1': {'sdtype': 'numerical'}}},
        'parent': {'columns': {'col_2': {'sdtype': 'numerical'}}},
        'child': {
            'columns': {
                'col_3': {'sdtype': 'numerical'},
                'col_4': {'sdtype': 'id'},
                'col_5': {'sdtype': 'email'},
                'col_6': {'sdtype': 'categorical'},
            }
        },
        'grandchild': {'columns': {'col_7': {'sdtype': 'numerical'}}},
        'other_table': {'columns': {'col_8': {'sdtype': 'numerical'}}},
        'other_root': {'columns': {'col_9': {'sdtype': 'numerical'}}},
    }
    metadata = Metadata().load_from_dict({
        'relationships': relationships,
        'tables': tables,
    })
    mock_hma._estimate_num_columns.return_value = {'child': 10, 'parent': 20, 'other_table': 30}

    # Run
    metadata_result = _simplify_metadata(metadata)

    # Assert
    expected_tables = {
        'grandparent': {'columns': {'col_1': {'sdtype': 'numerical'}}},
        'parent': {'columns': {'col_2': {'sdtype': 'numerical'}}},
        'child': {
            'columns': {
                'col_5': {'sdtype': 'email'},
            }
        },
        'other_table': {'columns': {'col_8': {'sdtype': 'numerical'}}},
    }
    expected_relationships = [
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'parent',
            'child_foreign_key': 'fk',
            'parent_primary_key': 'pk',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'child_foreign_key': 'fk',
            'parent_primary_key': 'pk',
        },
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'other_table',
            'child_foreign_key': 'fk',
            'parent_primary_key': 'pk',
        },
    ]
    metadata_dict = metadata_result.to_dict()
    assert metadata_dict['tables'] == expected_tables
    assert metadata_dict['relationships'] == expected_relationships


@patch('sdv.multi_table.utils.HMASynthesizer')
@patch('sdv.multi_table.utils._get_columns_to_drop_child')
def test__simplify_metadata(mock_get_columns_to_drop_child, mock_hma):
    """Test the ``_simplify_metadata`` method."""
    # Setup
    relationships = [
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'parent',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'other_table',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'col_9',
        },
        {'parent_table_name': 'other_root', 'child_table_name': 'child'},
    ]
    tables = {
        'grandparent': {
            'columns': {
                'col_1': {'sdtype': 'numerical'},
                'id_parent': {'sdtype': 'id'},
            },
            'primary_key': 'id_parent',
        },
        'parent': {
            'columns': {
                'col_2': {'sdtype': 'numerical'},
                'col_3': {'sdtype': 'categorical'},
                'col_4': {'sdtype': 'id'},
                'col_5': {'sdtype': 'datetime'},
                'parent_foreign_key': {'sdtype': 'id'},
                'id_child': {'sdtype': 'id'},
            },
            'primary_key': 'id_child',
        },
        'child': {
            'columns': {
                'col_6': {'sdtype': 'numerical'},
                'col_7': {'sdtype': 'id'},
                'col_8': {'sdtype': 'email'},
                'col_9': {'sdtype': 'categorical'},
                'child_foreign_key': {'sdtype': 'id'},
            }
        },
        'grandchild': {'columns': {'col_10': {'sdtype': 'numerical'}}},
        'other_table': {
            'columns': {
                'col_8': {'sdtype': 'numerical'},
                'col_9': {'sdtype': 'id'},
                'col_10': {'sdtype': 'categorical'},
            }
        },
        'other_root': {'columns': {'col_9': {'sdtype': 'numerical'}}},
    }
    metadata = Metadata().load_from_dict({
        'relationships': relationships,
        'tables': tables,
    })
    mock_hma._estimate_num_columns.return_value = {'child': 800, 'parent': 900, 'other_table': 10}
    mock_hma._get_num_extended_columns.side_effect = [500, 700, 10]
    mock_get_columns_to_drop_child.side_effect = [
        ['col_2', 'col_3'],
        ['col_8'],
    ]

    # Run
    metadata_result = _simplify_metadata(metadata)

    # Assert
    expected_tables = {
        'grandparent': {
            'columns': {
                'col_1': {'sdtype': 'numerical'},
                'id_parent': {'sdtype': 'id'},
            },
            'primary_key': 'id_parent',
        },
        'parent': {
            'columns': {
                'col_4': {'sdtype': 'id'},
                'col_5': {'sdtype': 'datetime'},
                'parent_foreign_key': {'sdtype': 'id'},
                'id_child': {'sdtype': 'id'},
            },
            'primary_key': 'id_child',
        },
        'child': {
            'columns': {
                'col_8': {'sdtype': 'email'},
                'child_foreign_key': {'sdtype': 'id'},
            }
        },
        'other_table': {
            'columns': {
                'col_9': {'sdtype': 'id'},
                'col_10': {'sdtype': 'categorical'},
            }
        },
    }
    expected_relationships = [
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'parent',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'parent_foreign_key',
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key',
        },
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'other_table',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'col_9',
        },
    ]
    metadata_dict = metadata_result.to_dict()

    assert metadata_dict['tables'] == expected_tables
    assert metadata_dict['relationships'] == expected_relationships


def test__simplify_data():
    """Test the ``_simplify_data`` method."""
    # Setup
    metadata = Metadata().load_from_dict({
        'tables': {
            'parent': {'columns': {'col_1': {'sdtype': 'id'}}},
            'child': {'columns': {'col_2': {'sdtype': 'id'}}},
            'grandchild': {'columns': {'col_4': {'sdtype': 'id'}}},
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'col_1',
                'child_foreign_key': 'col_2',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'grandchild',
                'parent_primary_key': 'col_1',
                'child_foreign_key': 'col_4',
            },
        ],
    })
    data = {
        'parent': pd.DataFrame({'col_1': [1, 2, 3]}),
        'child': pd.DataFrame({'col_2': [2, 2, 3], 'col_3': [7, 8, 9]}),
        'grandchild': pd.DataFrame({
            'col_4': [3, 2, 1],
            'col_5': [10, 11, 12],
            'col_7': [13, 14, 15],
        }),
        'grandchild_2': pd.DataFrame({'col_5': [10, 11, 12]}),
    }

    # Run
    data_result = _simplify_data(data, metadata)

    # Assert
    expected_results = {
        'parent': pd.DataFrame({'col_1': [1, 2, 3]}),
        'child': pd.DataFrame({
            'col_2': [2, 2, 3],
        }),
        'grandchild': pd.DataFrame({
            'col_4': [3, 2, 1],
        }),
    }
    for table_name in metadata.tables:
        pd.testing.assert_frame_equal(data_result[table_name], expected_results[table_name])


def test__print_simplified_schema_summary(capsys):
    """Test the ``_print_simplified_schema_summary`` method."""
    # Setup
    data_before_1 = pd.DataFrame({
        'col_1': [1, 2, 3],
        'col_2': [2, 2, 3],
        'col_3': [7, 8, 9],
        'col_4': [3, 2, 1],
    })
    data_before_2 = pd.DataFrame({
        'col_5': [10, 11, 12],
        'col_6': [13, 14, 15],
    })
    data_before_3 = pd.DataFrame({'col_7': [10, 11, 12]})
    data_before = {
        'Table 1': data_before_1,
        'Table 2': data_before_2,
        'Table 3': data_before_3,
    }

    data_after_1 = pd.DataFrame({
        'col_1': [1, 2, 3],
        'col_2': [2, 2, 3],
    })
    data_after_2 = pd.DataFrame({
        'col_5': [2, 2, 3],
    })
    data_after = {
        'Table 1': data_after_1,
        'Table 2': data_after_2,
    }

    # Run
    _print_simplified_schema_summary(data_before, data_after)
    captured = capsys.readouterr()

    # Assert
    expected_output = re.compile(
        r'Success! The schema has been simplified\.\s*'
        r'Table Name\s*#\s*Columns \(Before\)\s*#\s*Columns \(After\)\s*'
        r'Table 1\s*4\s*2\s*'
        r'Table 2\s*2\s*1\s*'
        r'Table 3\s*1\s*0'
    )
    assert expected_output.match(captured.out.strip())


@patch('sdv.multi_table.utils._get_disconnected_roots_from_table')
@patch('sdv.multi_table.utils._drop_rows')
def test__subsample_disconnected_roots(mock_drop_rows, mock_get_disconnected_roots_from_table):
    """Test the ``_subsample_disconnected_roots`` method."""
    # Setup
    data = {
        'disconnected_root': pd.DataFrame({
            'col_1': [1, 2, 3, 4, 5],
            'col_2': [6, 7, 8, 9, 10],
        }),
        'grandparent': pd.DataFrame({
            'col_3': [1, 2, 3, 4, 5],
            'col_4': [6, 7, 8, 9, 10],
        }),
        'other_root': pd.DataFrame({
            'col_5': [1, 2, 3, 4, 5],
            'col_6': [6, 7, 8, 9, 10],
        }),
        'child': pd.DataFrame({
            'col_7': [1, 2, 3, 4, 5],
            'col_8': [6, 7, 8, 9, 10],
        }),
        'other_table': pd.DataFrame({
            'col_9': [1, 2, 3, 4, 5],
            'col_10': [6, 7, 8, 9, 10],
        }),
        'parent': pd.DataFrame({
            'col_11': [1, 2, 3, 4, 5],
            'col_12': [6, 7, 8, 9, 10],
        }),
    }
    metadata = Metadata().load_from_dict({
        'tables': {
            'disconnected_root': {
                'columns': {
                    'col_1': {'sdtype': 'numerical'},
                    'col_2': {'sdtype': 'numerical'},
                },
            },
            'grandparent': {
                'columns': {
                    'col_3': {'sdtype': 'numerical'},
                    'col_4': {'sdtype': 'numerical'},
                },
            },
            'other_root': {
                'columns': {
                    'col_5': {'sdtype': 'numerical'},
                    'col_6': {'sdtype': 'numerical'},
                },
            },
            'child': {
                'columns': {
                    'col_7': {'sdtype': 'numerical'},
                    'col_8': {'sdtype': 'numerical'},
                },
            },
            'other_table': {
                'columns': {
                    'col_9': {'sdtype': 'numerical'},
                    'col_10': {'sdtype': 'numerical'},
                },
            },
            'parent': {
                'columns': {
                    'col_11': {'sdtype': 'numerical'},
                    'col_12': {'sdtype': 'numerical'},
                },
            },
        },
        'relationships': [
            {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
            {'parent_table_name': 'parent', 'child_table_name': 'child'},
            {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
            {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
            {'parent_table_name': 'other_root', 'child_table_name': 'child'},
            {'parent_table_name': 'disconnected_root', 'child_table_name': 'disconnect_child'},
        ],
    })
    mock_get_disconnected_roots_from_table.return_value = {'grandparent', 'other_root'}
    ratio_to_keep = 0.6
    expected_result = deepcopy(data)

    # Run
    _subsample_disconnected_roots(
        data, metadata, 'disconnected_root', ratio_to_keep, drop_missing_values=False
    )

    # Assert
    mock_get_disconnected_roots_from_table.assert_called_once_with(
        metadata.relationships, 'disconnected_root'
    )
    mock_drop_rows.assert_called_once_with(data, metadata, False)
    for table_name in metadata.tables:
        if table_name not in {'grandparent', 'other_root'}:
            pd.testing.assert_frame_equal(data[table_name], expected_result[table_name])
        else:
            assert len(data[table_name]) == 3


@patch('sdv.multi_table.utils._drop_rows')
def test__subsample_table_and_descendants(mock_drop_rows):
    """Test the ``_subsample_table_and_descendants`` method."""
    # Setup
    data = {
        'grandparent': pd.DataFrame({
            'col_1': [1, 2, 3, 4, 5],
            'col_2': [6, 7, 8, 9, 10],
        }),
        'parent': pd.DataFrame({
            'col_3': [1, 2, 3, 4, 5],
            'col_4': [6, 7, 8, 9, 10],
        }),
        'child': pd.DataFrame({
            'col_5': [1, 2, 3, 4, 5],
            'col_6': [6, 7, 8, 9, 10],
        }),
        'grandchild': pd.DataFrame({
            'col_7': [1, 2, 3, 4, 5],
            'col_8': [6, 7, 8, 9, 10],
        }),
    }
    metadata = Mock()
    metadata.relationships = Mock()

    # Run
    _subsample_table_and_descendants(data, metadata, 'parent', 3, drop_missing_values=False)

    # Assert
    mock_drop_rows.assert_called_once_with(data, metadata, False)
    assert len(data['parent']) == 3


def test__get_primary_keys_referenced():
    """Test the ``_get_primary_keys_referenced`` method."""
    data = {
        'grandparent': pd.DataFrame({
            'pk_gp': [1, 2, 3, 4, 5],
            'col_1': [6, 7, 8, 9, 10],
        }),
        'parent': pd.DataFrame({
            'fk_gp': [1, 2, 2, 3, 1],
            'pk_p': [11, 12, 13, 14, 15],
            'col_2': [16, 17, 18, 19, 20],
        }),
        'child': pd.DataFrame({
            'fk_gp': [5, 2, 2, 3, 1],
            'fk_p_1': [11, 11, 11, 11, 11],
            'fk_p_2': [12, 12, 12, 12, 12],
            'pk_c': [21, 22, 23, 24, 25],
            'col_3': [26, 27, 28, 29, 30],
        }),
        'grandchild': pd.DataFrame({
            'fk_p_3': [13, 14, 13, 13, 13],
            'fk_p_4': [14, 13, 14, 14, 14],
            'fk_c': [21, 22, 23, 24, 25],
            'col_4': [36, 37, 38, 39, 40],
        }),
    }

    metadata = Metadata().load_from_dict({
        'tables': {
            'grandparent': {
                'columns': {
                    'pk_gp': {'type': 'id'},
                    'col_1': {'type': 'numerical'},
                },
                'primary_key': 'pk_gp',
            },
            'parent': {
                'columns': {
                    'fk_gp': {'type': 'id'},
                    'pk_p': {'type': 'id'},
                    'col_2': {'type': 'numerical'},
                },
                'primary_key': 'pk_p',
            },
            'child': {
                'columns': {
                    'fk_gp': {'type': 'id'},
                    'fk_p_1': {'type': 'id'},
                    'fk_p_2': {'type': 'id'},
                    'pk_c': {'type': 'id'},
                    'col_3': {'type': 'numerical'},
                },
                'primary_key': 'pk_c',
            },
            'grandchild': {
                'columns': {
                    'fk_p_3': {'type': 'id'},
                    'fk_p_4': {'type': 'id'},
                    'fk_c': {'type': 'id'},
                    'col_4': {'type': 'numerical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'parent',
                'parent_primary_key': 'pk_gp',
                'child_foreign_key': 'fk_gp',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_1',
            },
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_gp',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_2',
            },
            {
                'parent_table_name': 'child',
                'child_table_name': 'grandchild',
                'parent_primary_key': 'pk_c',
                'child_foreign_key': 'fk_c',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'grandchild',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_3',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'grandchild',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_4',
            },
        ],
    })

    # Run
    result = _get_primary_keys_referenced(data, metadata)

    # Assert
    expected_result = {
        'grandparent': {1, 2, 3, 5},
        'parent': {11, 12, 13, 14},
        'child': {21, 22, 23, 24, 25},
    }
    assert result == expected_result


def test__subsample_parent_all_reeferenced_before():
    """Test the ``_subsample_parent`` when all primary key were referenced before.

    Here the primary keys ``4`` and ``5`` are no longer referenced and should be dropped.
    """
    # Setup
    data = {
        'parent': pd.DataFrame({
            'pk_p': [1, 2, 3, 4, 5],
            'col_2': [16, 17, 18, 19, 20],
        }),
        'child': pd.DataFrame({
            'fk_p_1': [1, 2, 2, 2, 3],
        }),
    }

    pk_referenced_before = defaultdict(set)
    pk_referenced_before['parent'] = {1, 2, 3, 4, 5}
    unreferenced_pk = {4, 5}

    # Run
    data['parent'] = _subsample_parent(
        data['parent'], 'pk_p', pk_referenced_before['parent'], unreferenced_pk
    )

    # Assert
    expected_result = {
        'parent': pd.DataFrame({
            'pk_p': [1, 2, 3],
            'col_2': [16, 17, 18],
        }),
        'child': pd.DataFrame({
            'fk_p_1': [1, 2, 2, 2, 3],
        }),
    }
    pd.testing.assert_frame_equal(data['parent'], expected_result['parent'])
    pd.testing.assert_frame_equal(data['child'], expected_result['child'])


def test__subsample_parent_not_all_referenced_before():
    """Test the ``_subsample_parent`` when not all primary key were referenced before.

    In this example:
    - The primary key ``5`` is no longer referenced and should be dropped.
    - One unreferenced primary key must be dropped to keep the same ratio of
    referenced/unreferenced primary keys.
    """
    # Setup
    data = {
        'parent': pd.DataFrame({
            'pk_p': [1, 2, 3, 4, 5, 6, 7, 8],
            'col_2': [16, 17, 18, 19, 20, 21, 22, 23],
        }),
        'child': pd.DataFrame({
            'fk_p_1': [1, 2, 2, 2, 3],
        }),
    }

    pk_referenced_before = defaultdict(set)
    pk_referenced_before['parent'] = {1, 2, 3, 5}
    unreferenced_pk = {5}

    # Run
    data['parent'] = _subsample_parent(
        data['parent'], 'pk_p', pk_referenced_before['parent'], unreferenced_pk
    )

    # Assert
    assert len(data['parent']) == 6
    assert set(data['parent']['pk_p']).issubset({1, 2, 3, 4, 6, 7, 8})


def test__subsample_ancestors():
    """Test the ``_subsample_ancestors`` method."""
    # Setup
    data = {
        'grandparent': pd.DataFrame({
            'pk_gp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'col_1': [
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
            ],
        }),
        'parent': pd.DataFrame({
            'fk_gp': [1, 2, 3, 4, 9, 6],
            'pk_p': [11, 12, 13, 14, 15, 16],
            'col_2': ['k', 'l', 'm', 'n', 'o', 'p'],
        }),
        'child': pd.DataFrame({
            'fk_gp': [4, 5, 6, 7, 8],
            'fk_p_1': [11, 11, 11, 14, 11],
            'fk_p_2': [12, 12, 12, 12, 15],
            'pk_c': [21, 22, 23, 24, 25],
            'col_3': ['q', 'r', 's', 't', 'u'],
        }),
        'grandchild': pd.DataFrame({
            'fk_p_3': [11, 12, 13, 11, 13],
            'fk_c': [21, 22, 23, 21, 22],
            'col_4': [36, 37, 38, 39, 40],
        }),
    }

    primary_key_referenced = {
        'grandparent': {1, 2, 3, 4, 5, 6, 7, 8, 9},
        'parent': {11, 12, 13, 14, 15},
        'child': {21, 22, 23, 24, 25},
    }

    metadata = Metadata().load_from_dict({
        'tables': {
            'grandparent': {
                'columns': {
                    'pk_gp': {'type': 'id'},
                    'col_1': {'type': 'numerical'},
                },
                'primary_key': 'pk_gp',
            },
            'parent': {
                'columns': {
                    'fk_gp': {'type': 'id'},
                    'pk_p': {'type': 'id'},
                    'col_2': {'type': 'numerical'},
                },
                'primary_key': 'pk_p',
            },
            'child': {
                'columns': {
                    'fk_gp': {'type': 'id'},
                    'fk_p_1': {'type': 'id'},
                    'fk_p_2': {'type': 'id'},
                    'pk_c': {'type': 'id'},
                    'col_3': {'type': 'numerical'},
                },
                'primary_key': 'pk_c',
            },
            'grandchild': {
                'columns': {
                    'fk_p_3': {'type': 'id'},
                    'fk_p_4': {'type': 'id'},
                    'fk_c': {'type': 'id'},
                    'col_4': {'type': 'numerical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'parent',
                'parent_primary_key': 'pk_gp',
                'child_foreign_key': 'fk_gp',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_1',
            },
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_gp',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_2',
            },
            {
                'parent_table_name': 'child',
                'child_table_name': 'grandchild',
                'parent_primary_key': 'pk_c',
                'child_foreign_key': 'fk_c',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'grandchild',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_3',
            },
        ],
    })

    # Run
    _subsample_ancestors(data, metadata, 'grandchild', primary_key_referenced)

    # Assert
    expected_result = {
        'parent': pd.DataFrame(
            {
                'fk_gp': [1, 2, 3, 6],
                'pk_p': [11, 12, 13, 16],
                'col_2': ['k', 'l', 'm', 'p'],
            },
            index=[0, 1, 2, 5],
        ),
        'child': pd.DataFrame(
            {
                'fk_gp': [4, 5, 6],
                'fk_p_1': [11, 11, 11],
                'fk_p_2': [12, 12, 12],
                'pk_c': [21, 22, 23],
                'col_3': ['q', 'r', 's'],
            },
            index=[0, 1, 2],
        ),
        'grandchild': pd.DataFrame(
            {
                'fk_p_3': [11, 12, 13, 11, 13],
                'fk_c': [21, 22, 23, 21, 22],
                'col_4': [36, 37, 38, 39, 40],
            },
            index=[0, 1, 2, 3, 4],
        ),
    }
    assert len(data['grandparent']) == 14
    assert set(data['grandparent']['pk_gp']).issubset({
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    })
    for table_name in ['parent', 'child', 'grandchild']:
        pd.testing.assert_frame_equal(data[table_name], expected_result[table_name])


def test__subsample_ancestors_schema_diamond_shape():
    """Test the ``_subsample_ancestors`` method with a diamond shape schema."""
    # Setup
    data = {
        'grandparent': pd.DataFrame({
            'pk_gp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'col_1': [
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
            ],
        }),
        'parent_1': pd.DataFrame({
            'fk_gp': [1, 2, 3, 4, 5, 6],
            'pk_p': [21, 22, 23, 24, 25, 26],
            'col_2': ['k', 'l', 'm', 'n', 'o', 'p'],
        }),
        'parent_2': pd.DataFrame({
            'fk_gp': [7, 8, 9, 10, 11],
            'pk_p': [31, 32, 33, 34, 35],
            'col_3': ['k', 'l', 'm', 'n', 'o'],
        }),
        'child': pd.DataFrame({
            'fk_p_1': [21, 22, 23, 23, 23],
            'fk_p_2': [31, 32, 33, 34, 34],
            'col_4': ['q', 'r', 's', 't', 'u'],
        }),
    }

    primary_key_referenced = {
        'grandparent': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        'parent_1': {21, 22, 23, 24, 25},
        'parent_2': {31, 32, 33, 34, 35},
    }

    metadata = Metadata().load_from_dict({
        'tables': {
            'grandparent': {
                'columns': {
                    'pk_gp': {'type': 'id'},
                    'col_1': {'type': 'numerical'},
                },
                'primary_key': 'pk_gp',
            },
            'parent_1': {
                'columns': {
                    'fk_gp': {'type': 'id'},
                    'pk_p': {'type': 'id'},
                    'col_2': {'type': 'numerical'},
                },
                'primary_key': 'pk_p',
            },
            'parent_2': {
                'columns': {
                    'fk_gp': {'type': 'id'},
                    'pk_p': {'type': 'id'},
                    'col_3': {'type': 'numerical'},
                },
                'primary_key': 'pk_p',
            },
            'child': {
                'columns': {
                    'fk_p_1': {'type': 'id'},
                    'fk_p_2': {'type': 'id'},
                    'col_4': {'type': 'numerical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'parent_1',
                'parent_primary_key': 'pk_gp',
                'child_foreign_key': 'fk_gp',
            },
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'parent_2',
                'parent_primary_key': 'pk_gp',
                'child_foreign_key': 'fk_gp',
            },
            {
                'parent_table_name': 'parent_1',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_1',
            },
            {
                'parent_table_name': 'parent_2',
                'child_table_name': 'child',
                'parent_primary_key': 'pk_p',
                'child_foreign_key': 'fk_p_2',
            },
        ],
    })

    # Run
    _subsample_ancestors(data, metadata, 'child', primary_key_referenced)

    # Assert
    expected_result = {
        'parent_1': pd.DataFrame(
            {
                'fk_gp': [1, 2, 3, 6],
                'pk_p': [21, 22, 23, 26],
                'col_2': ['k', 'l', 'm', 'p'],
            },
            index=[0, 1, 2, 5],
        ),
        'parent_2': pd.DataFrame(
            {
                'fk_gp': [7, 8, 9, 10],
                'pk_p': [31, 32, 33, 34],
                'col_3': ['k', 'l', 'm', 'n'],
            },
            index=[0, 1, 2, 3],
        ),
        'child': pd.DataFrame(
            {
                'fk_p_1': [21, 22, 23, 23, 23],
                'fk_p_2': [31, 32, 33, 34, 34],
                'col_4': ['q', 'r', 's', 't', 'u'],
            },
            index=[0, 1, 2, 3, 4],
        ),
    }
    assert len(data['grandparent']) == 14
    assert set(data['grandparent']['pk_gp']).issubset({
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    })
    for table_name in ['parent_1', 'parent_2', 'child']:
        pd.testing.assert_frame_equal(data[table_name], expected_result[table_name])


@patch('sdv.multi_table.utils._subsample_disconnected_roots')
@patch('sdv.multi_table.utils._subsample_table_and_descendants')
@patch('sdv.multi_table.utils._subsample_ancestors')
@patch('sdv.multi_table.utils._get_primary_keys_referenced')
@patch('sdv.multi_table.utils._drop_rows')
def test__subsample_data(
    mock_drop_rows,
    mock_get_primary_keys_referenced,
    mock_subsample_ancestors,
    mock_subsample_table_and_descendants,
    mock_subsample_disconnected_roots,
):
    """Test the ``_subsample_data`` method."""
    # Setup
    data = {
        'main_table': [1] * 10,
    }
    metadata = Mock()
    num_rows = 5
    main_table = 'main_table'
    primary_key_reference = {'main_table': {1, 2, 4}}
    mock_get_primary_keys_referenced.return_value = primary_key_reference

    # Run
    result = _subsample_data(data, metadata, main_table, num_rows)

    # Assert
    mock_drop_rows.assert_called_once_with(data, metadata, False)
    mock_get_primary_keys_referenced.assert_called_once_with(data, metadata)
    mock_subsample_disconnected_roots.assert_called_once_with(
        data, metadata, main_table, 0.5, False
    )
    mock_subsample_table_and_descendants.assert_called_once_with(
        data, metadata, main_table, num_rows, False
    )
    mock_subsample_ancestors.assert_called_once_with(
        data, metadata, main_table, primary_key_reference
    )
    assert result == data


def test__subsample_data_with_null_foreing_keys():
    """Test the ``_subsample_data`` method when there are null foreign keys."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'A': {'sdtype': 'categorical'},
                    'B': {'sdtype': 'numerical'},
                },
                'primary_key': 'id',
            },
            'child': {'columns': {'parent_id': {'sdtype': 'id'}, 'C': {'sdtype': 'categorical'}}},
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'id',
                'child_foreign_key': 'parent_id',
            }
        ],
    })

    parent = pd.DataFrame(
        data={
            'id': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
            'B': [0.434, 0.312, 0.212, 0.339, 0.491],
        }
    )

    child = pd.DataFrame(
        data={'parent_id': [0, 1, 2, 2, 5], 'C': ['Yes', 'No', 'Maybe', 'No', 'No']}
    )

    data = {'parent': parent, 'child': child}
    data['child'].loc[[2, 3, 4], 'parent_id'] = np.nan

    # Run
    result_with_nan = _subsample_data(data, metadata, 'child', 4, drop_missing_values=False)
    result_without_nan = _subsample_data(data, metadata, 'child', 2, drop_missing_values=True)

    # Assert
    assert len(result_with_nan['child']) == 4
    assert result_with_nan['child']['parent_id'].isna().sum() > 0
    assert len(result_without_nan['child']) == 2
    assert set(result_without_nan['child'].index) == {0, 1}


@patch('sdv.multi_table.utils._subsample_disconnected_roots')
@patch('sdv.multi_table.utils._get_primary_keys_referenced')
def test__subsample_data_empty_dataset(
    mock_get_primary_keys_referenced,
    mock_subsample_disconnected_roots,
):
    """Test the ``subsample_data`` method when a dataset is empty."""
    # Setup
    data = {
        'main_table': [1] * 10,
    }
    metadata = Mock()
    num_rows = 5
    main_table = 'main_table'
    mock_subsample_disconnected_roots.side_effect = InvalidDataError('All references in table')

    # Run and Assert
    expected_message = re.escape(
        'Subsampling main_table with 5 rows leads to some empty tables. '
        'Please try again with a bigger number of rows.'
    )
    with pytest.raises(SamplingError, match=expected_message):
        _subsample_data(data, metadata, main_table, num_rows)


def test__print_subsample_summary(capsys):
    """Test the ``_print_subsample_summary`` method."""
    # Setup
    data_before = {
        'grandparent': pd.DataFrame({
            'pk_gp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'col_1': [
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
            ],
        }),
        'parent_1': pd.DataFrame({
            'fk_gp': [1, 2, 3, 4, 5, 6],
            'pk_p': [21, 22, 23, 24, 25, 26],
            'col_2': ['k', 'l', 'm', 'n', 'o', 'p'],
        }),
        'parent_2': pd.DataFrame({
            'fk_gp': [7, 8, 9, 10, 11],
            'pk_p': [31, 32, 33, 34, 35],
            'col_3': ['k', 'l', 'm', 'n', 'o'],
        }),
        'child': pd.DataFrame({
            'fk_p_1': [21, 22, 23, 23, 23],
            'fk_p_2': [31, 32, 33, 34, 34],
            'col_4': ['q', 'r', 's', 't', 'u'],
        }),
    }

    data_after = {
        'grandparent': pd.DataFrame(
            {
                'pk_gp': [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 20],
                'col_1': ['a', 'b', 'c', 'f', 'g', 'h', 'i', 'j', 'n', 'o', 'p', 'q', 'r', 't'],
            },
            index=[0, 1, 2, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 19],
        ),
        'parent_1': pd.DataFrame(
            {
                'fk_gp': [1, 2, 3, 6],
                'pk_p': [21, 22, 23, 26],
                'col_2': ['k', 'l', 'm', 'p'],
            },
            index=[0, 1, 2, 5],
        ),
        'parent_2': pd.DataFrame(
            {
                'fk_gp': [7, 8, 9, 10],
                'pk_p': [31, 32, 33, 34],
                'col_3': ['k', 'l', 'm', 'n'],
            },
            index=[0, 1, 2, 3],
        ),
        'child': pd.DataFrame(
            {
                'fk_p_1': [21, 22, 23, 23, 23],
                'fk_p_2': [31, 32, 33, 34, 34],
                'col_4': ['q', 'r', 's', 't', 'u'],
            },
            index=[0, 1, 2, 3, 4],
        ),
    }

    # Run
    _print_subsample_summary(data_before, data_after)
    captured = capsys.readouterr()

    # Assert
    expected_output = re.compile(
        r'Success! Your subset has 25% less rows than the original\.\s*'
        r'Table Name\s*#\s*Rows \(Before\)\s*#\s*Rows \(After\)\s*'
        r'child\s*5\s*5\s*'
        r'grandparent\s*20\s*14\s*'
        r'parent_1\s*6\s*4\s*'
        r'parent_2\s*5\s*4'
    )
    assert expected_output.match(captured.out.strip())
