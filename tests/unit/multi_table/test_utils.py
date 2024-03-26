from collections import defaultdict
from unittest.mock import Mock, call, patch

import pandas as pd

from sdv.metadata import MultiTableMetadata
from sdv.multi_table.utils import (
    _get_all_descendant_per_root_at_order_n, _get_columns_to_drop_child, _get_n_order_descendants,
    _get_num_column_to_drop, _get_relationship_for_child, _get_relationship_for_parent,
    _get_root_tables, _get_rows_to_drop, _get_total_estimated_columns, _simplify_child,
    _simplify_children, _simplify_data, _simplify_grandchilds, _simplify_metadata,
    _simplify_non_descendants_tables, _simplify_relationships)


def test__get_root_tables():
    """Test the ``_get_root_tables`` method."""
    # Setup
    relationships = [
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'parent', 'child_table_name': 'grandchild'}
    ]

    # Run
    result = _get_root_tables(relationships)

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
    expected_gp_order_2 = {
        'order_1': ['parent', 'other_table'],
        'order_2': ['child']
    }
    expected_gp_order_3 = {
        'order_1': ['parent', 'other_table'],
        'order_2': ['child'],
        'order_3': ['grandchild']
    }
    expected_other_order_2 = {
        'order_1': [],
        'order_2': []
    }
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
        'grandparent': {'other_table', 'child', 'parent', 'grandchild'},
        'other_root': {'child', 'grandchild'}
    }
    assert result == expected_result


def test__simplify_relationships():
    """Test the ``_simplify_relationships`` method."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
        'relationships': [
            {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
            {'parent_table_name': 'parent', 'child_table_name': 'child'},
            {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
            {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
            {'parent_table_name': 'other_root', 'child_table_name': 'child'},
        ]
    })
    descendant_to_keep = ['parent', 'child']

    # Run
    metadata_result, childs, grandchilds = _simplify_relationships(
        metadata, 'grandparent', descendant_to_keep
    )

    # Assert
    expected_result = [
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
    ]
    assert metadata_result.relationships == expected_result
    assert childs == ['parent']
    assert grandchilds == ['child']


def test__simplify_non_descendants_tables():
    """Test the ``_simplify_non_descendants_tables`` method."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'grandparent': {'columns': {'col_1': {'sdtype': 'numerical'}}},
            'parent': {'columns': {'col_2': {'sdtype': 'numerical'}}},
            'child': {'columns': {'col_3': {'sdtype': 'numerical'}}},
            'grandchild': {'columns': {'col_4': {'sdtype': 'numerical'}}},
            'other_table': {'columns': {'col_5': {'sdtype': 'numerical'}}},
            'other_root': {'columns': {'col_6': {'sdtype': 'numerical'}}}
        }
    })
    root_table = 'grandparent'
    descendant_to_keep = ['parent', 'child']

    # Run
    metadata_result = _simplify_non_descendants_tables(metadata, root_table, descendant_to_keep)

    # Assert
    assert metadata_result.tables.keys() == {'grandparent', 'parent', 'child'}


def test__simplify_grandchilds():
    """Test the ``_simplify_grandchilds`` method."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
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
                    'col_12': {'sdtype': 'categorical'}
                }
            },
        }
    })
    grandchilds = {'child_1', 'child_2'}

    # Run
    metadata_result = _simplify_grandchilds(metadata, grandchilds)

    # Assert
    expected_child_1 = {
        'col_7': {'sdtype': 'email'},
        'col_8': {'sdtype': 'id'},
        'col_9': {'sdtype': 'unknown'},
    }
    expected_child_2 = {
        'col_10': {'sdtype': 'id'},
        'col_11': {'sdtype': 'phone_number'},
    }
    assert metadata_result.tables['child_1'].columns == expected_child_1
    assert metadata_result.tables['child_2'].columns == expected_child_2


def test__get_num_column_to_drop():
    """Test the ``_get_num_column_to_drop`` method."""
    # Setup
    metadata = Mock()
    categorical_columns = {
        f'col_{i}': {'sdtype': 'categorical'} for i in range(300)
    }
    numerical_columns = {
        f'col_{i}': {'sdtype': 'numerical'} for i in range(300, 600)
    }
    datetime_columns = {
        f'col_{i}': {'sdtype': 'datetime'} for i in range(600, 900)
    }
    id_columns = {
        f'col_{i}': {'sdtype': 'id'} for i in range(900, 910)
    }
    email_columns = {
        f'col_{i}': {'sdtype': 'email'} for i in range(910, 920)
    }
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'child': {
                'columns': {
                    **categorical_columns, **numerical_columns,
                    **datetime_columns, **id_columns, **email_columns
                }
            }
        }
    })

    child_table = 'child'
    max_col_per_relationship = 500
    num_modelable_column = (len(metadata.tables[child_table].columns) - 20)

    # Run
    num_col_to_drop, modelable_columns = _get_num_column_to_drop(
        metadata, child_table, max_col_per_relationship
    )

    # Assert
    actual_num_modelable_column = sum([len(value) for value in modelable_columns.values()])
    assert num_col_to_drop == 873
    assert actual_num_modelable_column == num_modelable_column


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
    columns_to_drop = _get_columns_to_drop_child(
        metadata, 'child', max_col_per_relationship
    )

    # Assert
    mock_get_num_column_to_drop.assert_called_once_with(
        metadata, 'child', max_col_per_relationship
    )
    assert columns_to_drop == ['col_1', 'col_3', 'col_3', 'col_4', 'col_5']


@patch('sdv.multi_table.utils._get_num_column_to_drop')
def test__get_columns_to_drop_child_only_one_sdtyoe(mock_get_num_column_to_drop):
    """Test ``_get_columns_to_drop_child`` when all modelable columns are from the same sdtype."""
    # Setup
    metadata = Mock()
    max_col_per_relationship = 10
    modelable_column = {
        'numerical': ['col_1', 'col_2', 'col_3'],
        'categorical': []
    }
    mock_get_num_column_to_drop.return_value = (2, modelable_column)

    # Run
    columns_to_drop = _get_columns_to_drop_child(
        metadata, 'child', max_col_per_relationship
    )

    # Assert
    mock_get_num_column_to_drop.assert_called_once_with(
        metadata, 'child', max_col_per_relationship
    )
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
        'categorical': ['col_7', 'col_8', 'col_9', 'col_10']
    }
    mock_get_num_column_to_drop.return_value = (5, modelable_column)

    # Run
    columns_to_drop = _get_columns_to_drop_child(
        metadata, 'child', max_col_per_relationship
    )

    # Assert
    mock_get_num_column_to_drop.assert_called_once_with(
        metadata, 'child', max_col_per_relationship
    )
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
    metadata_result = _simplify_child(metadata, 'child', max_col_per_relationship)

    # Assert
    mock_get_columns_to_drop_child.assert_called_once_with(
        metadata, 'child', max_col_per_relationship
    )
    assert metadata_result.tables['child'].columns == {
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
    metadata_result = _simplify_children(metadata, children, 'parent', num_data_column)

    # Assert
    mock_hma._get_num_extended_columns.assert_has_calls([
        call(metadata, 'child_1', 'parent', 3),
        call(metadata, 'child_2', 'parent', 3)
    ])
    assert metadata_result == metadata


@patch('sdv.multi_table.utils.HMASynthesizer')
@patch('sdv.multi_table.utils._simplify_child')
def test__simplify_children(mock_simplify_child, mock_hma):
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
    child_1_before_simplify = child_1.copy()
    child_1_before_simplify['columns']['col_4'] = {'sdtype': 'categorical'}
    child_2_before_simplify = child_2.copy()
    child_2_before_simplify['columns']['col_8'] = {'sdtype': 'categorical'}
    metadata = MultiTableMetadata().load_from_dict({
        'relationships': relatioships,
        'tables': {
            'child_1': child_1_before_simplify,
            'child_2': child_2_before_simplify
        }
    })

    metadata_after_simplify_1 = MultiTableMetadata().load_from_dict({
        'relationships': relatioships,
        'tables': {
            'child_1': child_1,
            'child_2': child_2_before_simplify
        }
    })
    metadata_after_simplify_2 = MultiTableMetadata().load_from_dict({
        'relationships': relatioships,
        'tables': {
            'child_1': child_1,
            'child_2': child_2
        }
    })
    mock_hma._get_num_extended_columns.side_effect = [800, 700]
    mock_simplify_child.side_effect = [metadata_after_simplify_1, metadata_after_simplify_2]

    # Run
    metadata_result = _simplify_children(metadata, children, 'parent', num_data_column)

    # Assert
    mock_hma._get_num_extended_columns.assert_has_calls([
        call(metadata, 'child_1', 'parent', 3),
        call(metadata_after_simplify_1, 'child_2', 'parent', 3)
    ])
    mock_simplify_child.assert_has_calls([
        call(metadata, 'child_1', 500),
        call(metadata_after_simplify_1, 'child_2', 500)
    ])
    assert metadata_result.tables == metadata_after_simplify_2.tables


@patch('sdv.multi_table.utils.HMASynthesizer')
def test__get_total_estimated_columns(mock_hma):
    """Test the ``_get_total_estimated_columns`` method."""
    # Setup
    mock_hma._estimate_num_columns.return_value = {
        'child_1': 500,
        'child_2': 700
    }
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
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
        {'parent_table_name': 'other_root', 'child_table_name': 'child'},
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
        'other_root': {'columns': {'col_9': {'sdtype': 'numerical'}}}
    }
    metadata = MultiTableMetadata().load_from_dict({
        'relationships': relationships,
        'tables': tables
    })
    mock_hma._estimate_num_columns.return_value = {
        'child': 10,
        'parent': 20,
        'other_table': 30
    }

    # Run
    metadata_result = _simplify_metadata(metadata)

    # Assert
    expected_tables = {
        'grandparent': {'columns': {'col_1': {'sdtype': 'numerical'}}},
        'parent': {'columns': {'col_2': {'sdtype': 'numerical'}}},
        'child': {
            'columns': {
                'col_4': {'sdtype': 'id'},
                'col_5': {'sdtype': 'email'},
            }
        },
        'other_table': {'columns': {'col_8': {'sdtype': 'numerical'}}},
    }
    expected_relationships = [
        {'parent_table_name': 'grandparent', 'child_table_name': 'parent'},
        {'parent_table_name': 'parent', 'child_table_name': 'child'},
        {'parent_table_name': 'grandparent', 'child_table_name': 'other_table'},
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
            'child_foreign_key': 'parent_foreign_key'
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key'
        },
        {'parent_table_name': 'child', 'child_table_name': 'grandchild'},
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'other_table',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'col_9'
        },
        {'parent_table_name': 'other_root', 'child_table_name': 'child'},
    ]
    tables = {
        'grandparent': {
            'columns': {
                'col_1': {'sdtype': 'numerical'},
                'id_parent': {'sdtype': 'id'},
            },
            'primary_key': 'id_parent'
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
            'primary_key': 'id_child'
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
        'other_root': {'columns': {'col_9': {'sdtype': 'numerical'}}}
    }
    metadata = MultiTableMetadata().load_from_dict({
        'relationships': relationships,
        'tables': tables
    })
    mock_hma._estimate_num_columns.return_value = {
        'child': 800, 'parent': 900, 'other_table': 10
    }
    mock_hma._get_num_extended_columns.side_effect = [500, 700, 10]
    mock_get_columns_to_drop_child.side_effect = [
        [],
        ['col_2', 'col_3'],
        ['col_6', 'col_9'],
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
            'primary_key': 'id_parent'
        },
        'parent': {
            'columns': {
                'col_4': {'sdtype': 'id'},
                'col_5': {'sdtype': 'datetime'},
                'parent_foreign_key': {'sdtype': 'id'},
                'id_child': {'sdtype': 'id'},
            },
            'primary_key': 'id_child'
        },
        'child': {
            'columns': {
                'col_8': {'sdtype': 'email'},
                'col_7': {'sdtype': 'id'},
                'child_foreign_key': {'sdtype': 'id'},
            }
        },
        'other_table': {
            'columns': {
                'col_8': {'sdtype': 'numerical'},
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
            'child_foreign_key': 'parent_foreign_key'
        },
        {
            'parent_table_name': 'parent',
            'child_table_name': 'child',
            'parent_primary_key': 'id_child',
            'child_foreign_key': 'child_foreign_key'
        },
        {
            'parent_table_name': 'grandparent',
            'child_table_name': 'other_table',
            'parent_primary_key': 'id_parent',
            'child_foreign_key': 'col_9'
        },
    ]
    metadata_dict = metadata_result.to_dict()

    assert metadata_dict['tables'] == expected_tables
    assert metadata_dict['relationships'] == expected_relationships


def test__simplify_data():
    """Test the ``_simplify_data`` method."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
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
                'child_foreign_key': 'col_2'
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'grandchild',
                'parent_primary_key': 'col_1',
                'child_foreign_key': 'col_4'
            },
        ]
    })
    data = {
        'parent': pd.DataFrame({
            'col_1': [1, 2, 3]
        }),
        'child': pd.DataFrame({
            'col_2': [2, 2, 3],
            'col_3': [7, 8, 9]
        }),
        'grandchild': pd.DataFrame({
            'col_4': [3, 2, 1],
            'col_5': [10, 11, 12],
            'col_7': [13, 14, 15]
        }),
        'grandchild_2': pd.DataFrame({
            'col_5': [10, 11, 12]
        })
    }

    # Run
    data_result = _simplify_data(data, metadata)

    # Assert
    expected_results = {
        'parent': pd.DataFrame({
            'col_1': [1, 2, 3]
        }),
        'child': pd.DataFrame({
            'col_2': [2, 2, 3],
        }),
        'grandchild': pd.DataFrame({
            'col_4': [3, 2, 1],
        }),
    }
    for table_name in metadata.tables:
        pd.testing.assert_frame_equal(data_result[table_name], expected_results[table_name])
