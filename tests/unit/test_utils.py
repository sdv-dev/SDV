import re
from collections import defaultdict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.errors import InvalidDataError
from sdv.utils import drop_unknown_references


@patch('sdv.utils._get_rows_to_drop')
def test_drop_unknown_references(mock_get_rows_to_drop):
    """Test ``drop_unknown_references``."""
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
            'B': ['Yes', 'No', 'No', 'No', 'No']
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6],
            'child_foreign_key': [9, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No']
        })
    }
    mock_get_rows_to_drop.return_value = defaultdict(set, {
        'child': {4},
        'grandchild': {0, 2, 4}
    })

    # Run
    result = drop_unknown_references(metadata, data)

    # Assert
    metadata.validate.assert_called_once()
    metadata.validate_data.assert_called_once_with(data)
    mock_get_rows_to_drop.assert_called_once()
    expected_result = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2],
            'id_child': [5, 6, 7, 8],
            'B': ['Yes', 'No', 'No', 'No']
        }, index=[0, 1, 2, 3]),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [1, 2],
            'child_foreign_key': [5, 6],
            'C': ['No', 'No']
        }, index=[1, 3])
    }
    for table_name, table in result.items():
        pd.testing.assert_frame_equal(table, expected_result[table_name])


def test_drop_unknown_references_valid_data_mock():
    """Test ``drop_unknown_references`` when data has referential integrity."""
    # Setup
    metadata = Mock()
    metadata._get_all_foreign_keys.side_effect = [
        [], ['parent_foreign_key'], ['child_foreign_key', 'parent_foreign_key']
    ]
    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 3],
            'id_child': [5, 6, 7, 8, 9],
            'B': ['Yes', 'No', 'No', 'No', 'No']
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 3],
            'child_foreign_key': [6, 5, 7, 6, 9],
            'C': ['Yes', 'No', 'No', 'No', 'No']
        })
    }

    # Run
    result = drop_unknown_references(metadata, data)

    # Assert
    metadata.validate.assert_called_once()
    metadata.validate_data.assert_called_once_with(data)
    for table_name, table in result.items():
        pd.testing.assert_frame_equal(table, data[table_name])


@patch('sdv.utils._get_rows_to_drop')
@patch('sdv.utils._validate_foreign_keys_not_null')
def test_drop_unknown_references_with_nan(mock_validate_foreign_keys, mock_get_rows_to_drop):
    """Test ``drop_unknown_references`` whith NaNs and drop_missing_values True."""
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
    metadata.tables = {'parent', 'child', 'grandchild'}
    mock_validate_foreign_keys.side_effect = InvalidDataError('Invalid data')

    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 5, None],
            'id_child': [5, 6, 7, 8, 9, 10],
            'B': ['Yes', 'No', 'No', 'No', 'No', 'No']
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6, 4],
            'child_foreign_key': [9, np.nan, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No', 'No']
        })
    }
    mock_get_rows_to_drop.return_value = defaultdict(set, {
        'child': {4},
        'grandchild': {0, 3, 4}
    })

    # Run
    result = drop_unknown_references(metadata, data)

    # Assert
    metadata.validate.assert_called_once()
    metadata.validate_data.assert_called_once_with(data)
    mock_validate_foreign_keys.assert_called_once_with(metadata, data)
    mock_validate_foreign_keys.assert_called_once_with(metadata, data)
    mock_get_rows_to_drop.assert_called_once()
    expected_result = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0., 1., 2., 2.],
            'id_child': [5, 6, 7, 8],
            'B': ['Yes', 'No', 'No', 'No']
        }, index=[0, 1, 2, 3]),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [2, 4],
            'child_foreign_key': [5., 4.],
            'C': ['No', 'No']
        }, index=[2, 5])
    }
    for table_name, table in result.items():
        pd.testing.assert_frame_equal(table, expected_result[table_name])


@patch('sdv.utils._get_rows_to_drop')
def test_drop_unknown_references_drop_missing_values_false(mock_get_rows_to_drop):
    """Test ``drop_unknown_references`` with NaNs and drop_missing_values False."""
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
            'B': ['Yes', 'No', 'No', 'No', 'No', 'No']
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6, 4],
            'child_foreign_key': [9, np.nan, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No', 'No']
        })
    }
    mock_get_rows_to_drop.return_value = defaultdict(set, {
        'child': {4},
        'grandchild': {0, 3, 4}
    })

    # Run
    result = drop_unknown_references(metadata, data, drop_missing_values=False)

    # Assert
    mock_get_rows_to_drop.assert_called_once()
    expected_result = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0., 1., 2., 2., None],
            'id_child': [5, 6, 7, 8, 10],
            'B': ['Yes', 'No', 'No', 'No', 'No']
        }, index=[0, 1, 2, 3, 5]),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [1, 2, 4],
            'child_foreign_key': [np.nan, 5, 4.],
            'C': ['No', 'No', 'No']
        }, index=[1, 2, 5])
    }
    for table_name, table in result.items():
        pd.testing.assert_frame_equal(table, expected_result[table_name])


@patch('sdv.utils._get_rows_to_drop')
def test_drop_unknown_references_drop_all_rows(mock_get_rows_to_drop):
    """Test ``drop_unknown_references`` when all rows are dropped."""
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
            'B': ['Yes', 'No', 'No', 'No', 'No']
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 6],
            'child_foreign_key': [9, 5, 11, 6, 4],
            'C': ['Yes', 'No', 'No', 'No', 'No']
        })
    }

    mock_get_rows_to_drop.return_value = defaultdict(set, {
        'child': {0, 1, 2, 3, 4}
    })

    # Run and Assert
    expected_message = re.escape(
        'The provided data does not match the metadata:\n'
        "All references in table 'child' are unknown and must be dropped."
        'Try providing different data for this table.'
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        drop_unknown_references(metadata, data)
