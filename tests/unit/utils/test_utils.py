import re
from collections import defaultdict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.errors import InvalidDataError
from sdv.metadata import SingleTableMetadata
from sdv.metadata.metadata import Metadata
from sdv.utils.utils import drop_unknown_references, get_random_sequence_subset


@patch('sdv.utils.utils._drop_rows')
def test_drop_unknown_references(mock_drop_rows):
    """Test ``drop_unknown_references``."""
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

    def _drop_rows(data, metadata, drop_missing_values):
        data['child'] = data['child'].iloc[:4]
        data['grandchild'] = data['grandchild'].iloc[[1, 3]]

    mock_drop_rows.side_effect = _drop_rows

    # Run
    result = drop_unknown_references(data, metadata)

    # Assert
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
    metadata.validate.assert_called_once()
    metadata.validate_data.assert_called_once_with(data)
    mock_drop_rows.assert_called_once_with(result, metadata, True)
    for table_name, table in result.items():
        pd.testing.assert_frame_equal(table, expected_result[table_name])


@patch('sys.stdout.write')
def test_drop_unknown_references_valid_data_mock(mock_stdout_write):
    """Test ``drop_unknown_references`` when data has referential integrity."""
    # Setup
    metadata = Mock()
    metadata._get_all_foreign_keys.side_effect = [
        [],
        ['parent_foreign_key'],
        ['child_foreign_key', 'parent_foreign_key'],
    ]
    metadata.tables = {'parent', 'child', 'grandchild'}
    data = {
        'parent': pd.DataFrame({
            'id_parent': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
        }),
        'child': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 3],
            'id_child': [5, 6, 7, 8, 9],
            'B': ['Yes', 'No', 'No', 'No', 'No'],
        }),
        'grandchild': pd.DataFrame({
            'parent_foreign_key': [0, 1, 2, 2, 3],
            'child_foreign_key': [6, 5, 7, 6, 9],
            'C': ['Yes', 'No', 'No', 'No', 'No'],
        }),
    }

    # Run
    result = drop_unknown_references(data, metadata)

    # Assert
    expected_pattern = re.compile(
        r'Success! All foreign keys have referential integrity\.\s*'
        r'Table Name\s*#\s*Rows \(Original\)\s*#\s*Invalid Rows\s*#\s*Rows \(New\)\s*'
        r'child\s*5\s*0\s*5\s*'
        r'grandchild\s*5\s*0\s*5\s*'
        r'parent\s*5\s*0\s*5'
    )
    output = mock_stdout_write.call_args[0][0]
    assert expected_pattern.match(output)
    metadata.validate.assert_called_once()
    metadata.validate_data.assert_called_once_with(data)
    for table_name, table in result.items():
        pd.testing.assert_frame_equal(table, data[table_name])


@patch('sdv.multi_table.utils._get_rows_to_drop')
@patch('sdv.utils.utils._validate_foreign_keys_not_null')
def test_drop_unknown_references_with_nan(mock_validate_foreign_keys, mock_get_rows_to_drop):
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
    mock_validate_foreign_keys.side_effect = InvalidDataError('Invalid data')

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
    result = drop_unknown_references(data, metadata, verbose=False)

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
    for table_name, table in result.items():
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
    result = drop_unknown_references(data, metadata, drop_missing_values=False, verbose=False)

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
    for table_name, table in result.items():
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
        "All references in table 'child' are unknown and must be dropped."
        'Try providing different data for this table.'
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        drop_unknown_references(data, metadata)


@patch('sdv.metadata.Metadata._convert_to_unified_metadata')
def test_get_random_sequence_subset_no_sequence_key(mock_convert_metadata):
    """Test that an error is raised if no sequence_key is provided in the metadata."""
    # Setup
    metadata = Mock(spec=SingleTableMetadata)
    unified_metadata_mock = Mock(spec=Metadata)
    unified_metadata_mock.get_sequence_key.return_value = None
    mock_convert_metadata.return_value = unified_metadata_mock

    # Run and Assert
    error_message = (
        'Your metadata does not include a sequence key. A sequence key must be provided to subset'
        ' the sequential data.'
    )
    with pytest.raises(ValueError, match=error_message):
        get_random_sequence_subset(pd.DataFrame(), metadata, 3)


@patch('sdv.metadata.Metadata._convert_to_unified_metadata')
def test_get_random_sequence_subset_sequence_key_not_in_data(mock_convert_metadata):
    """Test that an error is raised if the data doesn't contain the sequence_key."""
    # Setup
    metadata = Mock(spec=SingleTableMetadata)
    unified_metadata_mock = Mock(spec=Metadata)
    unified_metadata_mock.get_sequence_key.return_value = 'key'
    mock_convert_metadata.return_value = unified_metadata_mock

    # Run and Assert
    error_message = (
        'Your provided sequence key is not in the data. This is required to get a subset.'
    )
    with pytest.raises(ValueError, match=error_message):
        get_random_sequence_subset(pd.DataFrame(), metadata, 3)


@patch('sdv.metadata.Metadata._convert_to_unified_metadata')
def test_get_random_sequence_subset_bad_long_sequence_subsampling_method(mock_convert_metadata):
    """Test that an error is raised if the long_sequence_subsampling_method is invalid."""
    # Setup
    metadata = Mock(spec=SingleTableMetadata)
    unified_metadata_mock = Mock(spec=Metadata)
    unified_metadata_mock.get_sequence_key.return_value = 'key'
    mock_convert_metadata.return_value = unified_metadata_mock

    # Run and Assert
    error_message = (
        'long_sequence_subsampling_method must be one of "first_rows", "last_rows" or "random"'
    )
    with pytest.raises(ValueError, match=error_message):
        get_random_sequence_subset(pd.DataFrame(), metadata, 3, 10, 'blah')


@patch('sdv.metadata.Metadata._convert_to_unified_metadata')
@patch('sdv.utils.utils.np')
def test_get_random_sequence_subset_no_max_sequence_length(mock_np, mock_convert_metadata):
    """Test that the sequences are subsetted but each sequence is full."""
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    unified_metadata_mock = Mock(spec=Metadata)
    unified_metadata_mock.get_sequence_key.return_value = 'key'
    mock_convert_metadata.return_value = unified_metadata_mock
    mock_np.random.permutation.return_value = np.array(['a', 'd'])

    # Run
    subset = get_random_sequence_subset(data, metadata, num_sequences=2)

    # Assert
    expected = pd.DataFrame({
        'key': ['a'] * 10 + ['d'] * 4,
        'value': list(range(10)) + [26, 27, 28, 29],
    })
    pd.testing.assert_frame_equal(expected, subset)


@patch('sdv.metadata.Metadata._convert_to_unified_metadata')
@patch('sdv.utils.utils.np')
def test_get_random_sequence_subset_use_first_rows(mock_np, mock_convert_metadata):
    """Test that the sequences are subsetted and subsampled properly.

    If 'long_sequence_subsampling_method' isn't set, the sequences should be clipped using the
    first 'max_sequence_length' rows.
    """
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    unified_metadata_mock = Mock(spec=Metadata)
    unified_metadata_mock.get_sequence_key.return_value = 'key'
    mock_convert_metadata.return_value = unified_metadata_mock
    mock_np.random.permutation.return_value = np.array(['a', 'b', 'd'])

    # Run
    subset = get_random_sequence_subset(data, metadata, num_sequences=3, max_sequence_length=6)

    # Assert
    expected = pd.DataFrame({
        'key': ['a'] * 6 + ['b'] * 6 + ['d'] * 4,
        'value': [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29],
    })
    pd.testing.assert_frame_equal(expected, subset)


@patch('sdv.metadata.Metadata._convert_to_unified_metadata')
@patch('sdv.utils.utils.np')
def test_get_random_sequence_subset_use_last_rows(mock_np, mock_convert_metadata):
    """Test that the sequences are subsetted and subsampled properly.

    If 'long_sequence_subsampling_method' isn't set, the sequences should be clipped using the
    last 'max_sequence_length' rows.
    """
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    unified_metadata_mock = Mock(spec=Metadata)
    unified_metadata_mock.get_sequence_key.return_value = 'key'
    mock_convert_metadata.return_value = unified_metadata_mock
    mock_np.random.permutation.return_value = np.array(['a', 'b', 'd'])

    # Run
    subset = get_random_sequence_subset(
        data,
        metadata,
        num_sequences=3,
        max_sequence_length=6,
        long_sequence_subsampling_method='last_rows',
    )

    # Assert
    expected = pd.DataFrame({
        'key': ['a'] * 6 + ['b'] * 6 + ['d'] * 4,
        'value': [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 26, 27, 28, 29],
    })
    pd.testing.assert_frame_equal(expected, subset)


@patch('sdv.metadata.Metadata._convert_to_unified_metadata')
@patch('sdv.utils.utils.np')
def test_get_random_sequence_subset_use_random_rows(mock_np, mock_convert_metadata):
    """Test that the sequences are subsetted and subsampled properly.

    If 'long_sequence_subsampling_method' isn't set, the sequences should be clipped using random
    'max_sequence_length' rows.
    """
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    unified_metadata_mock = Mock(spec=Metadata)
    unified_metadata_mock.get_sequence_key.return_value = 'key'
    mock_convert_metadata.return_value = unified_metadata_mock
    mock_np.random.permutation.side_effect = [
        np.array(['a', 'b', 'd']),
        np.array([0, 2, 4, 5, 7, 9]),
        np.array([6, 5, 1, 2, 4, 0]),
    ]

    # Run
    subset = get_random_sequence_subset(
        data,
        metadata,
        num_sequences=3,
        max_sequence_length=6,
        long_sequence_subsampling_method='random',
    )

    # Assert
    expected = pd.DataFrame({
        'key': ['a'] * 6 + ['b'] * 6 + ['d'] * 4,
        'value': [0, 2, 4, 5, 7, 9, 10, 11, 12, 14, 15, 16, 26, 27, 28, 29],
    })
    pd.testing.assert_frame_equal(expected, subset)
