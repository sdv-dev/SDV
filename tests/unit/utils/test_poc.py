import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.errors import InvalidDataError
from sdv.metadata import MultiTableMetadata, SingleTableMetadata
from sdv.metadata.errors import InvalidMetadataError
from sdv.utils.poc import (
    drop_unknown_references,
    get_random_sequence_subset,
    get_random_subset,
    simplify_schema,
)


@patch('sdv.utils.poc.utils_drop_unknown_references')
def test_drop_unknown_references(mock_drop_unknown_references):
    """Test ``drop_unknown_references`` raise a FutureWarning when called from sdv.utils.poc."""
    # Setup
    data = Mock()
    metadata = Mock()
    expected_message = re.escape(
        "Please access the 'drop_unknown_references' function directly from the sdv.utils module"
        'instead of sdv.utils.poc.'
    )

    # Run
    with pytest.warns(FutureWarning, match=expected_message):
        drop_unknown_references(data, metadata)

    # Assert
    mock_drop_unknown_references.assert_called_once_with(data, metadata)


@patch('sdv.utils.poc._get_total_estimated_columns')
@patch('sdv.utils.poc._print_simplified_schema_summary')
def test_simplify_schema_nothing_to_simplify(mock_print_summary, mock_get_total_estimated_columns):
    """Test ``simplify_schema`` when the schema is already simple."""
    # Setup
    data = Mock()
    metadata = Mock()
    mock_get_total_estimated_columns.return_value = 5

    # Run
    result_data, result_metadata = simplify_schema(data, metadata)

    # Assert
    mock_print_summary.assert_called_once_with(data, data)
    mock_get_total_estimated_columns.assert_called_once_with(metadata)
    assert result_data is data
    assert result_metadata is metadata


@patch('sdv.utils.poc._simplify_metadata')
@patch('sdv.utils.poc._simplify_data')
@patch('sdv.utils.poc._get_total_estimated_columns')
@patch('sdv.utils.poc._print_simplified_schema_summary')
def test_simplify_schema(
    mock_print_summary, mock_get_total_estimated_columns, mock_simplify_data, mock_simplify_metadata
):
    """Test ``simplify_schema``."""
    # Setup
    data = Mock()
    metadata = Mock()
    simplified_metatadata = MultiTableMetadata()
    mock_get_total_estimated_columns.return_value = 2000
    mock_simplify_metadata.return_value = simplified_metatadata
    mock_simplify_data.return_value = {
        'table1': pd.DataFrame({'column1': [1, 2, 3]}),
    }

    # Run
    result_data, result_metadata = simplify_schema(data, metadata)

    # Assert
    mock_print_summary.assert_called_once_with(data, result_data)
    mock_get_total_estimated_columns.assert_called_once_with(metadata)
    mock_simplify_metadata.assert_called_once_with(metadata)
    mock_simplify_data.assert_called_once_with(data, simplified_metatadata)
    pd.testing.assert_frame_equal(result_data['table1'], pd.DataFrame({'column1': [1, 2, 3]}))
    assert result_data.keys() == {'table1'}
    assert result_metadata == simplified_metatadata


def test_simplify_schema_invalid_metadata():
    """Test ``simplify_schema`` when the metadata is not invalid."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {'table1': {'columns': {'column1': {'sdtype': 'categorical'}}}},
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'column1',
                'child_foreign_key': 'column2',
            }
        ],
    })
    real_data = {
        'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        'table2': pd.DataFrame({'column2': [4, 5, 6]}),
    }

    # Run and Assert
    expected_message = re.escape(
        'The provided data/metadata combination is not valid. Please make sure that the'
        ' data/metadata combination is valid before trying to simplify the schema.'
    )
    with pytest.raises(InvalidMetadataError, match=expected_message):
        simplify_schema(real_data, metadata)


def test_simplify_schema_invalid_data():
    """Test ``simplify_schema`` when the data is not valid."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'table1': {'columns': {'column1': {'sdtype': 'id'}}, 'primary_key': 'column1'},
            'table2': {
                'columns': {'column2': {'sdtype': 'id'}},
            },
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'column1',
                'child_foreign_key': 'column2',
            }
        ],
    })
    real_data = {
        'table1': pd.DataFrame({'column1': [np.nan, 1, 2]}),
        'table2': pd.DataFrame({'column2': [1, 1, 2]}),
    }

    # Run and Assert
    expected_message = re.escape(
        'The provided data/metadata combination is not valid. Please make sure that the'
        ' data/metadata combination is valid before trying to simplify the schema.'
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        simplify_schema(real_data, metadata)


def test_get_random_subset_invalid_metadata():
    """Test ``get_random_subset`` when the metadata is invalid."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {'table1': {'columns': {'column1': {'sdtype': 'categorical'}}}},
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'column1',
                'child_foreign_key': 'column2',
            }
        ],
    })
    real_data = {
        'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        'table2': pd.DataFrame({'column2': [4, 5, 6]}),
    }

    # Run and Assert
    expected_message = re.escape(
        'The provided data/metadata combination is not valid. Please make sure that the'
        ' data/metadata combination is valid before trying to simplify the schema.'
    )
    with pytest.raises(InvalidMetadataError, match=expected_message):
        get_random_subset(real_data, metadata, 'table1', 2)


def test_get_random_subset_invalid_data():
    """Test ``get_random_subset`` when the data is not valid."""
    # Setup
    metadata = MultiTableMetadata().load_from_dict({
        'tables': {
            'table1': {'columns': {'column1': {'sdtype': 'id'}}, 'primary_key': 'column1'},
            'table2': {
                'columns': {'column2': {'sdtype': 'id'}},
            },
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'column1',
                'child_foreign_key': 'column2',
            }
        ],
    })
    real_data = {
        'table1': pd.DataFrame({'column1': [np.nan, 1, 2]}),
        'table2': pd.DataFrame({'column2': [1, 1, 2]}),
    }

    # Run and Assert
    expected_message = re.escape(
        'The provided data/metadata combination is not valid. Please make sure that the'
        ' data/metadata combination is valid before trying to simplify the schema.'
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        get_random_subset(real_data, metadata, 'table1', 2)


def test_get_random_subset_invalid_num_rows():
    """Test ``get_random_subset`` when ``num_rows`` is invalid."""
    # Setup
    data = Mock()
    metadata = Mock()

    # Run and Assert
    with pytest.raises(ValueError, match='``num_rows`` must be a positive integer.'):
        get_random_subset(data, metadata, 'table1', -1)
    with pytest.raises(ValueError, match='``num_rows`` must be a positive integer.'):
        get_random_subset(data, metadata, 'table1', 0)
    with pytest.raises(ValueError, match='``num_rows`` must be a positive integer.'):
        get_random_subset(data, metadata, 'table1', 0.5)


def test_get_random_subset_nothing_to_sample():
    """Test ``get_random_subset`` when there is nothing to sample."""
    # Setup
    data = {
        'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        'table2': pd.DataFrame({'column2': [4, 5, 6]}),
    }
    metadata = Mock()

    # Run
    result = get_random_subset(data, metadata, 'table1', 5)

    # Assert
    pd.testing.assert_frame_equal(result['table1'], data['table1'])
    pd.testing.assert_frame_equal(result['table2'], data['table2'])


@patch('sdv.utils.poc._subsample_data')
@patch('sdv.utils.poc._print_subsample_summary')
def test_get_random_subset(mock_print_summary, mock_subsample_data):
    """Test ``get_random_subset``."""
    # Setup
    data = {
        'table1': pd.DataFrame({'column1': [1, 2, 3, 4, 5]}),
        'table2': pd.DataFrame({'column2': [6, 7, 8, 9, 10]}),
    }
    metadata = Mock()
    output = {
        'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        'table2': pd.DataFrame({'column2': [6, 7, 8]}),
    }
    mock_subsample_data.return_value = output

    # Run
    get_random_subset(data, metadata, 'table1', 3)
    result = get_random_subset(data, metadata, 'table2', 3, verbose=False)

    # Assert
    pd.testing.assert_frame_equal(result['table1'], output['table1'])
    pd.testing.assert_frame_equal(result['table2'], output['table2'])
    mock_subsample_data.assert_has_calls([
        ((data, metadata, 'table1', 3),),
        ((data, metadata, 'table2', 3),),
    ])
    mock_print_summary.call_count == 1


def test_get_random_sequence_subset_no_sequence_key():
    """Test that an error is raised if no sequence_key is provided in the metadata."""
    # Setup
    metadata = Mock(spec=SingleTableMetadata)
    metadata.sequence_key = None

    # Run and Assert
    error_message = (
        'Your metadata does not include a sequence key. A sequence key must be provided to subset'
        ' the sequential data.'
    )
    with pytest.raises(ValueError, match=error_message):
        get_random_sequence_subset(pd.DataFrame(), metadata, 3)


def test_get_random_sequence_bad_long_sequence_subsampling_method():
    """Test that an error is raised if the long_sequence_subsampling_method is invalid."""
    # Setup
    metadata = Mock(spec=SingleTableMetadata)
    metadata.sequence_key = 'key'

    # Run and Assert
    error_message = (
        'long_sequence_subsampling_method must be one of "first_rows", "last_rows" or "random"'
    )
    with pytest.raises(ValueError, match=error_message):
        get_random_sequence_subset(pd.DataFrame(), metadata, 3, 10, 'blah')


@patch('sdv.utils.poc.np')
def test_get_random_sequence_subset_no_max_sequence_length(mock_np):
    """Test that the sequences are subsetted but each sequence is full."""
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    metadata.sequence_key = 'key'
    mock_np.random.permutation.return_value = np.array(['a', 'd'])

    # Run
    subset = get_random_sequence_subset(data, metadata, num_sequences=2)

    # Assert
    expected = pd.DataFrame({
        'key': ['a'] * 10 + ['d'] * 4,
        'value': list(range(10)) + [26, 27, 28, 29],
    })
    pd.testing.assert_frame_equal(expected, subset)


@patch('sdv.utils.poc.np')
def test_get_random_sequence_subset_use_first_rows(mock_np):
    """Test that the sequences are subsetted but each sequence is full."""
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    metadata.sequence_key = 'key'
    mock_np.random.permutation.return_value = np.array(['a', 'b', 'd'])

    # Run
    subset = get_random_sequence_subset(data, metadata, num_sequences=3, max_sequence_length=6)

    # Assert
    expected = pd.DataFrame({
        'key': ['a'] * 6 + ['b'] * 6 + ['d'] * 4,
        'value': [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29],
    })
    pd.testing.assert_frame_equal(expected, subset)


@patch('sdv.utils.poc.np')
def test_get_random_sequence_subset_use_last_rows(mock_np):
    """Test that the sequences are subsetted but each sequence is full."""
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    metadata.sequence_key = 'key'
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


@patch('sdv.utils.poc.np')
def test_get_random_sequence_subset_use_random_rows(mock_np):
    """Test that the sequences are subsetted but each sequence is full."""
    # Setup
    data = pd.DataFrame({'key': ['a'] * 10 + ['b'] * 7 + ['c'] * 9 + ['d'] * 4, 'value': range(30)})
    metadata = Mock(spec=SingleTableMetadata)
    metadata.sequence_key = 'key'
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
