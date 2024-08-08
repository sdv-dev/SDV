import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.errors import InvalidDataError
from sdv.metadata import MultiTableMetadata
from sdv.utils import drop_unknown_references, get_random_sequence_subset


@pytest.fixture
def metadata():
    return MultiTableMetadata.load_from_dict({
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


@pytest.fixture
def data():
    parent = pd.DataFrame(
        data={
            'id': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
            'B': [0.434, 0.312, 0.212, 0.339, 0.491],
        }
    )

    child = pd.DataFrame(
        data={'parent_id': [0, 1, 2, 2, 5], 'C': ['Yes', 'No', 'Maye', 'No', 'No']}
    )

    return {'parent': parent, 'child': child}


def test_drop_unknown_references(metadata, data, capsys):
    """Test ``drop_unknown_references`` end to end."""
    # Run
    expected_message = re.escape(
        'The provided data does not match the metadata:\n'
        'Relationships:\n'
        "Error: foreign key column 'parent_id' contains unknown references: (5)"
        ". Please use the method 'drop_unknown_references' from sdv.utils to clean the data."
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        metadata.validate_data(data)

    cleaned_data = drop_unknown_references(data, metadata)
    metadata.validate_data(cleaned_data)
    captured = capsys.readouterr()

    # Assert
    pd.testing.assert_frame_equal(cleaned_data['parent'], data['parent'])
    pd.testing.assert_frame_equal(cleaned_data['child'], data['child'].iloc[:4])
    assert len(cleaned_data['child']) == 4
    expected_output = (
        'Success! All foreign keys have referential integrity.\n\n'
        'Table Name  # Rows (Original)  # Invalid Rows  # Rows (New)\n'
        '     child                  5               1             4\n'
        '    parent                  5               0             5'
    )
    assert captured.out.strip() == expected_output


def test_drop_unknown_references_valid_data(metadata, data, capsys):
    """Test ``drop_unknown_references`` when data has referential integrity."""
    # Setup
    data = deepcopy(data)
    data['child'].loc[4, 'parent_id'] = 2

    # Run
    result = drop_unknown_references(data, metadata)
    captured = capsys.readouterr()

    # Assert
    pd.testing.assert_frame_equal(result['parent'], data['parent'])
    pd.testing.assert_frame_equal(result['child'], data['child'])
    expected_message = (
        'Success! All foreign keys have referential integrity.\n\n'
        'Table Name  # Rows (Original)  # Invalid Rows  # Rows (New)\n'
        '     child                  5               0             5\n'
        '    parent                  5               0             5'
    )
    assert captured.out.strip() == expected_message


def test_drop_unknown_references_drop_missing_values(metadata, data, capsys):
    """Test ``drop_unknown_references`` when there is missing values in the foreign keys."""
    # Setup
    data = deepcopy(data)
    data['child'].loc[4, 'parent_id'] = np.nan

    # Run
    cleaned_data = drop_unknown_references(data, metadata)
    metadata.validate_data(cleaned_data)
    captured = capsys.readouterr()

    # Assert
    pd.testing.assert_frame_equal(cleaned_data['parent'], data['parent'])
    pd.testing.assert_frame_equal(cleaned_data['child'], data['child'].iloc[:4])
    assert len(cleaned_data['child']) == 4
    expected_output = (
        'Success! All foreign keys have referential integrity.\n\n'
        'Table Name  # Rows (Original)  # Invalid Rows  # Rows (New)\n'
        '     child                  5               1             4\n'
        '    parent                  5               0             5'
    )
    assert captured.out.strip() == expected_output


def test_drop_unknown_references_not_drop_missing_values(metadata, data):
    """Test ``drop_unknown_references`` when the missing values in the foreign keys are kept."""
    # Setup
    data['child'].loc[3, 'parent_id'] = np.nan

    # Run
    cleaned_data = drop_unknown_references(data, metadata, drop_missing_values=False, verbose=False)

    # Assert
    pd.testing.assert_frame_equal(cleaned_data['parent'], data['parent'])
    pd.testing.assert_frame_equal(cleaned_data['child'], data['child'].iloc[:4])
    assert pd.isna(cleaned_data['child']['parent_id']).any()
    assert len(cleaned_data['child']) == 4


def test_get_random_sequence_subset():
    """Test that the sequences are subsetted and properly clipped."""
    # Setup
    data, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019')

    # Run
    metadata = metadata._convert_to_single_table()
    subset = get_random_sequence_subset(data, metadata, num_sequences=3, max_sequence_length=5)

    # Assert
    selected_sequences = subset[metadata.sequence_key].unique()
    assert len(selected_sequences) == 3
    for sequence_key in selected_sequences:
        pd.testing.assert_frame_equal(
            subset[subset[metadata.sequence_key] == sequence_key].reset_index(drop=True),
            data[data[metadata.sequence_key] == sequence_key].head(5).reset_index(drop=True),
        )


def test_get_random_sequence_subset_random_clipping():
    """Test that the sequences are subsetted and properly clipped.

    If the long_sequence_sampling_method is set to 'random', the selected sequences should be
    subsampled randomly, but maintain the same order.
    """
    # Setup
    data, metadata = download_demo(modality='sequential', dataset_name='nasdaq100_2019')
    metadata = metadata._convert_to_single_table()

    # Run
    subset = get_random_sequence_subset(
        data,
        metadata,
        num_sequences=3,
        max_sequence_length=5,
        long_sequence_subsampling_method='random',
    )

    # Assert
    selected_sequences = subset[metadata.sequence_key].unique()
    assert len(selected_sequences) == 3
    for sequence_key in selected_sequences:
        selected_sequence = subset[subset[metadata.sequence_key] == sequence_key]
        assert len(selected_sequence) <= 5
        subset_data = data[
            data['Date'].isin(selected_sequence['Date'])
            & data['Symbol'].isin(selected_sequence['Symbol'])
        ]
        pd.testing.assert_frame_equal(
            subset_data.reset_index(drop=True), selected_sequence.reset_index(drop=True)
        )
