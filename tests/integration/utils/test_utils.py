import json
import re
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.errors import InvalidDataError
from sdv.metadata.metadata import Metadata
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS
from sdv.multi_table.utils import _get_total_estimated_columns
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.utils import (
    drop_unknown_references,
    get_random_sequence_subset,
    get_random_subset,
    load_constraints,
    load_synthesizer,
    simplify_schema,
)


@pytest.fixture
def metadata():
    return Metadata.load_from_dict({
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
        "Error: foreign key column 'parent_id' contains unknown references:\n"
        '   parent_id\n'
        '4          5\n'
        "Please use the method 'drop_unknown_references' from sdv.utils to clean the data."
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
    cleaned_data = drop_unknown_references(data, metadata, drop_missing_values=True)
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


def test_load_synthesizer(tmp_path):
    """Test the `load_synthesizer` method."""
    # Setup
    data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(data)
    synthesizer.save(tmp_path / 'GCSynthesizer.pkl')

    # Run
    loaded_synthesizer = load_synthesizer(tmp_path / 'GCSynthesizer.pkl')
    synthetic_data = loaded_synthesizer.sample(num_rows=10)

    # Assert
    assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)
    assert set(synthetic_data.columns) == set(data.columns)


def test_load_constraints(tmp_path, constraint_object):
    """Test the `load_constraints` method."""
    # Setup
    constraints = [constraint_object.get_constraint_dict()]
    filepath = tmp_path / 'constraints.json'
    with open(filepath, 'w') as f:
        json.dump(constraints, f)

    # Run
    loaded_constraints = load_constraints(filepath)

    # Assert
    assert len(loaded_constraints) == 1
    assert loaded_constraints[0].get_constraint_dict() == constraints[0]


@pytest.fixture
def large_data():
    great_grandparent = pd.DataFrame({'ggp_id': [1, 2, 3], 'ggp_data': ['A', 'B', 'C']})
    grandparent = pd.DataFrame({
        'gp_id': [10, 11, 12, 13],
        'ggp_id': [1, 1, 2, 3],
        'gp_data': ['X', 'Y', 'Z', 'W'],
    })
    parent = pd.DataFrame({
        'p_id': [100, 101, 102, 103, 104],
        'gp_id': [10, 10, 11, 12, 13],
        'p_data': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
    })
    child = pd.DataFrame({
        'c_id': [1000, 1001, 1002, 1003, 1004, 1005],
        'p_id': [100, 100, 101, 102, 103, 104],
        'c_data': ['One', 'Two', 'Three', 'Four', 'Five', 'Six'],
    })
    return {
        'great_grandparent': great_grandparent,
        'grandparent': grandparent,
        'parent': parent,
        'child': child,
    }


@pytest.fixture
def large_metadata():
    return Metadata.load_from_dict({
        'tables': {
            'great_grandparent': {
                'columns': {'ggp_id': {'sdtype': 'id'}, 'ggp_data': {'sdtype': 'categorical'}},
                'primary_key': 'ggp_id',
            },
            'grandparent': {
                'columns': {
                    'gp_id': {'sdtype': 'id'},
                    'ggp_id': {'sdtype': 'id'},
                    'gp_data': {'sdtype': 'categorical'},
                },
                'primary_key': 'gp_id',
            },
            'parent': {
                'columns': {
                    'p_id': {'sdtype': 'id'},
                    'gp_id': {'sdtype': 'id'},
                    'p_data': {'sdtype': 'categorical'},
                },
                'primary_key': 'p_id',
            },
            'child': {
                'columns': {
                    'c_id': {'sdtype': 'id'},
                    'p_id': {'sdtype': 'id'},
                    'c_data': {'sdtype': 'categorical'},
                },
                'primary_key': 'c_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'great_grandparent',
                'parent_primary_key': 'ggp_id',
                'child_table_name': 'grandparent',
                'child_foreign_key': 'ggp_id',
            },
            {
                'parent_table_name': 'grandparent',
                'parent_primary_key': 'gp_id',
                'child_table_name': 'parent',
                'child_foreign_key': 'gp_id',
            },
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'p_id',
                'child_table_name': 'child',
                'child_foreign_key': 'p_id',
            },
        ],
    })


def test_simplify_schema(capsys):
    """Test ``simplify_schema`` end to end."""
    # Setup
    data, metadata = download_demo('multi_table', 'AustralianFootball')
    num_estimated_column_before_simplification = _get_total_estimated_columns(metadata)

    # Run
    data_simplify, metadata_simplify = simplify_schema(data, metadata)
    captured_after_simplification = capsys.readouterr()

    # Assert
    expected_message_after = re.compile(
        r'Success! The schema has been simplified\.\s+'
        r'Table Name\s*#\s*Columns \(Before\)\s*#\s*Columns \(After\)\s*'
        r'match_stats\s*28\s*3\s*'
        r'matches\s*42\s*21\s*'
        r'players\s*6\s*0\s*'
        r'teams\s*2\s*2'
    )
    assert expected_message_after.match(captured_after_simplification.out.strip())
    metadata_simplify.validate()
    metadata_simplify.validate_data(data_simplify)
    num_estimated_column_after_simplification = _get_total_estimated_columns(metadata_simplify)
    assert num_estimated_column_before_simplification == 173818
    assert num_estimated_column_after_simplification == 517


def test_simpliy_nothing_to_simplify():
    """Test ``simplify_schema`` end to end when no simplification is required."""
    # Setup
    data, metadata = download_demo('multi_table', 'fake_hotels')

    # Run
    data_simplify, metadata_simplify = simplify_schema(data, metadata)

    # Assert
    metadata_simplify.validate()
    metadata_simplify.validate_data(data_simplify)
    assert metadata.to_dict() == metadata_simplify.to_dict()
    for table in data:
        pd.testing.assert_frame_equal(data[table], data_simplify[table])


def test_simplify_no_grandchild():
    """Test ``simplify_schema`` end to end when there is no grandchild table."""
    # Setup
    parent_data = pd.DataFrame({
        'parent_id': range(500),
        'parent_col1': np.random.choice(['A', 'B', 'C'], 500),
        'parent_col2': np.random.randn(500),
    })
    child_columns = {'child_id': range(500), 'parent_id': np.random.choice(range(500), 500)}
    for i in range(168):
        child_columns[f'child_col_{i}'] = np.random.choice(['X', 'Y', 'Z'], 500)
    child_data = pd.DataFrame(child_columns)
    data = {'parent': parent_data, 'child': child_data}
    parent_columns = {
        'parent_id': {'sdtype': 'id'},
        'parent_col1': {'sdtype': 'categorical'},
        'parent_col2': {'sdtype': 'numerical'},
    }
    child_columns_meta = {'child_id': {'sdtype': 'id'}, 'parent_id': {'sdtype': 'id'}}
    for i in range(168):
        child_columns_meta[f'child_col_{i}'] = {'sdtype': 'categorical'}

    metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {'columns': parent_columns, 'primary_key': 'parent_id'},
            'child': {'columns': child_columns_meta, 'primary_key': 'child_id'},
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

    # Run
    num_estimated_column_before_simplification = _get_total_estimated_columns(metadata)
    data_simplify, metadata_simplify = simplify_schema(data, metadata)

    # Assert
    metadata_simplify.validate()
    metadata_simplify.validate_data(data_simplify)
    num_estimated_column_after_simplification = _get_total_estimated_columns(metadata_simplify)
    assert num_estimated_column_before_simplification > num_estimated_column_after_simplification


def test_simplify_schema_big_demo_datasets(large_data, large_metadata):
    """Test ``simplify_schema`` end to end for demo datasets that require simplification.

    This test will fail if the number of estimated columns after simplification is greater than
    the maximum number of columns allowed for any dataset.
    """
    # Run
    _data_simplify, metadata_simplify = simplify_schema(large_data, large_metadata)

    # Assert
    estimate_column_before = _get_total_estimated_columns(large_metadata)
    estimate_column_after = _get_total_estimated_columns(metadata_simplify)
    assert estimate_column_before > MAX_NUMBER_OF_COLUMNS
    assert estimate_column_after <= MAX_NUMBER_OF_COLUMNS


def test_get_random_subset():
    """Test ``get_random_subset`` end to end.

    The goal here is test that the function works for various schema and also by subsampling
    different main tables.
    """
    # Setup
    real_data, metadata = download_demo('multi_table', 'fake_hotels')

    # Run
    result_1 = get_random_subset(real_data, metadata, 'hotels', 10, verbose=False)
    result_2 = get_random_subset(real_data, metadata, 'guests', 20, verbose=False)

    # Assert
    assert len(result_1['hotels']) == 10
    assert len(result_2['guests']) == 20


def test_get_random_subset_disconnected_schema():
    """Test ``get_random_subset`` end to end for a disconnected schema."""
    # Setup
    real_data, metadata = download_demo('multi_table', 'fake_hotels')
    metadata.remove_relationship('hotels', 'guests')
    metadata.validate = Mock()
    metadata.validate_data = Mock()
    proportion_to_keep = 0.6
    num_rows_to_keep = int(len(real_data['guests']) * proportion_to_keep)

    # Run
    result = get_random_subset(real_data, metadata, 'guests', num_rows_to_keep, verbose=False)

    # Assert
    assert len(result['guests']) == num_rows_to_keep
    assert len(result['hotels']) >= int(len(real_data['hotels']) * proportion_to_keep)


def test_get_random_subset_with_missing_values(metadata, data):
    """Test ``get_random_subset`` when there is missing values in the foreign keys.

    Here there should be at least one missing values in the random subset.
    """
    # Setup
    data = deepcopy(data)
    data['child'].loc[[2, 3, 4], 'parent_id'] = np.nan

    # Run
    result = get_random_subset(data, metadata, 'child', 3)

    # Assert
    assert len(result['child']) == 3
    assert result['child']['parent_id'].isna().sum() > 0
