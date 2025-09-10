import re
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata.metadata import Metadata
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS, HMASynthesizer
from sdv.multi_table.utils import _get_total_estimated_columns
from sdv.utils.poc import get_random_subset, simplify_schema


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
        data={'parent_id': [0, 1, 2, 2, 5], 'C': ['Yes', 'No', 'Maybe', 'No', 'No']}
    )

    return {'parent': parent, 'child': child}


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


def test_simplify_schema(capsys, large_data, large_metadata):
    """Test ``simplify_schema`` end to end."""
    # Setup
    num_estimated_column_before_simplification = _get_total_estimated_columns(large_metadata)
    HMASynthesizer(large_metadata)
    captured_before_simplification = capsys.readouterr()

    # Run
    data_simplify, metadata_simplify = simplify_schema(large_data, large_metadata)
    captured_after_simplification = capsys.readouterr()

    # Assert
    expected_message_before = re.compile(
        r'PerformanceAlert: Using the HMASynthesizer on this metadata schema is not recommended\.'
        r' To model this data, HMA will generate a large number of columns\. \(1034 columns\)\s+'
        r'Table Name\s*#\s*Columns in Metadata\s*Est # Columns\s*'
        r'great_grandparent\s*1\s*986\s*'
        r'grandparent\s*1\s*41\s*'
        r'parent\s*1\s*6\s*'
        r'child\s*1\s*1\s*'
        r'We recommend simplifying your metadata schema using '
        r"'sdv.utils.poc.simplify_schema'\.\s*"
        r'If this is not possible, please visit '
        r'datacebo.com and reach out to us for enterprise solutions\.'
    )
    expected_message_after = re.compile(
        r'Success! The schema has been simplified\.\s+'
        r'Table Name\s*#\s*Columns \(Before\)\s*#\s*Columns \(After\)\s*'
        r'child\s*3\s*0\s*'
        r'grandparent\s*3\s*3\s*'
        r'great_grandparent\s*2\s*2\s*'
        r'parent\s*3\s*2'
    )
    assert expected_message_before.match(captured_before_simplification.out.strip())
    assert expected_message_after.match(captured_after_simplification.out.strip())
    metadata_simplify.validate()
    metadata_simplify.validate_data(data_simplify)
    num_estimated_column_after_simplification = _get_total_estimated_columns(metadata_simplify)
    assert num_estimated_column_before_simplification == 1034
    assert num_estimated_column_after_simplification == 13


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


@pytest.mark.skip()
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
