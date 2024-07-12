import re
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata import MultiTableMetadata
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS, HMASynthesizer
from sdv.multi_table.utils import _get_total_estimated_columns
from sdv.utils.poc import get_random_subset, simplify_schema


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
        data={'parent_id': [0, 1, 2, 2, 5], 'C': ['Yes', 'No', 'Maybe', 'No', 'No']}
    )

    return {'parent': parent, 'child': child}


def test_simplify_schema(capsys):
    """Test ``simplify_schema`` end to end."""
    # Setup
    data, metadata = download_demo('multi_table', 'AustralianFootball_v1')
    num_estimated_column_before_simplification = _get_total_estimated_columns(metadata)
    HMASynthesizer(metadata)
    captured_before_simplification = capsys.readouterr()

    # Run
    data_simplify, metadata_simplify = simplify_schema(data, metadata)
    captured_after_simplification = capsys.readouterr()

    # Assert
    expected_message_before = re.compile(
        r'PerformanceAlert: Using the HMASynthesizer on this metadata schema is not recommended\.'
        r' To model this data, HMA will generate a large number of columns\. \(173818 columns\)\s+'
        r'Table Name\s*#\s*Columns in Metadata\s*Est # Columns\s*'
        r'match_stats\s*24\s*24\s*'
        r'matches\s*39\s*412\s*'
        r'players\s*5\s*378\s*'
        r'teams\s*1\s*173004\s*'
        r"We recommend simplifying your metadata schema using 'sdv.utils.poc.simplify_schema'\.\s*"
        r'If this is not possible, contact us at info@sdv.dev for enterprise solutions\.'
    )
    expected_message_after = re.compile(
        r'Success! The schema has been simplified\.\s+'
        r'Table Name\s*#\s*Columns \(Before\)\s*#\s*Columns \(After\)\s*'
        r'match_stats\s*28\s*4\s*'
        r'matches\s*42\s*21\s*'
        r'players\s*6\s*0\s*'
        r'teams\s*2\s*2'
    )
    assert expected_message_before.match(captured_before_simplification.out.strip())
    assert expected_message_after.match(captured_after_simplification.out.strip())
    metadata_simplify.validate()
    metadata_simplify.validate_data(data_simplify)
    num_estimated_column_after_simplification = _get_total_estimated_columns(metadata_simplify)
    assert num_estimated_column_before_simplification == 173818
    assert num_estimated_column_after_simplification == 517


def test_simpliy_nothing_to_simplify():
    """Test ``simplify_schema`` end to end when no simplification is required."""
    # Setup
    data, metadata = download_demo('multi_table', 'Biodegradability_v1')

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
    data, metadata = download_demo('multi_table', 'MuskSmall_v1')
    num_estimated_column_before_simplification = _get_total_estimated_columns(metadata)

    # Run
    data_simplify, metadata_simplify = simplify_schema(data, metadata)

    # Assert
    metadata_simplify.validate()
    metadata_simplify.validate_data(data_simplify)
    num_estimated_column_after_simplification = _get_total_estimated_columns(metadata_simplify)
    assert num_estimated_column_before_simplification == 14527
    assert num_estimated_column_after_simplification == 982


def test_simplify_schema_big_demo_datasets():
    """Test ``simplify_schema`` end to end for demo datasets that require simplification.

    This test will fail if the number of estimated columns after simplification is greater than
    the maximum number of columns allowed for any dataset.
    """
    # Setup
    list_datasets = [
        'AustralianFootball_v1',
        'MuskSmall_v1',
        'Countries_v1',
        'NBA_v1',
        'NCAA_v1',
        'PremierLeague_v1',
        'financial_v1',
    ]
    for dataset in list_datasets:
        real_data, metadata = download_demo('multi_table', dataset)

        # Run
        _data_simplify, metadata_simplify = simplify_schema(real_data, metadata)

        # Assert
        estimate_column_before = _get_total_estimated_columns(metadata)
        estimate_column_after = _get_total_estimated_columns(metadata_simplify)
        assert estimate_column_before > MAX_NUMBER_OF_COLUMNS
        assert estimate_column_after <= MAX_NUMBER_OF_COLUMNS


@pytest.mark.parametrize(
    ('dataset_name', 'main_table_1', 'main_table_2', 'num_rows_1', 'num_rows_2'),
    [
        ('AustralianFootball_v1', 'matches', 'players', 1000, 1000),
        ('MuskSmall_v1', 'molecule', 'conformation', 50, 150),
        ('NBA_v1', 'Team', 'Actions', 10, 200),
        ('NCAA_v1', 'tourney_slots', 'tourney_compact_results', 1000, 1000),
    ],
)
def test_get_random_subset(dataset_name, main_table_1, main_table_2, num_rows_1, num_rows_2):
    """Test ``get_random_subset`` end to end.

    The goal here is test that the function works for various schema and also by subsampling
    different main tables.

    For `AustralianFootball_v1` (parent with child and grandparent):
    - main table 1 = `matches` which is the child of `teams` and the parent of `match_stats`.
    - main table 2 = `players` which is the parent of `matches`.

    For `MuskSmall_v1` (1 parent - 1 child relationship):
    - main table 1 = `molecule` which is the parent of `conformation`.
    - main table 2 = `conformation` which is the child of `molecule`.

    For `NBA_v1` (child with parents and grandparent):
    - main table 1 = `Team` which is the root table.
    - main table 2 = `Actions` which is the last child. It has relationships with `Game` and `Team`
      and `Player`.

    For `NCAA_v1` (child with multiple parents):
    - main table 1 = `tourney_slots` which is only the child of `seasons`.
    - main table 2 = `tourney_compact_results` which is the child of `teams` with two relationships
      and of `seasons` with one relationship.
    """
    # Setup
    real_data, metadata = download_demo('multi_table', dataset_name)

    # Run
    result_1 = get_random_subset(real_data, metadata, main_table_1, num_rows_1, verbose=False)
    result_2 = get_random_subset(real_data, metadata, main_table_2, num_rows_2, verbose=False)

    # Assert
    assert len(result_1[main_table_1]) == num_rows_1
    assert len(result_2[main_table_2]) == num_rows_2


def test_get_random_subset_disconnected_schema():
    """Test ``get_random_subset`` end to end for a disconnected schema.

    Here we break the schema so there is only parent-child relationships between
    `Player`-`Action` and `Team`-`Game`.
    The part that is not connected to the main table (`Player`) should be subsampled also
    in a similar proportion.
    """
    # Setup
    real_data, metadata = download_demo('multi_table', 'NBA_v1')
    metadata.remove_relationship('Game', 'Actions')
    metadata.remove_relationship('Team', 'Actions')
    metadata.validate = Mock()
    metadata.validate_data = Mock()
    proportion_to_keep = 0.6
    num_rows_to_keep = int(len(real_data['Player']) * proportion_to_keep)

    # Run
    result = get_random_subset(real_data, metadata, 'Player', num_rows_to_keep, verbose=False)

    # Assert
    assert len(result['Player']) == num_rows_to_keep
    assert len(result['Team']) == int(len(real_data['Team']) * proportion_to_keep)


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
    assert result['child']['parent_id'].ina().sum() > 0
