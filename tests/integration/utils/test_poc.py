import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.errors import InvalidDataError
from sdv.metadata import MultiTableMetadata
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS, HMASynthesizer
from sdv.multi_table.utils import _get_total_estimated_columns
from sdv.utils.poc import drop_unknown_references, simplify_schema


@pytest.fixture
def metadata():
    return MultiTableMetadata.load_from_dict(
        {
            'tables': {
                'parent': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'A': {'sdtype': 'categorical'},
                        'B': {'sdtype': 'numerical'}
                    },
                    'primary_key': 'id'
                },
                'child': {
                    'columns': {
                        'parent_id': {'sdtype': 'id'},
                        'C': {'sdtype': 'categorical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'child_table_name': 'child',
                    'parent_primary_key': 'id',
                    'child_foreign_key': 'parent_id'
                }
            ]
        }
    )


@pytest.fixture
def data():
    parent = pd.DataFrame(data={
        'id': [0, 1, 2, 3, 4],
        'A': [True, True, False, True, False],
        'B': [0.434, 0.312, 0.212, 0.339, 0.491]
    })

    child = pd.DataFrame(data={
        'parent_id': [0, 1, 2, 2, 5],
        'C': ['Yes', 'No', 'Maye', 'No', 'No']
    })

    return {
        'parent': parent,
        'child': child
    }


def test_drop_unknown_references(metadata, data, capsys):
    """Test ``drop_unknown_references`` end to end."""
    # Run
    expected_message = re.escape(
        'The provided data does not match the metadata:\n'
        'Relationships:\n'
        "Error: foreign key column 'parent_id' contains unknown references: (5)"
        ". Please use the utility method 'drop_unknown_references' to clean the data."
    )
    with pytest.raises(InvalidDataError, match=expected_message):
        metadata.validate_data(data)

    cleaned_data = drop_unknown_references(metadata, data)
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
    result = drop_unknown_references(metadata, data)
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
    cleaned_data = drop_unknown_references(metadata, data)
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
    cleaned_data = drop_unknown_references(
        metadata, data, drop_missing_values=False, verbose=False
    )

    # Assert
    pd.testing.assert_frame_equal(cleaned_data['parent'], data['parent'])
    pd.testing.assert_frame_equal(cleaned_data['child'], data['child'].iloc[:4])
    assert pd.isna(cleaned_data['child']['parent_id']).any()
    assert len(cleaned_data['child']) == 4


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
        'financial_v1'
    ]
    for dataset in list_datasets:
        real_data, metadata = download_demo('multi_table', dataset)

        # Run
        data_simplify, metadata_simplify = simplify_schema(real_data, metadata)

        # Assert
        estimate_column_before = _get_total_estimated_columns(metadata)
        estimate_column_after = _get_total_estimated_columns(metadata_simplify)
        assert estimate_column_before > MAX_NUMBER_OF_COLUMNS
        assert estimate_column_after <= MAX_NUMBER_OF_COLUMNS
