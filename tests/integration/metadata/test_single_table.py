"""Integration tests for Single Table Metadata."""
import json
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata
from sdv.metadata.errors import InvalidMetadataError


def test_single_table_metadata():
    """Test ``SingleTableMetadata``."""

    # Create an instance
    instance = SingleTableMetadata()

    # To dict
    result = instance.to_dict()

    # Assert
    assert result == {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
    }
    assert instance.columns == {}
    assert instance._version == 'SINGLE_TABLE_V1'
    assert instance.primary_key is None
    assert instance.sequence_key is None
    assert instance.alternate_keys == []
    assert instance.sequence_index is None


@patch('rdt.transformers')
def test_add_column_relationship(mock_rdt_transformers):
    """Test ``add_column_relationship`` method."""
    # Setup
    class RandomLocationGeneratorMock:
        @classmethod
        def _validate_sdtypes(cls, columns_to_sdtypes):
            pass

    mock_rdt_transformers.address.RandomLocationGenerator = RandomLocationGeneratorMock
    instance = SingleTableMetadata()
    instance.add_column('col1', sdtype='id')
    instance.add_column('col2', sdtype='street_address')
    instance.add_column('col3', sdtype='state_abbr')
    instance.set_primary_key('col1')

    # Run
    instance.add_column_relationship(relationship_type='address', column_names=['col2', 'col3'])

    # Assert
    instance.validate()
    assert instance.column_relationships == [
        {'type': 'address', 'column_names': ['col2', 'col3']}
    ]


@patch('rdt.transformers')
def test_validate(mock_rdt_transformers):
    """Test ``SingleTableMetadata.validate``.

    Ensure the method doesn't crash for a valid metadata.
    """
    # Setup
    class RandomLocationGeneratorMock:
        @classmethod
        def _validate_sdtypes(cls, columns_to_sdtypes):
            pass

    mock_rdt_transformers.address.RandomLocationGenerator = RandomLocationGeneratorMock
    instance = SingleTableMetadata()
    instance.add_column('col1', sdtype='id')
    instance.add_column('col2', sdtype='id')
    instance.add_column('col3', sdtype='numerical')
    instance.add_column('col4', sdtype='street_address')
    instance.add_column('col5', sdtype='state_abbr')
    instance.set_primary_key('col1')
    instance.add_alternate_keys(['col2'])
    instance.set_sequence_index('col3')
    instance.set_sequence_key('col2')
    instance.add_column_relationship(relationship_type='address', column_names=['col4', 'col5'])

    # Run
    instance.validate()


@patch('rdt.transformers')
def test_validate_errors(mock_rdt_transformers):
    """Test ``SingleTableMetadata.validate`` raises the correct errors."""
    # Setup
    class RandomLocationGeneratorMock:
        @classmethod
        def _validate_sdtypes(cls, columns_to_sdtypes):
            valid_sdtypes = (
                'country_code', 'administrative_unit', 'city', 'postcode', 'street_address',
                'secondary_address', 'state', 'state_abbr'
            )
            bad_columns = []
            for column_name, sdtypes in columns_to_sdtypes.items():
                if sdtypes not in valid_sdtypes:
                    bad_columns.append(column_name)

            if bad_columns:
                raise InvalidMetadataError(
                    f'Columns {bad_columns} have unsupported sdtypes for column relationship '
                    "type 'address'."
                )

    mock_rdt_transformers.address.RandomLocationGenerator = RandomLocationGeneratorMock
    instance = SingleTableMetadata()
    instance.columns = {
        'col1': {'sdtype': 'id'},
        'col2': {'sdtype': 'numerical'},
        'col4': {'sdtype': 'categorical', 'invalid1': 'value'},
        'col5': {'sdtype': 'categorical', 'order': ''},
        'col6': {'sdtype': 'categorical', 'order_by': ''},
        'col7': {'sdtype': 'categorical', 'order': '', 'order_by': ''},
        'col8': {'sdtype': 'numerical', 'computer_representation': 'value'},
        'col9': {'sdtype': 'datetime', 'datetime_format': '%1-%Y-%m-%d-%'},
        'col10': {'sdtype': 'id', 'regex_format': '[A-{6}'},
        'col11': {'sdtype': 'state'},
    }
    instance.primary_key = 10
    instance.alternate_keys = 'col1'
    instance.sequence_key = ('col3', 'col1')
    instance.sequence_index = 'col3'
    instance.column_relationships = [
        {'type': 'address', 'column_names': ['col11']},
        {'type': 'address', 'column_names': ['col1', 'col2']},
        {'type': 'fake_relationship', 'column_names': ['col3', 'col4']}
    ]

    err_msg = re.escape(
        'The following errors were found in the metadata:'
        "\n\n'primary_key' must be a string."
        "\n'sequence_key' must be a string."
        "\nUnknown sequence index value {'col3'}. Keys should be columns that exist in the table."
        "\n'sequence_index' and 'sequence_key' have the same value {'col3'}."
        ' These columns must be different.'
        "\n'alternate_keys' must be a list of strings."
        "\nInvalid values '(invalid1)' for categorical column 'col4'."
        "\nInvalid order value provided for categorical column 'col5'."
        " The 'order' must be a list with 1 or more elements."
        "\nUnknown ordering method '' provided for categorical column 'col6'."
        " Ordering method must be 'numerical_value' or 'alphabetical'."
        "\nCategorical column 'col7' has both an 'order' and 'order_by' attribute."
        ' Only 1 is allowed.'
        "\nInvalid value for 'computer_representation' 'value' for column 'col8'."
        "\nInvalid datetime format string '%1-%Y-%m-%d-%' for datetime column 'col9'."
        "\nInvalid regex format string '[A-{6}' for id column 'col10'."
        "\nColumn relationships have following errors:\nColumns ['col1', 'col2'] have "
        "unsupported sdtypes for column relationship type 'address'.\nUnknown column "
        "relationship type 'fake_relationship'. Must be one of ['address', 'gps']."
    )
    # Run / Assert
    with pytest.raises(InvalidMetadataError, match=err_msg):
        instance.validate()


def test_upgrade_metadata(tmp_path):
    """Test the ``upgrade_metadata`` method."""
    # Setup
    old_metadata = {
        'fields': {
            'start_date': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
            'end_date': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
            'salary': {
                'type': 'numerical',
                'subtype': 'integer'
            },
            'duration': {
                'type': 'categorical'
            },
            'student_id': {
                'type': 'id',
                'subtype': 'integer'
            },
            'high_perc': {
                'type': 'numerical',
                'subtype': 'float'
            },
            'placed': {
                'type': 'boolean'
            },
            'ssn': {
                'type': 'id',
                'subtype': 'integer'
            },
            'drivers_license': {
                'type': 'id',
                'subtype': 'string',
                'regex': 'regex'
            }
        },
        'primary_key': 'student_id'
    }
    filepath = tmp_path / 'old.json'
    old_metadata_file = open(filepath, 'w')
    json.dump(old_metadata, old_metadata_file)
    old_metadata_file.close()

    # Run
    new_metadata = SingleTableMetadata.upgrade_metadata(filepath=filepath).to_dict()

    # Assert
    expected_metadata = {
        'columns': {
            'start_date': {
                'sdtype': 'datetime',
                'datetime_format': '%Y-%m-%d'
            },
            'end_date': {
                'sdtype': 'datetime',
                'datetime_format': '%Y-%m-%d'
            },
            'salary': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            },
            'duration': {
                'sdtype': 'categorical'
            },
            'student_id': {
                'sdtype': 'id',
                'regex_format': r'\d{30}'
            },
            'high_perc': {
                'sdtype': 'numerical',
                'computer_representation': 'Float'
            },
            'placed': {
                'sdtype': 'boolean'
            },
            'ssn': {
                'sdtype': 'id',
                'regex_format': r'\d{30}'
            },
            'drivers_license': {
                'sdtype': 'id',
                'regex_format': 'regex'
            }
        },
        'primary_key': 'student_id',
        'alternate_keys': ['ssn', 'drivers_license'],
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
    }
    assert new_metadata == expected_metadata


def test_validate_unknown_sdtype():
    """Test ``validate`` method works with ``unknown`` sdtype."""
    # Setup
    data, _ = download_demo(
        modality='multi_table',
        dataset_name='fake_hotels'
    )

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data['hotels'])

    # Run
    metadata.validate()

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': {
            'hotel_id': {'sdtype': 'id'},
            'city': {'sdtype': 'city', 'pii': True},
            'state': {'sdtype': 'administrative_unit', 'pii': True},
            'rating': {'sdtype': 'numerical'},
            'classification': {'sdtype': 'unknown', 'pii': True}
        },
        'primary_key': 'hotel_id'
    }
    assert metadata.to_dict() == expected_metadata


def test_detect_from_dataframe_with_none_nan_and_nat():
    """Test ``detect_from_dataframe`` with ``None``, ``np.nan`` and ``pd.NaT``."""
    # Setup
    data = pd.DataFrame({
        'pk2': list(range(100)),
        'pk1': list(range(100)),
        'f_nan_data': [float('nan')] * 100,
        'none_data': [None] * 100,
        'np_nan_data': [np.nan] * 100,
        'pd_nat_data': [pd.NaT] * 100
    })
    stm = SingleTableMetadata()

    # Run
    stm.detect_from_dataframe(data)

    # Assert
    assert stm.columns['f_nan_data']['sdtype'] == 'numerical'
    assert stm.columns['none_data']['sdtype'] == 'categorical'
    assert stm.columns['np_nan_data']['sdtype'] == 'numerical'
    assert stm.columns['pd_nat_data']['sdtype'] == 'datetime'


def test_detect_from_dataframe_with_pii_names():
    """Test ``detect_from_dataframe`` with pii column names."""
    # Setup
    data = pd.DataFrame({
        'USER PHONE NUMBER': [1, 2, 3],
        'addr_line_1': [1, 2, 3],
        'First Name': [1, 2, 3],
        'guest_email': [1, 2, 3],

    })
    metadata = SingleTableMetadata()

    # Run
    metadata.detect_from_dataframe(data)

    # Assert
    expected_metadata = {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'primary_key': 'USER PHONE NUMBER',
        'columns': {
            'USER PHONE NUMBER': {'sdtype': 'phone_number', 'pii': True},
            'addr_line_1': {'sdtype': 'street_address', 'pii': True},
            'First Name': {'sdtype': 'first_name', 'pii': True},
            'guest_email': {'sdtype': 'email', 'pii': True}
        }
    }

    assert metadata.to_dict() == expected_metadata
