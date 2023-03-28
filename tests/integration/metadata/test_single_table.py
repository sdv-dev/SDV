"""Integration tests for Single Table Metadata."""

import json
import re

import pytest

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


def test_validate():
    """Test ``SingleTableMetadata.validate``.

    Ensure the method doesn't crash for a valid metadata.
    """
    # Setup
    instance = SingleTableMetadata()
    instance.add_column('col1', sdtype='id')
    instance.add_column('col2', sdtype='id')
    instance.add_column('col3', sdtype='numerical')
    instance.set_primary_key('col1')
    instance.add_alternate_keys([('col1', 'col2')])
    instance.set_sequence_index('col3')
    instance.set_sequence_key('col2')

    # Run
    instance.validate()


def test_validate_errors():
    """Test ``SingleTableMetadata.validate`` raises the correct errors."""
    # Setup
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
    }
    instance.primary_key = 10
    instance.alternate_keys = 'col1'
    instance.sequence_key = ('col3', 'col1')
    instance.sequence_index = 'col3'

    err_msg = re.escape(
        'The following errors were found in the metadata:'
        "\n\n'primary_key' must be a string or tuple of strings."
        "\nUnknown sequence key values {'col3'}. Keys should be columns that exist in the table."
        "\nUnknown sequence index value {'col3'}. Keys should be columns that exist in the table."
        "\n'sequence_index' and 'sequence_key' have the same value {'col3'}."
        ' These columns must be different.'
        "\n'alternate_keys' must be a list of strings or a list of tuples of strings."
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
