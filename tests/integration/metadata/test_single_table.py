"""Integration tests for Single Table Metadata."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
        'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
    }
    assert instance._columns == {}
    assert instance._constraints == []
    assert instance._version == 'SINGLE_TABLE_V1'
    assert instance._primary_key is None
    assert instance._sequence_key is None
    assert instance._alternate_keys == []
    assert instance._sequence_index is None


def test_validate():
    """Test ``SingleTableMetadata.validate``.

    Ensure the method doesn't crash for a valid metadata.
    """
    # Setup
    instance = SingleTableMetadata()
    instance.add_column('col1', sdtype='numerical')
    instance.add_column('col2', sdtype='numerical')
    instance.add_constraint(
        constraint_name='Inequality',
        low_column_name='col1',
        high_column_name='col2'
    )
    instance.add_constraint(
        constraint_name='ScalarInequality',
        column_name='col1',
        relation='<',
        value=10
    )
    instance.set_primary_key('col1')
    instance.add_alternate_keys([('col1', 'col2')])
    instance.set_sequence_index('col1')
    instance.set_sequence_key('col2')

    # Run
    instance.validate()


def test_validate_errors():
    """Test ``SingleTableMetadata.validate`` raises the correct errors."""
    # Setup
    instance = SingleTableMetadata()
    instance._columns = {
        'col1': {'sdtype': 'numerical'},
        'col2': {'sdtype': 'numerical'},
        'col4': {'sdtype': 'categorical', 'invalid1': 'value'},
        'col5': {'sdtype': 'categorical', 'order': ''},
        'col6': {'sdtype': 'categorical', 'order_by': ''},
        'col7': {'sdtype': 'categorical', 'order': '', 'order_by': ''},
        'col8': {'sdtype': 'numerical', 'computer_representation': 'value'},
        'col9': {'sdtype': 'datetime', 'datetime_format': '%1-%Y-%m-%d-%'},
        'col10': {'sdtype': 'text', 'regex_format': '[A-{6}'},
    }
    instance._constraints = [
        {'constraint_name': 'Inequality', 'low_column_name': 'col1', 'wrong_arg': 'col2'},
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'col1',
            'relation': '<',
            'value': 'string'
        }
    ]
    instance._primary_key = 10
    instance._alternate_keys = 'col1'
    instance._sequence_key = ('col3', 'col1')
    instance._sequence_index = 'col3'

    err_msg = re.escape(
        'The following errors were found in the metadata:'
        "\n\nMissing required values {'high_column_name'} in an Inequality constraint."
        "\nInvalid values {'wrong_arg'} are present in an Inequality constraint."
        "\n'value' must be an int or float."
        "\n'primary_key' must be a string or tuple of strings."
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
        "\nInvalid regex format string '[A-{6}' for text column 'col10'."
    )
    # Run / Assert
    with pytest.raises(InvalidMetadataError, match=err_msg):
        instance.validate()


def test_upgrade_metadata():
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

    # Run
    with TemporaryDirectory() as temp_dir:
        old_path = Path(temp_dir) / 'old.json'
        new_path = Path(temp_dir) / 'new.json'
        old_metadata_file = open(old_path, 'w')
        json.dump(old_metadata, old_metadata_file)
        old_metadata_file.close()
        SingleTableMetadata.upgrade_metadata(old_filepath=old_path, new_filepath=new_path)
        new_metadata_file = open(new_path,)
        new_metadata = json.load(new_metadata_file)
        new_metadata_file.close()

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
                'sdtype': 'numerical'
            },
            'high_perc': {
                'sdtype': 'numerical',
                'computer_representation': 'Float'
            },
            'placed': {
                'sdtype': 'boolean'
            },
            'ssn': {
                'sdtype': 'numerical'
            },
            'drivers_license': {
                'sdtype': 'text',
                'regex_format': 'regex'
            }
        },
        'primary_key': 'student_id',
        'alternate_keys': ['ssn', 'drivers_license'],
        'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
    }
    assert new_metadata == expected_metadata


@patch('sdv.metadata.single_table.warnings')
def test_upgrade_metadata_with_constraints(warnings_mock):
    """Test the ``upgrade_metadata`` method with constraints."""
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
                'type': 'numerical',
                'subtype': 'integer'
            },
            'student_id': {
                'type': 'id',
                'subtype': 'integer'
            },
            'high_perc': {
                'type': 'numerical',
                'subtype': 'float'
            },
            'perc': {
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
            },
            'city': {
                'type': 'categorical'
            },
            'state': {
                'type': 'categorical'
            }
        },
        'primary_key': 'student_id',
        'constraints': [
            {
                'constraint': 'sdv.constraints.tabular.GreaterThan',
                'scalar': None,
                'high': 'end_date',
                'low': 'start_date',
                'strict': False
            },
            {
                'constraint': 'sdv.constraints.tabular.Positive',
                'columns': ['salary', 'duration'],
                'strict': True
            },
            {
                'constraint': 'sdv.constraints.tabular.UniqueCombinations',
                'columns': ['city', 'state']
            },
            {
                'constraint': 'sdv.constraints.tabular.Between',
                'constraint_column': 'perc',
                'high_is_scalar': False,
                'low_is_scalar': True,
                'low': 0,
                'high': 'high_perc'
            }
        ]
    }

    # Run
    with TemporaryDirectory() as temp_dir:
        old_path = Path(temp_dir) / 'old.json'
        new_path = Path(temp_dir) / 'new.json'
        old_metadata_file = open(old_path, 'w')
        json.dump(old_metadata, old_metadata_file)
        old_metadata_file.close()
        SingleTableMetadata.upgrade_metadata(old_filepath=old_path, new_filepath=new_path)
        new_metadata_file = open(new_path,)
        new_metadata = json.load(new_metadata_file)
        new_metadata_file.close()

    # Assert
    expected_constraints = [
        {
            'constraint_name': 'Inequality',
            'high_column_name': 'end_date',
            'low_column_name': 'start_date',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'Positive',
            'column_name': 'salary',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Positive',
            'column_name': 'duration',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'FixedCombinations',
            'column_names': ['city', 'state']
        },
        {
            'constraint_name': 'Inequality',
            'low_column_name': 'perc',
            'high_column_name': 'high_perc',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'perc',
            'relation': '>=',
            'value': 0
        }
    ]

    constraints = new_metadata['constraints']
    assert len(expected_constraints) == len(constraints)
    for constraint in expected_constraints:
        assert constraint in constraints

    warnings_mock.warn.assert_not_called()
