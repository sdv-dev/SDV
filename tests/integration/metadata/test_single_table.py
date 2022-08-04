"""Integration tests for Single Table Metadata."""

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
    instance.set_alternate_keys([('col1', 'col2')])
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
        'col8': {'sdtype': 'numerical', 'representation': 'value'},
        'col9': {'sdtype': 'datetime', 'datetime_format': '%1-%Y-%m-%d-%'},
        'col10': {'sdtype': 'text', 'regex_format': '[A-{6}'},
    }
    instance._constraints = [
        ('Inequality', {'low_column_name': 'col1', 'wrong_arg': 'col2'}),
        ('ScalarInequality', {'column_name': 'col1', 'relation': '<', 'value': 'string'})
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
        "\n'alternate_keys' must be a list of strings or a list of tuples of strings."
        "\nUnknown sequence index value {'col3'}. Keys should be columns that exist in the table."
        "\n'sequence_index' and 'sequence_key' have the same value {'col3'}."
        ' These columns must be different.'
        "\nInvalid values '(invalid1)' for categorical column 'col4'."
        "\nInvalid order value provided for categorical column 'col5'."
        " The 'order' must be a list with 1 or more elements."
        "\nUnknown ordering method '' provided for categorical column 'col6'."
        " Ordering method must be 'numerical_value' or 'alphabetical'."
        "\nCategorical column 'col7' has both an 'order' and 'order_by' attribute."
        ' Only 1 is allowed.'
        "\nInvalid value for 'representation' 'value' for column 'col8'."
        "\nInvalid datetime format string '%1-%Y-%m-%d-%' for datetime column 'col9'."
        "\nInvalid regex format string '[A-{6}' for text column 'col10'."
    )
    # Run / Assert
    with pytest.raises(InvalidMetadataError, match=err_msg):
        instance.validate()
