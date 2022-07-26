"""Integration tests for Single Table Metadata."""

from sdv.metadata import SingleTableMetadata


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
    assert instance._metadata == {
        'columns': {},
        'primary_key': None,
        'alternate_keys': [],
        'sequence_key': None,
        'sequence_index': None,
        'constraints': [],
        'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
    }
