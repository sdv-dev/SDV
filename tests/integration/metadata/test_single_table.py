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
    assert instance._primary_key is None
    assert instance._alternate_keys == []
    assert instance._constraints == []
    assert instance._version == 'SINGLE_TABLE_V1'
    assert instance._metadata == {
        'columns': {},
        'primary_key': None,
        'alternate_keys': [],
        'constraints': [],
        'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
    }
