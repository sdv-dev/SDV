"""Integration tests for Multi Table Metadata."""

from sdv.metadata import MultiTableMetadata


def test_single_table_metadata():
    """Test ``MultiTableMetadata``."""

    # Create an instance
    instance = MultiTableMetadata()

    # To dict
    result = instance.to_dict()

    # Assert
    assert result == {
        'tables': {},
        'relationships': []
    }
    assert instance._tables == {}
    assert instance._relationships == []
