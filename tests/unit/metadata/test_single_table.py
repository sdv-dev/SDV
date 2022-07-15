"""Test Single Table Metadata."""

from unittest.mock import patch
from sdv.metadata.single_table import SingleTableMetadata


class TestSingleTableMetadata:
    """Test ``SingleTableMetadata`` class."""

    def test___init__(self):
        """Test creating an instance of ``SingleTableMetadata``."""
        # Run
        instance = SingleTableMetadata()

        # Assert
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

    def test_to_dict(self):
        """Test the ``to_dict`` method from ``SingleTableMetadata``.

        Setup:
            - Instance of ``SingleTableMetadata`` and modify the ``instance._columns`` to ensure
            that ``to_dict`` works properly.
        Output:
            - A dictionary representation of the ``instance`` that does not modify the
              internal dictionaries.
        """
        # Setup
        instance = SingleTableMetadata()
        instance._columns['my_column'] = 'value'

        # Run
        result = instance.to_dict()

        # Assert
        assert result == {
            'columns': {'my_column': 'value'},
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        result['columns']['my_column'] = 1
        assert instance._columns['my_column'] == 'value'

    def test__set_metadata_dict(self):
        """Test the ``_set_metadata_dict`` to a instance.

        Setup:
            - Instance of ``SingleTableMetadata``.
            - Dictionary representing ``SingleTableMetadata``.

        Output:
            - ``SingleTableMetadata`` instance with the dictionary represented values.
        """
        # Setup
        instance = SingleTableMetadata()
        metadata = {
            'columns': {'my_column': 'value'},
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance._set_metadata_dict(metadata)

        # Assert
        assert instance._metadata == {
            'columns': {'my_column': 'value'},
            'primary_key': None,
            'alternate_keys': [],
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        assert instance._columns == {'my_column': 'value'}

    def test__load_from_dict(self):
        """Test that ``_load_from_dict`` returns a instance with the ``dict`` updated objects."""
        # Setup
        my_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'constraints': [],
            'SCHEMA_VERSION': 'SINGLE_TABLE_V1'
        }

        # Run
        instance = SingleTableMetadata._load_from_dict(my_metadata)

        # Assert
        assert instance._metadata == my_metadata
        assert instance._columns == {'my_column': 'value'}
        assert instance._primary_key == 'pk'
        assert instance._alternate_keys == []
        assert instance._constraints == []

    @patch('sdv.metadata.single_table.json')
    def test___repr__(self, mock_json):
        """Test that the ``__repr__`` method.

        Test that the ``__repr__`` method calls the ``json.dumps``  method and
        returns its output.

        Setup:
            - Instance of ``SingleTableMetadata``.
        Mock:
            - ``json`` from ``sdv.metadata.single_table``.

        Output:
            - ``json.dumps`` return value.
        """
        # Setup
        instance = SingleTableMetadata()

        # Run
        res = instance.__repr__()

        # Assert
        mock_json.dumps.assert_called_once_with(instance.to_dict(), indent=4)
        assert res == mock_json.dumps.return_value
