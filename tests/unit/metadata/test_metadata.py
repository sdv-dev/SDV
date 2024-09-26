from unittest.mock import patch

import pytest

from sdv.metadata.metadata import Metadata
from tests.utils import get_multi_table_data, get_multi_table_metadata


class TestMetadataClass:
    """Test ``Metadata`` class."""

    def get_multi_table_metadata(self):
        """Set the tables and relationships for metadata."""
        metadata = {}
        metadata['tables'] = {
            'users': {
                'columns': {'id': {'sdtype': 'id'}, 'country': {'sdtype': 'categorical'}},
                'primary_key': 'id',
            },
            'payments': {
                'columns': {
                    'payment_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'},
                },
                'primary_key': 'payment_id',
            },
            'sessions': {
                'columns': {
                    'session_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'device': {'sdtype': 'categorical'},
                },
                'primary_key': 'session_id',
            },
            'transactions': {
                'columns': {
                    'transaction_id': {'sdtype': 'id'},
                    'session_id': {'sdtype': 'id'},
                    'timestamp': {'sdtype': 'datetime'},
                },
                'primary_key': 'transaction_id',
            },
        }

        metadata['relationships'] = [
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'id',
                'child_table_name': 'sessions',
                'child_foreign_key': 'user_id',
            },
            {
                'parent_table_name': 'sessions',
                'parent_primary_key': 'session_id',
                'child_table_name': 'transactions',
                'child_foreign_key': 'session_id',
            },
            {
                'parent_table_name': 'users',
                'parent_primary_key': 'id',
                'child_table_name': 'payments',
                'child_foreign_key': 'user_id',
            },
        ]

        return Metadata.load_from_dict(metadata)

    @patch('sdv.metadata.utils.Path')
    def test_load_from_json_path_does_not_exist(self, mock_path):
        """Test the ``load_from_json`` method.

        Test that the method raises a ``ValueError`` when the specified path does not
        exist.

        Mock:
            - Mock the ``Path`` library in order to return ``False``, that the file does not exist.

        Input:
            - String representing a filepath.

        Side Effects:
            - A ``ValueError`` is raised pointing that the ``file`` does not exist.
        """
        # Setup
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.name = 'filepath.json'

        # Run / Assert
        error_msg = (
            "A file named 'filepath.json' does not exist. Please specify a different filename."
        )
        with pytest.raises(ValueError, match=error_msg):
            Metadata.load_from_json('filepath.json')

    @patch('sdv.metadata.utils.Path')
    @patch('sdv.metadata.utils.json')
    def test_load_from_json_single_table(self, mock_json, mock_path):
        """Test the ``load_from_json`` method.

        Test that ``load_from_json`` function creates an instance with the contents returned by the
        ``json`` load function when passing in a single table metadata json.

        Mock:
            - Mock the ``Path`` library in order to return ``True``.
            - Mock the ``json`` library in order to use a custom return.

        Input:
            - String representing a filepath.

        Output:
            - ``SingleTableMetadata`` instance with the custom configuration from the ``json``
                file (``json.load`` return value)
        """
        # Setup
        instance = Metadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'
        mock_json.load.return_value = {
            'columns': {'animals': {'type': 'categorical'}},
            'primary_key': 'animals',
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        # Run
        instance = Metadata.load_from_json('filepath.json', 'filepath')

        # Assert
        assert list(instance.tables.keys()) == ['filepath']
        assert instance.tables['filepath'].columns == {'animals': {'type': 'categorical'}}
        assert instance.tables['filepath'].primary_key == 'animals'
        assert instance.tables['filepath'].sequence_key is None
        assert instance.tables['filepath'].alternate_keys == []
        assert instance.tables['filepath'].sequence_index is None
        assert instance.tables['filepath']._version == 'SINGLE_TABLE_V1'

    @patch('sdv.metadata.utils.Path')
    @patch('sdv.metadata.utils.json')
    def test_load_from_json_multi_table(self, mock_json, mock_path):
        """Test the ``load_from_json`` method.

        Test that ``load_from_json`` function creates an instance with the contents returned by the
        ``json`` load function when passing in a multi-table metadata json.

        Mock:
            - Mock the ``Path`` library in order to return ``True``.
            - Mock the ``json`` library in order to use a custom return.

        Input:
            - String representing a filepath.

        Output:
            - ``SingleTableMetadata`` instance with the custom configuration from the ``json``
              file (``json.load`` return value)
        """
        # Setup
        instance = Metadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'
        mock_json.load.return_value = {
            'tables': {
                'table1': {
                    'columns': {'animals': {'type': 'categorical'}},
                    'primary_key': 'animals',
                    'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
                }
            },
            'relationships': {},
        }

        # Run
        instance = Metadata.load_from_json('filepath.json')

        # Asserts
        assert list(instance.tables.keys()) == ['table1']
        assert instance.tables['table1'].columns == {'animals': {'type': 'categorical'}}
        assert instance.tables['table1'].primary_key == 'animals'
        assert instance.tables['table1'].sequence_key is None
        assert instance.tables['table1'].alternate_keys == []
        assert instance.tables['table1'].sequence_index is None
        assert instance.tables['table1']._version == 'SINGLE_TABLE_V1'

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_load_from_dict_multi_table(self, mock_singletablemetadata):
        """Test that ``load_from_dict`` returns a instance of multi-table ``Metadata``.

        Test that when calling the ``load_from_dict`` method a new instance with the passed
        python ``dict`` details should be created.

        Setup:
            - A dict representing a multi-table ``Metadata``.

        Mock:
            - Mock ``SingleTableMetadata`` from ``sdv.metadata.multi_table``

        Output:
            - ``instance`` that contains ``instance.tables`` and ``instance.relationships``.

        Side Effects:
            - ``SingleTableMetadata.load_from_dict`` has been called.
        """
        # Setup
        multitable_metadata = {
            'tables': {
                'accounts': {
                    'id': {'sdtype': 'numerical'},
                    'branch_id': {'sdtype': 'numerical'},
                    'amount': {'sdtype': 'numerical'},
                    'start_date': {'sdtype': 'datetime'},
                    'owner': {'sdtype': 'id'},
                },
                'branches': {
                    'id': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'id'},
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'accounts',
                    'parent_primary_key': 'id',
                    'child_table_name': 'branches',
                    'child_foreign_key': 'branch_id',
                }
            ],
        }

        single_table_accounts = object()
        single_table_branches = object()
        mock_singletablemetadata.load_from_dict.side_effect = [
            single_table_accounts,
            single_table_branches,
        ]

        # Run
        instance = Metadata.load_from_dict(multitable_metadata)

        # Assert
        assert instance.tables == {
            'accounts': single_table_accounts,
            'branches': single_table_branches,
        }

        assert instance.relationships == [
            {
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'child_foreign_key': 'branch_id',
            }
        ]

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_load_from_dict_integer_multi_table(self, mock_singletablemetadata):
        """Test that ``load_from_dict`` returns a instance of multi-table ``Metadata``.

        Test that when calling the ``load_from_dict`` method a new instance with the passed
        python ``dict`` details should be created. Make sure that integers passed in are
        turned into strings to ensure metadata is properly typed.

        Setup:
            - A dict representing a multi-table ``Metadata``.

        Mock:
            - Mock ``SingleTableMetadata`` from ``sdv.metadata.multi_table``

        Output:
            - ``instance`` that contains ``instance.tables`` and ``instance.relationships``.

        Side Effects:
            - ``SingleTableMetadata.load_from_dict`` has been called.
        """
        # Setup
        multitable_metadata = {
            'tables': {
                'accounts': {
                    1: {'sdtype': 'numerical'},
                    2: {'sdtype': 'numerical'},
                    'amount': {'sdtype': 'numerical'},
                    'start_date': {'sdtype': 'datetime'},
                    'owner': {'sdtype': 'id'},
                },
                'branches': {
                    1: {'sdtype': 'numerical'},
                    'name': {'sdtype': 'id'},
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'accounts',
                    'parent_primary_key': 1,
                    'child_table_name': 'branches',
                    'child_foreign_key': 1,
                }
            ],
        }

        single_table_accounts = {
            '1': {'sdtype': 'numerical'},
            '2': {'sdtype': 'numerical'},
            'amount': {'sdtype': 'numerical'},
            'start_date': {'sdtype': 'datetime'},
            'owner': {'sdtype': 'id'},
        }
        single_table_branches = {
            '1': {'sdtype': 'numerical'},
            'name': {'sdtype': 'id'},
        }
        mock_singletablemetadata.load_from_dict.side_effect = [
            single_table_accounts,
            single_table_branches,
        ]

        # Run
        instance = Metadata.load_from_dict(multitable_metadata)

        # Assert
        assert instance.tables == {
            'accounts': single_table_accounts,
            'branches': single_table_branches,
        }

        assert instance.relationships == [
            {
                'parent_table_name': 'accounts',
                'parent_primary_key': '1',
                'child_table_name': 'branches',
                'child_foreign_key': '1',
            }
        ]

    def test_load_from_dict_single_table(self):
        """Test that ``load_from_dict`` returns a instance of single-table ``Metadata``.

        Test that when calling the ``load_from_dict`` method a new instance with the passed
        python ``dict`` details should be created.
        """
        # Setup
        my_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        # Run
        instance = Metadata.load_from_dict(my_metadata)

        # Assert
        assert list(instance.tables.keys()) == ['default_table_name']
        assert instance.tables['default_table_name'].columns == {'my_column': 'value'}
        assert instance.tables['default_table_name'].primary_key == 'pk'
        assert instance.tables['default_table_name'].sequence_key is None
        assert instance.tables['default_table_name'].alternate_keys == []
        assert instance.tables['default_table_name'].sequence_index is None
        assert instance.tables['default_table_name']._version == 'SINGLE_TABLE_V1'

    def test_load_from_dict_integer_single_table(self):
        """Test that ``load_from_dict`` returns a instance of single-table ``Metadata``.

        Test that when calling the ``load_from_dict`` method a new instance with the passed
        python ``dict`` details should be created. Make sure that integers passed in are
        turned into strings to ensure metadata is properly typed.
        """

        # Setup
        my_metadata = {
            'columns': {1: 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        # Run
        instance = Metadata.load_from_dict(my_metadata)

        # Assert
        assert list(instance.tables.keys()) == ['default_table_name']
        assert instance.tables['default_table_name'].columns == {'1': 'value'}
        assert instance.tables['default_table_name'].primary_key == 'pk'
        assert instance.tables['default_table_name'].sequence_key is None
        assert instance.tables['default_table_name'].alternate_keys == []
        assert instance.tables['default_table_name'].sequence_index is None

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test__set_metadata_multi_table(self, mock_singletablemetadata):
        """Test the ``_set_metadata`` method for ``Metadata``.

        Setup:
            - instance of ``Metadata``.
            - A dict representing a ``Metadata``.

        Mock:
            - Mock ``SingleTableMetadata`` from ``sdv.metadata.multi_table``

        Side Effects:
            - ``instance`` now contains ``instance.tables`` and ``instance.relationships``.
            - ``SingleTableMetadata.load_from_dict`` has been called.
        """
        # Setup
        multitable_metadata = {
            'tables': {
                'accounts': {
                    'id': {'sdtype': 'numerical'},
                    'branch_id': {'sdtype': 'numerical'},
                    'amount': {'sdtype': 'numerical'},
                    'start_date': {'sdtype': 'datetime'},
                    'owner': {'sdtype': 'id'},
                },
                'branches': {
                    'id': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'id'},
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'accounts',
                    'parent_primary_key': 'id',
                    'child_table_name': 'branches',
                    'chil_foreign_key': 'branch_id',
                }
            ],
        }

        single_table_accounts = object()
        single_table_branches = object()
        mock_singletablemetadata.load_from_dict.side_effect = [
            single_table_accounts,
            single_table_branches,
        ]

        instance = Metadata()

        # Run
        instance._set_metadata_dict(multitable_metadata)

        # Assert
        assert instance.tables == {
            'accounts': single_table_accounts,
            'branches': single_table_branches,
        }

        assert instance.relationships == [
            {
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'chil_foreign_key': 'branch_id',
            }
        ]

    def test__set_metadata_single_table(self):
        """Test the ``_set_metadata`` method for ``Metadata``.

        Setup:
            - instance of ``Metadata``.
            - A dict representing a single table``Metadata``.
        """
        # Setup
        single_table_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        instance = Metadata()

        # Run
        instance._set_metadata_dict(single_table_metadata)

        # Assert
        assert instance.tables['default_table_name'].columns == {'my_column': 'value'}
        assert instance.tables['default_table_name'].primary_key == 'pk'
        assert instance.tables['default_table_name'].alternate_keys == []
        assert instance.tables['default_table_name'].sequence_key is None
        assert instance.tables['default_table_name'].sequence_index is None
        assert instance.tables['default_table_name'].METADATA_SPEC_VERSION == 'SINGLE_TABLE_V1'

    def test_validate(self):
        """Test the method ``validate``.

        Test that when a valid ``Metadata`` has been provided no errors are being raised.

        Setup:
            - Instance of ``Metadata`` with all valid tables and relationships.
        """
        # Setup
        instance = self.get_multi_table_metadata()

        # Run
        instance.validate()

    def test_validate_no_relationships(self):
        """Test the method ``validate`` without relationships.

        Test that when a valid ``Metadata`` has been provided no errors are being raised.

        Setup:
            - Instance of ``Metadata`` with all valid tables and no relationships.
        """
        # Setup
        metadata = self.get_multi_table_metadata()
        metadata_no_relationships = metadata.to_dict()
        del metadata_no_relationships['relationships']
        test_metadata = Metadata.load_from_dict(metadata_no_relationships)

        # Run
        test_metadata.validate()
        assert test_metadata.METADATA_SPEC_VERSION == 'V1'

    def test_validate_data(self):
        """Test that no error is being raised when the data is valid."""
        # Setup
        metadata_dict = get_multi_table_metadata().to_dict()
        metadata = Metadata.load_from_dict(metadata_dict)
        data = get_multi_table_data()

        # Run and Assert
        metadata.validate_data(data)
        assert metadata.METADATA_SPEC_VERSION == 'V1'

    def test_validate_data_no_relationships(self):
        """Test that no error is being raised when the data is valid but has no relationships."""
        # Setup
        metadata_dict = get_multi_table_metadata().to_dict()
        del metadata_dict['relationships']
        del metadata_dict['METADATA_SPEC_VERSION']
        metadata = Metadata.load_from_dict(metadata_dict)
        data = get_multi_table_data()

        # Run and Assert
        metadata.validate_data(data)
        assert metadata.METADATA_SPEC_VERSION == 'V1'
