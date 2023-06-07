"""Test Multi Table Metadata."""

import json
import re
from collections import defaultdict
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.multi_table import MultiTableMetadata, SingleTableMetadata


class TestMultiTableMetadata:
    """Test ``MultiTableMetadata`` class."""

    def get_metadata(self):
        """Set the tables and relationships for metadata."""
        metadata = {}
        metadata['tables'] = {
            'users': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'country': {'sdtype': 'categorical'}
                },
                'primary_key': 'id'
            },
            'payments': {
                'columns': {
                    'payment_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'date': {'sdtype': 'datetime'}
                },
                'primary_key': 'payment_id'
            },
            'sessions': {
                'columns': {
                    'session_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'device': {'sdtype': 'categorical'}
                },
                'primary_key': 'session_id'
            },
            'transactions': {
                'columns': {
                    'transaction_id': {'sdtype': 'id'},
                    'session_id': {'sdtype': 'id'},
                    'timestamp': {'sdtype': 'datetime'}
                },
                'primary_key': 'transaction_id'
            }
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
            }
        ]

        return MultiTableMetadata.load_from_dict(metadata)

    def test___init__(self):
        """Test the ``__init__`` method of ``MultiTableMetadata``."""
        # Run
        instance = MultiTableMetadata()

        # Assert
        assert instance.tables == {}
        assert instance.relationships == []

    def test__validate_missing_relationship_keys_foreign_key(self):
        """Test the ``_validate_missing_relationship_keys`` method of ``MultiTableMetadata``.


        Setup:
            - Mock ``parent_table`` and ``child_table``.
            - Instance of ``MultiTableMetadata``.
            - Store the input in variables.

        Mock:
            - Mock instance of ``MultiTableMetadata``.
            - ``SingleTableMetadata`` instance that represents the ``parent_table``.

        Input:
            - ``parent_table`` that represents ``SingleTableMetadata``.
            - ``parent_table_name`` string.
            - ``parent_primary_key`` a string that is not the parent primary key.
            - ``child_table_name`` string.
            - ``child_foreign_key`` string.

        Side Effects:
            - Raises ``InvalidMetadataError`` stating that foreign key is unknown.
        """
        # Setup
        parent_table = Mock()
        parent_table.primary_key = 'id'
        parent_table.columns = {
            'id': {'sdtype': 'numerical'},
            'session': {'sdtype': 'numerical'},
            'transactions': {'sdtype': 'numerical'},
        }
        parent_table_name = 'users'
        parent_primary_key = 'id'

        child_table = Mock()
        child_table.primary_key = 'session_id'
        child_table.columns = {
            'user_id': {'sdtype': 'numerical'},
            'session_id': {'sdtype': 'numerical'},
            'transactions': {'sdtype': 'numerical'},
        }
        child_table_name = 'sessions'
        child_foreign_key = 'id'

        instance = Mock()
        instance.tables = {
            'users': parent_table,
            'sessions': child_table,
        }

        # Run / Assert
        error_msg = re.escape(
            'Relationship between tables (users, sessions) contains '
            "an unknown foreign key {'id'}."
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            MultiTableMetadata._validate_missing_relationship_keys(
                instance,
                parent_table_name,
                parent_primary_key,
                child_table_name,
                child_foreign_key
            )

    def test__validate_missing_relationship_keys_primary_key(self):
        """Test the ``_validate_missing_relationship_keys`` method of ``MultiTableMetadata``.

        Test that when the provided ``child_foreign_key`` key is not in the
        ``parent_table.columns``, this raises an error.

        Setup:
            - Create ``parent_table``.
            - Store the input in variables.

        Mock:
            - ``SingleTableMetadata`` instance that represents the ``parent_table``.

        Input:
            - ``parent_table`` that represents ``SingleTableMetadata``.
            - ``parent_table_name`` string.
            - ``parent_primary_key`` a string that is the parent primary key.
            - ``child_table_name`` string.
            - ``child_foreign_key`` a string that is not in the ``parent_table.columns``.

        Side Effects:
            - Raises ``InvalidMetadataError`` stating that primary key is unknown.
        """
        # Setup
        parent_table = Mock()
        parent_table.primary_key = 'users_id'
        parent_table_name = 'users'
        parent_primary_key = 'primary_key'
        child_table_name = 'sessions'
        child_foreign_key = 'session_id'

        # Run / Assert
        error_msg = re.escape(
            'Relationship between tables (users, sessions) contains '
            "an unknown primary key {'primary_key'}."
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            MultiTableMetadata._validate_missing_relationship_keys(
                parent_table,
                parent_table_name,
                parent_primary_key,
                child_table_name,
                child_foreign_key
            )

    def test__validate_no_missing_tables_in_relationship(self):
        """Test the ``_validate_no_missing_tables_in_relationship`` method of ``MultiTableMetadata``.

        Setup:
            - Create a list of ``tables``.

        Input:
            - ``parent_table_name`` string.
            - ``child_table_name`` string that is not inside tables.
            - ``tables`` list of table names.
        """
        # Setup
        tables = ['users', 'sessions', 'transactions']

        # Run
        error_msg = re.escape("Relationship contains an unknown table {'session'}.")
        with pytest.raises(InvalidMetadataError, match=error_msg):
            MultiTableMetadata._validate_no_missing_tables_in_relationship(
                'users',
                'session',
                tables
            )

    def test__validate_missing_relationship_key_length(self):
        """Test the ``_validate_missing_relationship_key_length`` method of ``MultiTableMetadata``.

        Test that the length of the primary key and foreign key are the same, and raise an error
        when those are different.

        Input:
            - ``parent_table_name`` string.
            - ``child_table_name`` string that is not inside tables.
            - ``parent_primary_key`` list of keys.
            - ``child_foreign_key`` string representing one foreign key.
        """
        # Setup
        parent_table_name = 'users'
        parent_primary_key = ['users_id', 'users_name']
        child_table_name = 'sessions'
        child_foreign_key = 'session_id'

        # Run / Assert
        error_msg = re.escape(
            "Relationship between tables ('users', 'sessions') is invalid. "
            'Primary key has length 2 but the foreign key has length 1.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            MultiTableMetadata._validate_relationship_key_length(
                parent_table_name,
                parent_primary_key,
                child_table_name,
                child_foreign_key
            )

    def test__validate_relationship_sdtype(self):
        """Test the ``_validate_relationship_sdtype`` method of ``MultiTableMetadata``.

        Validate that when a list of primary keys and foreign keys is passed and the ``sdtype``
        of those do not match, a value error is being raised.

        Setup:
            - Create ``parent_table`` and update it's ``_columns`` to contain primary keys and
              sdtypes.
            - Create ``child_table`` and update it's ``_columns`` to contain primary keys and
              foreign keys with their sdtypes.
            - Create all the input values.

        Input:
            - ``parent_table``, a mock representing a ``SingleTableMetadata`` instance with
              ``_columns``.
            - ``parent_primary_key`` a list representing the parent primary keys.
            - ``child_foreign_key`` a list representing the foreign keys.
            - ``child_table_name`` a string representing the name of the child table.
            - ``parent_table_name`` a string representing the name of the parent table.

        Mock:
            - ``parent_table`` to match the ``SingleTableMetadata`` description.

        Side Effcts:
            - An ``InvalidMetadataError`` is being raised because the ``sdtype`` on one of the keys
              does not match.
        """
        # Setup
        parent_table = Mock()
        parent_table.primary_key = 'id'
        parent_table.columns = {
            'id': {'sdtype': 'numerical'},
            'user_name': {'sdtype': 'categorical'},
            'transactions': {'sdtype': 'numerical'},
        }
        parent_table_name = 'users'
        parent_primary_key = ['id', 'user_name']

        child_table = Mock()
        child_table.primary_key = 'session_id'
        child_table.columns = {
            'user_id': {'sdtype': 'numerical'},
            'session_id': {'sdtype': 'numerical'},
            'timestamp': {'sdtype': 'datetime'},
        }
        child_table_name = 'sessions'
        child_foreign_key = ['user_id', 'timestamp']

        instance = Mock()
        instance.tables = {
            'users': parent_table,
            'sessions': child_table,
        }

        # Run / Assert
        error_msg = re.escape(
            "Relationship between tables ('users', 'sessions') is invalid. "
            'The primary and foreign key columns are not the same type.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            MultiTableMetadata._validate_relationship_sdtypes(
                instance,
                parent_table_name,
                parent_primary_key,
                child_table_name,
                child_foreign_key
            )

    def test__validate_relationship_does_not_exist(self):
        """Test the method raises an error if an existing relationship is added."""
        # Setup
        metadata = MultiTableMetadata()
        metadata.relationships = [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions',
                'parent_primary_key': 'id',
                'child_foreign_key': 'user_id',
            },
            {
                'parent_table_name': 'sessions',
                'child_table_name': 'transactions',
                'parent_primary_key': 'id',
                'child_foreign_key': 'session_id',
            }
        ]

        # Run and Assert
        error_msg = 'This relationship has already been added.'
        with pytest.raises(InvalidMetadataError, match=error_msg):
            metadata._validate_relationship_does_not_exist(
                parent_table_name='sessions',
                parent_primary_key='id',
                child_table_name='transactions',
                child_foreign_key='session_id'
            )

    def test__validate_circular_relationships(self):
        """Test the ``_validate_circular_relationships`` method of ``MultiTableMetadata``.

        Validate that the ``_validate_circular_relationships`` updates the ``errors`` list
        with the detected circular relationship.

        Setup:
            - ``child_map``, mapping of parent tables with their child tables.

        Input:
            - ``parent`` the name of the table that we want to add a relationship for.
            - ``child_map``, the mapping created previously
            - ``errors``, the list of errors that has to be updated.

        Side Effects:
            - The input list has been updated with the tables detected to cause circular
              relationship.
        """
        # Setup
        child_map = {
            'users': {'sessions', 'transactions'},
            'sessions': {'users', 'transactions'}
        }
        parent = 'users'
        errors = []

        # Run
        MultiTableMetadata()._validate_circular_relationships(
            parent, child_map=child_map, errors=errors)

        # Assert
        assert errors == ['users']

    def test__validate_child_map_circular_relationship(self):
        """Test the ``_validate_child_map_circular_relationship`` method of ``MultiTableMetadata``.

        Test that when a circular relationship occurs an ``InvalidMetadataError`` is being raised.

        Setup:
            - Instance of ``MultiTableMetadata``.
            - Mock of ``parent_table`` simulating a ``SingleTableMetadata``.
            - Update ``_tables``.

        Input:
            - ``parent_table_name`` string representing the parent table name.
            - ``child_table_name`` string representing the child table name.
            - ``parent_primary_key`` string representing the ``primary_key`` of the parent.
            - ``child_foreign_key`` string representing the ``foreing_Key`` of the child table.

        Side Effects:
            - ``InvalidMetadataError`` is raised.
        """
        # Setup
        instance = MultiTableMetadata()
        parent_table = Mock()
        instance.tables = {
            'users': parent_table,
            'sessions': Mock(),
            'transactions': Mock()
        }
        child_map = {
            'users': {'sessions', 'transactions'},
            'sessions': {'users'},
            'transactions': set()
        }

        # Run / Assert
        err_msg = re.escape(
            'The relationships in the dataset describe a circular dependency between tables '
            "['users', 'sessions']."
        )
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance._validate_child_map_circular_relationship(child_map)

    @patch(
        'sdv.metadata.multi_table.MultiTableMetadata._validate_no_missing_tables_in_relationship'
    )
    @patch('sdv.metadata.multi_table.MultiTableMetadata._validate_relationship_key_length')
    def test__validate_relationship(self,
                                    mock_validate_relationship_key_length,
                                    mock_validate_no_missing_tables_in_relationship):
        """Test thath the ``_validate_relationship`` method.

        Test that when calling the ``_validate_relationship`` method, the other validation methods
        for relationship are being called with the input values from this.

        Setup:
            - Instance of ``MultiTableMetadata``.
            - Update with ``tables`` and mock a ``parent_table``.

        Mock:
            - Mock all validation methods.

        Side Effects:
            - All the validation methods are being called as expected.
        """
        # Setup
        instance = MultiTableMetadata()

        parent_table = Mock()
        parent_table.primary_key = 'id'
        parent_table.columns = {
            'id': {'sdtype': 'numerical'},
            'session': {'sdtype': 'numerical'},
            'transactions': {'sdtype': 'numerical'},
        }

        child_table = Mock()
        child_table.primary_key = 'session_id'
        child_table.columns = {
            'user_id': {'sdtype': 'numerical'},
            'session_id': {'sdtype': 'numerical'},
            'transactions': {'sdtype': 'numerical'},
        }

        instance.tables = {
            'users': parent_table,
            'sessions': child_table,
        }
        instance.relationships = [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions',
                'parent_primary_key': 'id',
                'child_foreign_key': 'user_id',
            }
        ]

        instance._validate_relationship_sdtypes = Mock()
        instance._validate_missing_relationship_keys = Mock()

        # Run
        instance._validate_relationship('users', 'sessions', 'id', 'user_id')

        # Assert
        mock_validate_no_missing_tables_in_relationship.assert_called_once_with(
            'users', 'sessions', instance.tables.keys())
        instance._validate_missing_relationship_keys.assert_called_once_with(
            'users', 'id', 'sessions', 'user_id')
        mock_validate_relationship_key_length.assert_called_once_with(
            'users', 'id', 'sessions', 'user_id')
        instance._validate_relationship_sdtypes.assert_called_once_with(
            'users', 'id', 'sessions', 'user_id')

    def test__get_foreign_keys(self):
        """Test that this method returns the foreign keys for a given table name and child name."""
        # Setup
        metadata = self.get_metadata()

        # Run
        result = metadata._get_foreign_keys('users', 'sessions')

        # Assert
        assert result == ['user_id']

    def test__get_all_foreign_keys(self):
        """Test that this method returns the all the foreign keys for a table."""
        # Setup
        instance = self.get_metadata()
        instance.add_column('transactions', 'user_id', sdtype='id')
        instance.add_relationship(
            parent_table_name='users',
            parent_primary_key='id',
            child_table_name='transactions',
            child_foreign_key='user_id'
        )

        # Run
        result = instance._get_all_foreign_keys('transactions')

        # Assert
        assert set(result) == {'user_id', 'session_id'}

    def test_add_relationship(self):
        """Test the ``add_relationship`` method of ``MultiTableMetadata``.

        Test that when passing a valid ``relationship`` this is being added to the
        ``instance.relationships``.

        Setup:
            - Instance of ``MultiTableMetadata``.
            - Mock of ``parent_table`` simulating a ``SingleTableMetadata``.
            - Mock of ``child_table`` simulating a ``SingleTableMetadata``.
            - Add those to ``instance.tables``.

        Mock:
            - Mock ``validate_child_map_circular_relationship``.

        Input:
            - ``parent_table_name`` string representing the parent table name.
            - ``child_table_name`` string representing the child table name.
            - ``parent_primary_key`` string representing the ``primary_key`` of the parent.
            - ``child_foreign_key`` string representing the ``foreing_Key`` of the child table.

        Side Effects:
            - ``instance.relationships`` has been updated.
        """
        # Setup
        instance = MultiTableMetadata()
        instance._validate_child_map_circular_relationship = Mock()
        instance._validate_relationship_does_not_exist = Mock()
        parent_table = Mock()
        parent_table.primary_key = 'id'
        parent_table.columns = {
            'id': {'sdtype': 'numerical'},
            'session': {'sdtype': 'numerical'},
            'transactions': {'sdtype': 'numerical'},
        }

        child_table = Mock()
        child_table.primary_key = 'session_id'
        child_table.columns = {
            'user_id': {'sdtype': 'numerical'},
            'session_id': {'sdtype': 'numerical'},
            'transactions': {'sdtype': 'numerical'},
        }

        instance.tables = {
            'users': parent_table,
            'sessions': child_table,
        }

        # Run
        instance.add_relationship('users', 'sessions', 'id', 'user_id')

        # Assert
        instance.relationships == [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions',
                'parent_primary_key': 'id',
                'child_foreign_key': 'user_id',
            }
        ]
        instance._validate_child_map_circular_relationship.assert_called_once_with(
            {'users': {'sessions'}})
        instance._validate_relationship_does_not_exist.assert_called_once_with(
            'users', 'id', 'sessions', 'user_id'
        )

    def test_add_relationship_child_key_is_primary_key(self):
        """Test that passing a primary key as ``child_foreign_key`` crashes."""
        # Setup
        table = pd.DataFrame({
            'pk': [1, 2, 3],
            'col1': [.1, .1, .2],
            'col2': ['a', 'b', 'c']
        })
        metadata = MultiTableMetadata()
        metadata.detect_table_from_dataframe('table', table)
        metadata.update_column('table', 'pk', sdtype='id')
        metadata.set_primary_key('table', 'pk')
        metadata.detect_table_from_dataframe('table2', table)
        metadata.update_column('table2', 'pk', sdtype='id')
        metadata.set_primary_key('table2', 'pk')

        # Run and Assert
        err_msg = re.escape(
            "Invalid relationship between table 'table' and table "
            "'table2'. A relationship must connect a primary key "
            'with a non-primary key.'
        )
        with pytest.raises(InvalidMetadataError, match=err_msg):
            metadata.add_relationship('table', 'table2', 'pk', 'pk')

    def test__validate_single_table(self):
        """Test ``_validate_single_table``.

        Test that ``_validate_single_table`` iterates over the ``self.tables`` items and
        calls their ``validate()`` method, catches the error if raised and parses it to
        ``MultiTableMetadata`` error message.

        Setup:
            - Create a ``SingleTableMetadata`` that's invalid.
            - Instance of ``MultiTableMetadata`` that contains an invalid and a valid table.

        Side Effects:
            - Errors has been updated with the error message for that column.
        """
        # Setup
        table_accounts = SingleTableMetadata.load_from_dict({
            'columns': {
                'id': {'sdtype': 'numerical'},
                'branch_id': {'sdtype': 'numerical'},
                'amount': {'sdtype': 'numerical'},
                'start_date': {'sdtype': 'datetime'},
                'owner': {'sdtype': 'id'},
            },
            'primary_key': 'branches'
        })

        instance = Mock()
        users_mock = Mock()
        users_mock.columns = {}
        instance.tables = {
            'accounts': table_accounts,
            'users': users_mock
        }
        errors = []

        # Run
        MultiTableMetadata._validate_single_table(instance, errors)

        # Assert
        expected_error_msg = (
            "Table: accounts\nUnknown primary key values {'branches'}. "
            'Keys should be columns that exist in the table.'
        )
        empty_table_error_message = (
            "Table 'users' has 0 columns. Use 'add_column' to specify its columns."
        )
        assert errors == ['\n', expected_error_msg, empty_table_error_message]
        instance.tables['users'].validate.assert_called_once()

    def test__validate_all_tables_connected_connected(self):
        """Test ``_validate_all_tables_connected``.

        Test that ``_validate_all_tables_connected`` performs a ``DFS`` and marks nodes as
        ``connected`` if all are connected no error is being raised.

        Setup:
            - Create a mock instance of the ``MultiTableMetadata``.
            - Set ``_table`` to the ``instance``.
            - Create a list of ``relationships``.
            - Create ``parent_map`` and ``child_map``.

        Mock:
            - Mock the tables as they are not being used.

        Side Effects:
            - No error raised.
        """
        # Setup
        instance = Mock()
        instance.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
            'accounts': Mock()
        }
        relationships = [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions'
            },
            {
                'parent_table_name': 'users',
                'child_table_name': 'transactions'
            },
            {
                'parent_table_name': 'users',
                'child_table_name': 'accounts'
            },

        ]

        parent_map = defaultdict(set)
        child_map = defaultdict(set)
        for relation in relationships:
            parent_name = relation['parent_table_name']
            child_name = relation['child_table_name']
            parent_map[child_name].add(parent_name)
            child_map[parent_name].add(child_name)

        # Run
        MultiTableMetadata._validate_all_tables_connected(instance, parent_map, child_map)

    def test__validate_all_tables_connected_not_connected(self):
        """Test ``_validate_all_tables_connected``.

        Test that ``_validate_all_tables_connected`` performs a ``DFS`` and marks nodes as
        ``connected``. An ``InvalidMetadataError`` is being raised since one colmn is not
        connected.

        Setup:
            - Create a mock instance of the ``MultiTableMetadata``.
            - Set ``_table`` to the ``instance``.
            - Create a list of ``relationships``.
            - Create ``parent_map`` and ``child_map``.

        Mock:
            - Mock the tables as they are not being used.

        Side Effects:
            - ``InvalidMetadataError`` is raised stating that a ``table`` is disjointed.
        """
        # Setup
        instance = Mock()
        instance.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
            'accounts': Mock(),
        }
        relationships = [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions'
            },
            {
                'parent_table_name': 'users',
                'child_table_name': 'transactions'
            },

        ]

        parent_map = defaultdict(set)
        child_map = defaultdict(set)
        for relation in relationships:
            parent_name = relation['parent_table_name']
            child_name = relation['child_table_name']
            parent_map[child_name].add(parent_name)
            child_map[parent_name].add(child_name)

        # Run
        error_msg = re.escape(
            "The relationships in the dataset are disjointed. Table ['accounts'] "
            'is not connected to any of the other tables.'
        )
        with pytest.raises(InvalidMetadataError, match=error_msg):
            MultiTableMetadata._validate_all_tables_connected(instance, parent_map, child_map)

    def test__validate_all_tables_connected_multiple_not_connected(self):
        """Test ``_validate_all_tables_connected``.

        Test that ``_validate_all_tables_connected`` performs a ``DFS`` and marks nodes as
        ``connected``. An ``InvalidMetadataError`` is being raised since two colmns are not
        connected.

        Setup:
            - Create a mock instance of the ``MultiTableMetadata``.
            - Set ``_table`` to the ``instance``.
            - Create a list of ``relationships``.
            - Create ``parent_map`` and ``child_map``.

        Mock:
            - Mock the tables as they are not being used.

        Side Effects:
            - ``InvalidMetadataError`` is raised stating that more than one tables are disjointed.
        """
        # Setup
        instance = Mock()
        instance.tables = {
            'users': Mock(),
            'sessions': Mock(),
            'transactions': Mock(),
            'accounts': Mock(),
            'branches': Mock()
        }
        relationships = [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions'
            },
            {
                'parent_table_name': 'users',
                'child_table_name': 'transactions'
            },

        ]

        parent_map = defaultdict(set)
        child_map = defaultdict(set)
        for relation in relationships:
            parent_name = relation['parent_table_name']
            child_name = relation['child_table_name']
            parent_map[child_name].add(parent_name)
            child_map[parent_name].add(child_name)

        # Run
        err_msg = re.escape(
            "The relationships in the dataset are disjointed. Tables ['accounts', 'branches'] "
            'are not connected to any of the other tables.'
        )
        with pytest.raises(InvalidMetadataError, match=err_msg):
            MultiTableMetadata._validate_all_tables_connected(instance, parent_map, child_map)

    def test_validate(self):
        """Test the method ``validate``.

        Test that when a valid ``MultiTableMetadata`` has been provided no errors are being raised.

        Setup:
            - Instance of ``MultiTableMetadata`` with all valid tables and relationships.
        """
        # Setup
        instance = self.get_metadata()

        # Run
        instance.validate()

    def test_validate_raises_errors(self):
        """Test the method ``validate``.

        Test that when an invalid ``MultiTableMetadata`` has been provided, all different errors
        are being raised.

        Setup:
            - Instance of ``MultiTableMetadata`` with all valid tables and relationships.
        """
        # Setup
        instance = self.get_metadata()
        instance.tables['users'].primary_key = None
        instance.tables['transactions'].columns['session_id']['sdtype'] = 'datetime'
        instance.tables['payments'].columns['date']['sdtype'] = 'id'
        instance.tables['payments'].columns['date']['regex_format'] = '[A-z{'
        instance.relationships.pop(-1)

        # Run
        error_msg = re.escape(
            'The metadata is not valid\n'
            '\nTable: payments'
            "\nInvalid regex format string '[A-z{' for id column 'date'."
            '\n\nRelationships:'
            "\nThe parent table 'users' does not have a primary key set. "
            "Please use 'set_primary_key' in order to set one."
            "\nRelationship between tables ('sessions', 'transactions') is invalid. "
            'The primary and foreign key columns are not the same type.'
            "\nThe relationships in the dataset are disjointed. Table ['payments'] "
            'is not connected to any of the other tables.'
        )

        # Run / Assert
        with pytest.raises(InvalidMetadataError, match=error_msg):
            instance.validate()

    def test_validate_child_key_is_primary_key(self):
        """Test it crashes if the child key is a primary key."""
        # Setup
        table = pd.DataFrame({
            'pk': [1, 2, 3],
            'col1': [.1, .1, .2],
            'col2': ['a', 'b', 'c']
        })
        metadata = MultiTableMetadata()
        metadata.detect_table_from_dataframe('table', table)
        metadata.update_column('table', 'pk', sdtype='id')
        metadata.set_primary_key('table', 'pk')
        metadata.detect_table_from_dataframe('table2', table)
        metadata.update_column('table2', 'pk', sdtype='id')
        metadata.set_primary_key('table2', 'pk')

        metadata.relationships = [
            {
                'parent_table_name': 'table',
                'parent_primary_key': 'pk',
                'child_table_name': 'table2',
                'child_foreign_key': 'pk',
            }
        ]

        # Run and Assert
        err_msg = re.escape(
            'The metadata is not valid\n'
            'Relationships:\n'
            "Invalid relationship between table 'table' and table "
            "'table2'. A relationship must connect a primary key "
            'with a non-primary key.'
        )
        with pytest.raises(InvalidMetadataError, match=err_msg):
            metadata.validate()

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_add_table(self, table_metadata_mock):
        """Test that the method adds the table name to ``instance.tables``."""
        # Setup
        instance = MultiTableMetadata()

        # Run
        instance.add_table('users')

        # Assert
        assert instance.tables == {'users': table_metadata_mock.return_value}

    def test_add_table_empty_string(self):
        """Test that the method raises an error if the table name is an empty string."""
        # Setup
        instance = MultiTableMetadata()

        # Run and Assert
        error_message = re.escape(
            "Invalid table name (''). The table name must be a non-empty string."
        )
        with pytest.raises(InvalidMetadataError, match=error_message):
            instance.add_table('')

    def test_add_table_not_string(self):
        """Test that the method raises an error if the table name is not a string."""
        # Setup
        instance = MultiTableMetadata()

        # Run and Assert
        error_message = re.escape(
            "Invalid table name (''). The table name must be a non-empty string."
        )
        with pytest.raises(InvalidMetadataError, match=error_message):
            instance.add_table(Mock())

    def test_add_table_table_already_exists(self):
        """Test that the method raises an error if the table already exists."""
        # Setup
        instance = MultiTableMetadata()
        instance.tables = {'users': Mock()}

        # Run and Assert
        error_message = re.escape(
            "Cannot add a table named 'users' because it already exists in the metadata. Please "
            'choose a different name.'
        )
        with pytest.raises(InvalidMetadataError, match=error_message):
            instance.add_table('users')

    def test_to_dict(self):
        """Test the ``to_dict`` method of ``MultiTableMetadata``.

        Setup:
            - Instance of ``MultiTableMetadata``.
            - Add mocked values to ``instance.tables`` and ``instance.relationships``.
        Mock:
            - Mock ``SingleTableMetadata`` like object to ``instance.tables``.

        Output:
            - A dict representation containing ``tables`` and ``relationships`` has to be returned
              with ``dict`` values of ``tables``.
        """
        # Setup
        table_accounts = Mock()
        table_accounts.to_dict.return_value = {
            'id': {'sdtype': 'numerical'},
            'branch_id': {'sdtype': 'numerical'},
            'amount': {'sdtype': 'numerical'},
            'start_date': {'sdtype': 'datetime'},
            'owner': {'sdtype': 'id'},
        }
        table_branches = Mock()
        table_branches.to_dict.return_value = {
            'id': {'sdtype': 'numerical'},
            'name': {'sdtype': 'id'},
        }
        instance = MultiTableMetadata()
        instance.tables = {
            'accounts': table_accounts,
            'branches': table_branches
        }
        instance.relationships = [
            {
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'chil_foreign_key': 'branch_id',
            }
        ]

        # Run
        result = instance.to_dict()

        # Assert
        expected_result = {
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
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'accounts',
                    'parent_primary_key': 'id',
                    'child_table_name': 'branches',
                    'chil_foreign_key': 'branch_id',
                }
            ],
            'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
        }
        assert result == expected_result

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test__set_metadata(self, mock_singletablemetadata):
        """Test the ``_set_metadata`` method for ``MultiTableMetadata``.

        Setup:
            - instance of ``MultiTableMetadata``.
            - A dict representing a ``MultiTableMetadata``.

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
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'accounts',
                    'parent_primary_key': 'id',
                    'child_table_name': 'branches',
                    'chil_foreign_key': 'branch_id',
                }
            ]
        }

        single_table_accounts = object()
        single_table_branches = object()
        mock_singletablemetadata.load_from_dict.side_effect = [
            single_table_accounts,
            single_table_branches
        ]

        instance = MultiTableMetadata()

        # Run
        instance._set_metadata_dict(multitable_metadata)

        # Assert
        assert instance.tables == {
            'accounts': single_table_accounts,
            'branches': single_table_branches
        }

        assert instance.relationships == [
            {
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'chil_foreign_key': 'branch_id',
            }
        ]

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_load_from_dict(self, mock_singletablemetadata):
        """Test that ``load_from_dict`` returns a instance of ``MultiTableMetadata``.

        Test that when calling the ``load_from_dict`` method a new instance with the passed
        python ``dict`` details should be created.

        Setup:
            - A dict representing a ``MultiTableMetadata``.

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
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'accounts',
                    'parent_primary_key': 'id',
                    'child_table_name': 'branches',
                    'child_foreign_key': 'branch_id',
                }
            ]
        }

        single_table_accounts = object()
        single_table_branches = object()
        mock_singletablemetadata.load_from_dict.side_effect = [
            single_table_accounts,
            single_table_branches
        ]

        # Run
        instance = MultiTableMetadata.load_from_dict(multitable_metadata)

        # Assert
        assert instance.tables == {
            'accounts': single_table_accounts,
            'branches': single_table_branches
        }

        assert instance.relationships == [
            {
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'child_foreign_key': 'branch_id',
            }
        ]

    @patch('sdv.metadata.multi_table.json')
    def test___repr__(self, mock_json):
        """Test that the ``__repr__`` method.

        Test that the ``__repr__`` method calls the ``json.dumps``  method and
        returns its output.

        Setup:
            - Instance of ``MultiTableMetadata``.
        Mock:
            - ``json`` from ``sdv.metadata.multi_table``.

        Output:
            - ``json.dumps`` return value.
        """
        # Setup
        instance = MultiTableMetadata()

        # Run
        res = instance.__repr__()

        # Assert
        mock_json.dumps.assert_called_once_with(instance.to_dict(), indent=4)
        assert res == mock_json.dumps.return_value

    @patch('sdv.metadata.multi_table.visualize_graph')
    def test_visualize_show_relationship_and_details(self, visualize_graph_mock):
        """Test the ``visualize`` method.

        If both the ``show_relationship_labels`` and ``show_table_details`` parameters are
        True, then the edges should have labels and the labels for the nodes should include
        column info, primary keys and alternate keys.

        Setup:
            - Mock the ``visualize_graph`` function.
            - Set the tables and relationships for the multi-table metadata.

        Input:
            - Both ``show_relationship_labels`` and ``show_table_details`` set to True.

        Side effects:
            - The ``visualize_graph_mock`` should be called with the correct nodes and edges.
        """
        # Setup
        metadata = self.get_metadata()

        # Run
        metadata.visualize(True, True)

        # Assert
        expected_payments_label = (
            '{payments|payment_id : id\\luser_id : id\\ldate : datetime\\l|'
            'Primary key: payment_id\\lForeign key (users): user_id\\l}'
        )
        expected_sessions_label = (
            '{sessions|session_id : id\\luser_id : id\\ldevice : categorical\\l|'
            'Primary key: session_id\\lForeign key (users): user_id\\l}'
        )
        expected_transactions_label = (
            '{transactions|transaction_id : id\\lsession_id : id\\ltimestamp : '
            'datetime\\l|Primary key: transaction_id\\lForeign key (sessions): session_id\\l}'
        )
        expected_nodes = {
            'users': '{users|id : id\\lcountry : categorical\\l|Primary key: id\\l\\l}',
            'payments': expected_payments_label,
            'sessions': expected_sessions_label,
            'transactions': expected_transactions_label
        }
        expected_edges = [
            ('users', 'sessions', '  user_id → id'),
            ('sessions', 'transactions', '  session_id → session_id'),
            ('users', 'payments', '  user_id → id')
        ]
        visualize_graph_mock.assert_called_once_with(expected_nodes, expected_edges, None)

    @patch('sdv.metadata.multi_table.visualize_graph')
    def test_visualize_show_relationship_only(self, visualize_graph_mock):
        """Test the ``visualize`` method.

        If ``show_relationship_labels`` is True but ``show_table_details``is False,
        then the edges should have labels and the labels for the nodes should be just
        the table name.

        Setup:
            - Mock the ``visualize_graph`` function.
            - Set the tables and relationships for the multi-table metadata.

        Input:
            - ``show_relationship_labels`` set to True.
            - ``show_table_details`` set to False.

        Side effects:
            - The ``visualize_graph_mock`` should be called with the correct nodes and edges.
        """
        # Setup
        metadata = self.get_metadata()

        # Run
        metadata.visualize(False, True, 'output.jpg')

        # Assert
        expected_nodes = {
            'users': 'users',
            'payments': 'payments',
            'sessions': 'sessions',
            'transactions': 'transactions'
        }
        expected_edges = [
            ('users', 'sessions', '  user_id → id'),
            ('sessions', 'transactions', '  session_id → session_id'),
            ('users', 'payments', '  user_id → id')
        ]
        visualize_graph_mock.assert_called_once_with(expected_nodes, expected_edges, 'output.jpg')

    @patch('sdv.metadata.multi_table.visualize_graph')
    def test_visualize_show_table_details_only(self, visualize_graph_mock):
        """Test the ``visualize`` method.

        If ``show_relationship_labels`` is False but ``show_table_details``is True,
        then the edges should not have labels and the labels for the nodes should should include
        column info, primary keys and alternate keys.

        Setup:
            - Mock the ``visualize_graph`` function.
            - Set the tables and relationships for the multi-table metadata.

        Input:
            - ``show_relationship_labels`` set to False.
            - ``show_table_details`` set to True.

        Side effects:
            - The ``visualize_graph_mock`` should be called with the correct nodes and edges.
        """
        # Setup
        metadata = self.get_metadata()

        # Run
        metadata.visualize(True, False, 'output.jpg')

        # Assert
        expected_payments_label = (
            '{payments|payment_id : id\\luser_id : id\\ldate : datetime\\l|'
            'Primary key: payment_id\\lForeign key (users): user_id\\l}'
        )
        expected_sessions_label = (
            '{sessions|session_id : id\\luser_id : id\\ldevice : categorical\\l|'
            'Primary key: session_id\\lForeign key (users): user_id\\l}'
        )
        expected_transactions_label = (
            '{transactions|transaction_id : id\\lsession_id : id\\ltimestamp : '
            'datetime\\l|Primary key: transaction_id\\lForeign key (sessions): session_id\\l}'
        )
        expected_nodes = {
            'users': '{users|id : id\\lcountry : categorical\\l|Primary key: id\\l\\l}',
            'payments': expected_payments_label,
            'sessions': expected_sessions_label,
            'transactions': expected_transactions_label
        }
        expected_edges = [
            ('users', 'sessions', ''),
            ('sessions', 'transactions', ''),
            ('users', 'payments', '')
        ]
        visualize_graph_mock.assert_called_once_with(expected_nodes, expected_edges, 'output.jpg')

    def test_add_column(self):
        """Test the ``add_column`` method.

        The method should get the appropriate table and call ``add_column`` on it.

        Setup:
            - Set the ``_tables`` attribute to have a mock for the table name.

        Input:
            - table_name that matches what is in ``_tables``.
            - column_name.
            - Some key word arguments.

        Side effect:
            - The mock should have the ``add_column`` method called with the right attributes.
        """
        # Setup
        metadata = MultiTableMetadata()
        table = Mock()
        metadata.tables = {'table': table}

        # Run
        metadata.add_column('table', 'column', sdtype='numerical', pii=False)

        # Assert
        table.add_column.assert_called_once_with('column', sdtype='numerical', pii=False)

    def test_add_column_table_does_not_exist(self):
        """Test the ``add_column`` method.

        If the table doesn't exist, an error should be raised.

        Input:
            - table_name that isn't in ``_tables``.
            - column_name.
            - Some key word arguments.

        Side effect:
            - Should raise an error.
        """
        # Setup
        metadata = MultiTableMetadata()

        # Run
        error_message = re.escape("Unknown table name ('table')")
        with pytest.raises(InvalidMetadataError, match=error_message):
            metadata.add_column('table', 'column', sdtype='numerical', pii=False)

    def test_update_column(self):
        """Test the ``update_column`` method.

        The method should get the appropriate table and call ``update_column`` on it.

        Setup:
            - Set the ``_tables`` attribute to have a mock for the table name.

        Input:
            - table_name that matches what is in ``_tables``.
            - column_name.
            - Some key word arguments.

        Side effect:
            - The mock should have the ``update_column`` method called with the right attributes.
        """
        # Setup
        metadata = MultiTableMetadata()
        table = Mock()
        metadata.tables = {'table': table}

        # Run
        metadata.update_column('table', 'column', sdtype='numerical', pii=False)

        # Assert
        table.update_column.assert_called_once_with('column', sdtype='numerical', pii=False)

    def test_update_column_table_does_not_exist(self):
        """Test the ``update_column`` method.

        If the table doesn't exist, an error should be raised.

        Input:
            - table_name that isn't in ``_tables``.
            - column_name.
            - Some key word arguments.

        Side effect:
            - Should raise an error.
        """
        # Setup
        metadata = MultiTableMetadata()

        # Run
        error_message = re.escape("Unknown table name ('table')")
        with pytest.raises(InvalidMetadataError, match=error_message):
            metadata.update_column('table', 'column', sdtype='numerical', pii=False)

    @patch('sdv.metadata.multi_table.LOGGER')
    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_detect_table_from_csv(self, single_table_mock, log_mock):
        """Test the ``detect_table_from_csv`` method.

        If the table does not already exist, a ``SingleTableMetadata`` instance
        should be created and call the ``detect_from_csv`` method.

        Setup:
            - Mock the ``SingleTableMetadata`` class and the print function.

        Assert:
            - Table should be added to ``self.tables``.
        """
        # Setup
        metadata = MultiTableMetadata()
        fake_data = Mock()
        single_table_mock.return_value._load_data_from_csv.return_value = fake_data
        single_table_mock.return_value.to_dict.return_value = {
            'columns': {'a': {'sdtype': 'numerical'}}
        }

        # Run
        metadata.detect_table_from_csv('table', 'path.csv')

        # Assert
        single_table_mock.return_value._load_data_from_csv.assert_called_once_with('path.csv')
        single_table_mock.return_value._detect_columns.assert_called_once_with(fake_data)
        assert metadata.tables == {'table': single_table_mock.return_value}

        expected_log_calls = call(
            'Detected metadata:\n'
            '{\n'
            '    "columns": {\n'
            '        "a": {\n'
            '            "sdtype": "numerical"\n'
            '        }\n'
            '    }'
            '\n}'
        )
        log_mock.info.assert_has_calls([expected_log_calls])

    def test_detect_table_from_csv_table_already_exists(self):
        """Test the ``detect_table_from_csv`` method.

        If the table already exists, an error should be raised.

        Setup:
            - Set the ``_tables`` dict to already have the table.

        Input:
            - Table name.
            - Path.

        Side effect:
            - An error should be raised.
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata.tables = {'table': Mock()}

        # Run
        error_message = (
            "Metadata for table 'table' already exists. Specify a new table name or "
            'create a new MultiTableMetadata object for other data sources.'
        )
        with pytest.raises(InvalidMetadataError, match=error_message):
            metadata.detect_table_from_csv('table', 'path.csv')

    @patch('sdv.metadata.multi_table.LOGGER')
    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_detect_table_from_dataframe(self, single_table_mock, log_mock):
        """Test the ``detect_table_from_dataframe`` method.

        If the table does not already exist, a ``SingleTableMetadata`` instance
        should be created and call the ``detect_from_dataframe`` method.

        Setup:
            - Mock the ``SingleTableMetadata`` class and print function.

        Assert:
            - Table should be added to ``self.tables``.
        """
        # Setup
        metadata = MultiTableMetadata()
        data = pd.DataFrame()
        single_table_mock.return_value.to_dict.return_value = {
            'columns': {'a': {'sdtype': 'numerical'}}
        }

        # Run
        metadata.detect_table_from_dataframe('table', data)

        # Assert
        single_table_mock.return_value._detect_columns.assert_called_once_with(data)
        assert metadata.tables == {'table': single_table_mock.return_value}

        expected_log_calls = call(
            'Detected metadata:\n'
            '{\n'
            '    "columns": {\n'
            '        "a": {\n'
            '            "sdtype": "numerical"\n'
            '        }\n'
            '    }'
            '\n}'
        )
        log_mock.info.assert_has_calls([expected_log_calls])

    def test_detect_table_from_dataframe_table_already_exists(self):
        """Test the ``detect_table_from_dataframe`` method.

        If the table already exists, an error should be raised.

        Setup:
            - Set the ``_tables`` dict to already have the table.

        Input:
            - Table name.
            - Dataframe.

        Side effect:
            - An error should be raised.
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata.tables = {'table': Mock()}

        # Run
        error_message = (
            "Metadata for table 'table' already exists. Specify a new table name or "
            'create a new MultiTableMetadata object for other data sources.'
        )
        with pytest.raises(InvalidMetadataError, match=error_message):
            metadata.detect_table_from_dataframe('table', pd.DataFrame())

    def test__validate_table_exists(self):
        """Test ``_validate_table_exists``.

        Expected to raise an error when the passed table name is not present in the metadata.
        Expected to do nothing otherwise.

        Input:
            - Table name

        Raises:
            - ``InvalidMetadataError`` if the table name is not in the metadata.
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata.tables = {'table1': 'val', 'table2': 'val'}

        # Run
        metadata._validate_table_exists('table1')

        # Assert
        err_msg = re.escape("Unknown table name ('table3').")
        with pytest.raises(InvalidMetadataError, match=err_msg):
            metadata._validate_table_exists('table3')

    def test_set_primary_key(self):
        """Test ``set_primary_key``.

        The method should validate the table exists and call
        ``SingleTableMetadata.set_primary_key``.

        Setup:
            - Instantiate ``MultiTableMetadata`` with some ``_tables``.
            - Mock ``_validate_table_exists``.

        Input:
            - Table name
            - Column name
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata.tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        metadata.set_primary_key('table1', 'col')

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata.tables['table1'].set_primary_key.assert_called_once_with('col')

    def test_set_sequence_key(self):
        """Test ``set_sequence_key``.

        The method should validate the table exists and call
        ``SingleTableMetadata.set_sequence_key``.

        Setup:
            - Instantiate ``MultiTableMetadata`` with some ``_tables``.
            - Mock ``_validate_table_exists``.

        Input:
            - Table name
            - Column name
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata.tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        warn_msg = 'Sequential modeling is not yet supported on SDV Multi Table models.'
        with pytest.warns(Warning, match=warn_msg):
            metadata.set_sequence_key('table1', 'col')

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata.tables['table1'].set_sequence_key.assert_called_once_with('col')

    def test_add_alternate_keys(self):
        """Test ``add_alternate_keys``.

        The method should validate the table exists and call
        ``SingleTableMetadata.add_alternate_keys``.

        Setup:
            - Instantiate ``MultiTableMetadata`` with some ``_tables``.
            - Mock ``_validate_table_exists``.

        Input:
            - Table name
            - List of column names
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata.tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        metadata.add_alternate_keys('table1', ['col1', 'col2'])

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata.tables['table1'].add_alternate_keys.assert_called_once_with(['col1', 'col2'])

    def test_set_sequence_index(self):
        """Test ``set_sequence_index``.

        The method should validate the table exists and call
        ``SingleTableMetadata.set_sequence_index``.

        Setup:
            - Instantiate ``MultiTableMetadata`` with some ``_tables``.
            - Mock ``_validate_table_exists``.

        Input:
            - Table name
            - Column name
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata.tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        warn_msg = 'Sequential modeling is not yet supported on SDV Multi Table models.'
        with pytest.warns(Warning, match=warn_msg):
            metadata.set_sequence_index('table1', 'col')

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata.tables['table1'].set_sequence_index.assert_called_once_with('col')

    def test_add_constraint(self):
        """Test the ``add_constraint`` method.

        The method should get the appropriate table and call ``add_constraint`` on it.

        Setup:
            - Set the ``_tables`` attribute to have a mock for the table name.

        Input:
            - table_name that matches what is in ``_tables``.
            - column_name.
            - Some key word arguments.

        Side effect:
            - The mock should have the ``add_constraint`` method called with the right attributes.
        """
        # Setup
        metadata = MultiTableMetadata()
        table = Mock()
        metadata.tables = {'table': table}

        # Run
        metadata.add_constraint('table', 'Inequality', low_column_name='a', high_column_name='b')

        # Assert
        table.add_constraint.assert_called_once_with(
            'Inequality', low_column_name='a', high_column_name='b')

    def test_add_constraint_table_does_not_exist(self):
        """Test the ``add_constraint`` method.

        If the table doesn't exist, an error should be raised.

        Input:
            - table_name that isn't in ``_tables``.
            - column_name.
            - Some key word arguments.

        Side effect:
            - Should raise an error.
        """
        # Setup
        metadata = MultiTableMetadata()

        # Run
        error_message = re.escape("Unknown table name ('table')")
        with pytest.raises(InvalidMetadataError, match=error_message):
            metadata.add_constraint(
                'table', 'Inequality', low_column_name='a', high_column_name='b')

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
            MultiTableMetadata.load_from_json('filepath.json')

    @patch('sdv.metadata.utils.open')
    @patch('sdv.metadata.utils.Path')
    @patch('sdv.metadata.utils.json')
    def test_load_from_json(self, mock_json, mock_path, mock_open):
        """Test the ``load_from_json`` method.

        Test that ``load_from_json`` function creates an instance with the contents returned by the
        ``json`` load function.

        Mock:
            - Mock the ``Path`` library in order to return ``True``.
            - Mock the ``json`` library in order to use a custom return.
            - Mock the ``open`` in order to avoid loading a binary file.

        Input:
            - String representing a filepath.

        Output:
            - ``SingleTableMetadata`` instance with the custom configuration from the ``json``
              file (``json.load`` return value)
        """
        # Setup
        instance = MultiTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'
        mock_json.load.return_value = {
            'tables': {
                'table1': {
                    'columns': {
                        'animals': {
                            'type': 'categorical'
                        }
                    },
                    'primary_key': 'animals',
                    'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1'
                }
            },
            'relationships': {}
        }

        # Run
        instance = MultiTableMetadata.load_from_json('filepath.json')

        # Asserts
        assert list(instance.tables.keys()) == ['table1']
        assert instance.tables['table1'].columns == {'animals': {'type': 'categorical'}}
        assert instance.tables['table1'].primary_key == 'animals'
        assert instance.tables['table1'].sequence_key is None
        assert instance.tables['table1'].alternate_keys == []
        assert instance.tables['table1'].sequence_index is None
        assert instance.tables['table1']._version == 'SINGLE_TABLE_V1'

    @patch('sdv.metadata.utils.Path')
    def test_save_to_json_file_exists(self, mock_path):
        """Test the ``save_to_json`` method.

        Test that when attempting to write over a file that already exists, the method
        raises a ``ValueError``.

        Setup:
            - instance of ``MultiTableMetadata``.
        Mock:
            - Mock ``Path`` in order to point that the file does exist.

        Side Effects:
            - Raise ``ValueError`` pointing that the file does exist.
        """
        # Setup
        instance = MultiTableMetadata()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.name = 'filepath.json'

        # Run / Assert
        error_msg = (
            "A file named 'filepath.json' already exists in this folder. Please specify "
            'a different filename.'
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.save_to_json('filepath.json')

    def test_save_to_json(self, tmp_path):
        """Test the ``save_to_json`` method.

        Test that ``save_to_json`` stores a ``json`` file and dumps the instance dict into
        it.

        Setup:
            - instance of ``MultiTableMetadata``.
            - Use ``TemporaryDirectory`` to store the file in order to read it afterwards and
              assert it's contents.

        Side Effects:
            - Creates a json representation of the instance.
        """
        # Setup
        instance = MultiTableMetadata()

        # Run / Assert
        file_name = tmp_path / 'multitable.json'
        instance.save_to_json(file_name)

        with open(file_name, 'rb') as multi_table_file:
            saved_metadata = json.load(multi_table_file)
            assert saved_metadata == instance.to_dict()

    def test__convert_relationships(self):
        """Test the ``_convert_relationships`` method.

        The method should take in a metadata dictionary in the old schema and extract the
        relationship info into a dictionary for the relationship part of the new schema.

        Input:
            - A metadata dict in the old schema.

        Output:
            - The relationships portion of the new schema.
        """
        # Setup
        old_metadata = {
            'tables': {
                'nesreca': {
                    'fields': {
                        'upravna_enota': {
                            'type': 'id',
                            'subtype': 'integer',
                            'ref': {
                                'table': 'upravna_enota',
                                'field': 'id_upravna_enota'
                            }
                        },
                        'id_nesreca': {
                            'type': 'id',
                            'subtype': 'integer'
                        },
                    },
                    'primary_key': 'id_nesreca'
                },
                'oseba': {
                    'fields': {
                        'upravna_enota': {
                            'type': 'id',
                            'subtype': 'integer',
                            'ref': {
                                'table': 'upravna_enota',
                                'field': 'id_upravna_enota'
                            }
                        },
                        'id_nesreca': {
                            'type': 'id',
                            'subtype': 'integer',
                            'ref': {
                                'table': 'nesreca',
                                'field': 'id_nesreca'
                            }
                        },
                    },
                },
                'upravna_enota': {
                    'fields': {
                        'id_upravna_enota': {
                            'type': 'id',
                            'subtype': 'integer'
                        }
                    },
                    'primary_key': 'id_upravna_enota'
                }
            }
        }

        # Run
        relationships = MultiTableMetadata._convert_relationships(old_metadata)

        # Assert
        expected = [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'nesreca',
                'child_foreign_key': 'upravna_enota'
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca'
            },
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota'
            }
        ]
        for relationship in expected:
            assert relationship in relationships

    @patch('sdv.metadata.multi_table.read_json')
    @patch('sdv.metadata.multi_table.MultiTableMetadata._convert_relationships')
    @patch('sdv.metadata.multi_table.convert_metadata')
    @patch('sdv.metadata.multi_table.MultiTableMetadata.load_from_dict')
    def test_upgrade_metadata(
            self, from_dict_mock, convert_mock, relationships_mock, read_json_mock):
        """Test the ``upgrade_metadata`` method.

        The method should validate that the ``new_filepath`` does not exist, read the old metadata
        from a file, convert it and save it to the ``new_filepath``. It should loop through every
        table in the old metadata and convert it using ``SingleTableMetadata._convert_metadata``.

        Setup:
            - Mock ``read_json`` to return a metadata dict with a few tables.
            - Mock ``validate_file_does_not_exist``.
            - Mock the ``convert_metadata`` method to return something.
            - Mock the ``from_dict`` method to return a mock.
            - Mock the `SingleTableMetadata._convert_metadata`` method.

        Input:
            - A fake old filepath.
            - A fake new filepath.

        Side effect:
            - The mock should call ``save_to_json`` and ``validate``.
        """
        # Setup
        convert_mock.side_effect = [
            {'columns': {'column1': {'sdtype': 'numerical'}}},
            {'columns': {'column2': {'sdtype': 'categorical'}}},
        ]
        new_metadata = Mock()
        from_dict_mock.return_value = new_metadata
        read_json_mock.return_value = {
            'tables': {
                'table1': {'columns': {'column1': {'type': 'numerical'}}},
                'table2': {'columns': {'column2': {'type': 'categorical'}}}
            }
        }
        relationships_mock.return_value = [
            {
                'parent_table_name': 'table1',
                'parent_primary_key': 'id',
                'child_table_name': 'table2',
                'child_foreign_key': 'id'
            }
        ]

        # Run
        MultiTableMetadata.upgrade_metadata('old')

        # Assert
        read_json_mock.assert_called_once_with('old')
        relationships_mock.assert_called_once_with({
            'tables': {
                'table1': {'columns': {'column1': {'type': 'numerical'}}},
                'table2': {'columns': {'column2': {'type': 'categorical'}}}
            }
        })
        convert_mock.assert_has_calls([
            call({'columns': {'column1': {'type': 'numerical'}}}),
            call({'columns': {'column2': {'type': 'categorical'}}})
        ])
        expected_new_metadata = {
            'tables': {
                'table1': {'columns': {'column1': {'sdtype': 'numerical'}}},
                'table2': {'columns': {'column2': {'sdtype': 'categorical'}}}
            },
            'relationships': [
                {
                    'parent_table_name': 'table1',
                    'parent_primary_key': 'id',
                    'child_table_name': 'table2',
                    'child_foreign_key': 'id'
                }
            ],
            'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
        }
        from_dict_mock.assert_called_once_with(expected_new_metadata)
        new_metadata.validate.assert_called_once()

    @patch('sdv.metadata.multi_table.warnings')
    @patch('sdv.metadata.multi_table.read_json')
    @patch('sdv.metadata.multi_table.MultiTableMetadata._convert_relationships')
    @patch('sdv.metadata.multi_table.convert_metadata')
    @patch('sdv.metadata.multi_table.MultiTableMetadata.load_from_dict')
    def test_upgrade_metadata_validate_error(
            self, from_dict_mock, convert_mock, relationships_mock, read_json_mock, warnings_mock):
        """Test the ``upgrade_metadata`` method.

        The method should validate that the ``new_filepath`` does not exist, read the old metadata
        from a file, convert it and save it to the ``new_filepath``. It should loop through every
        table in the old metadata and convert it using ``convert_metadata``. If the ``validate``
        method raises an error, we should catch it and raise a warning.

        Setup:
            - Mock ``read_json`` to return a metadata dict with a few tables.
            - Mock ``validate_file_does_not_exist``.
            - Mock the ``convert_metadata`` method to return something.
            - Mock the ``from_dict`` method to return a mock.
            - Mock the `SingleTableMetadata._convert_metadata`` method.
            - Mock the ``validate`` method to raise an error.

        Input:
            - A fake old filepath.
            - A fake new filepath.

        Side effect:
            - The mock should call ``save_to_json`` and ``validate``.
        """
        # Setup
        convert_mock.return_value = {}
        new_metadata = Mock()
        from_dict_mock.return_value = new_metadata
        read_json_mock.return_value = {}
        relationships_mock.return_value = []
        new_metadata.validate.side_effect = InvalidMetadataError('blah')

        # Run
        MultiTableMetadata.upgrade_metadata('old')

        # Assert
        read_json_mock.assert_called_once_with('old')
        relationships_mock.assert_called_once_with({})
        expected_new_metadata = {
            'tables': {},
            'relationships': [],
            'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
        }
        from_dict_mock.assert_called_once_with(expected_new_metadata)
        new_metadata.validate.assert_called_once()
        warnings_mock.warn.assert_called_once_with(
            'Successfully converted the old metadata, but the metadata was not valid.'
            'To use this with the SDV, please fix the following errors.\n blah'
        )
