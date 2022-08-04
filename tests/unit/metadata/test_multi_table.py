"""Test Multi Table Metadata."""

import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.multi_table import MultiTableMetadata


class TestMultiTableMetadata:
    """Test ``MultiTableMetadata`` class."""

    def test___init__(self):
        """Test the ``__init__`` method of ``MultiTableMetadata``."""
        # Run
        instance = MultiTableMetadata()

        # Assert
        assert instance._tables == {}
        assert instance._relationships == []

    def test_to_dict(self):
        """Test the ``to_dict`` method of ``MultiTableMetadata``.

        Setup:
            - Instance of ``MultiTableMetadata``.
            - Add mocked values to ``instance._tables`` and ``instance._relationships``.
        Mock:
            - Mock ``SingleTableMetadata`` like object to ``instance._tables``.

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
            'owner': {'sdtype': 'text'},
        }
        table_branches = Mock()
        table_branches.to_dict.return_value = {
            'id': {'sdtype': 'numerical'},
            'name': {'sdtype': 'text'},
        }
        instance = MultiTableMetadata()
        instance._tables = {
            'accounts': table_accounts,
            'branches': table_branches
        }
        instance._relationships = [{
            'parent_table_name': 'accounts',
            'parent_primary_key': 'id',
            'child_table_name': 'branches',
            'chil_foreign_key': 'branch_id',
        }]

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
                    'owner': {'sdtype': 'text'},
                },
                'branches': {
                    'id': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'text'},
                }
            },
            'relationships': [{
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'chil_foreign_key': 'branch_id',
            }]
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
            - ``instance`` now contains ``instance._tables`` and ``instance._relationships``.
            - ``SingleTableMetadata._load_from_dict`` has been called.
        """
        # Setup
        multitable_metadata = {
            'tables': {
                'accounts': {
                    'id': {'sdtype': 'numerical'},
                    'branch_id': {'sdtype': 'numerical'},
                    'amount': {'sdtype': 'numerical'},
                    'start_date': {'sdtype': 'datetime'},
                    'owner': {'sdtype': 'text'},
                },
                'branches': {
                    'id': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'text'},
                }
            },
            'relationships': [{
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'chil_foreign_key': 'branch_id',
            }]
        }

        single_table_accounts = object()
        single_table_branches = object()
        mock_singletablemetadata._load_from_dict.side_effect = [
            single_table_accounts,
            single_table_branches
        ]

        instance = MultiTableMetadata()

        # Run
        instance._set_metadata_dict(multitable_metadata)

        # Assert
        assert instance._tables == {
            'accounts': single_table_accounts,
            'branches': single_table_branches
        }

        assert instance._relationships == [{
            'parent_table_name': 'accounts',
            'parent_primary_key': 'id',
            'child_table_name': 'branches',
            'chil_foreign_key': 'branch_id',
        }]

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test__load_from_dict(self, mock_singletablemetadata):
        """Test that ``_load_from_dict`` returns a instance of ``MultiTableMetadata``.

        Test that when calling the ``_load_from_dict`` method a new instance with the passed
        python ``dict`` details should be created.

        Setup:
            - A dict representing a ``MultiTableMetadata``.

        Mock:
            - Mock ``SingleTableMetadata`` from ``sdv.metadata.multi_table``

        Output:
            - ``instance`` that contains ``instance._tables`` and ``instance._relationships``.

        Side Effects:
            - ``SingleTableMetadata._load_from_dict`` has been called.
        """
        # Setup
        multitable_metadata = {
            'tables': {
                'accounts': {
                    'id': {'sdtype': 'numerical'},
                    'branch_id': {'sdtype': 'numerical'},
                    'amount': {'sdtype': 'numerical'},
                    'start_date': {'sdtype': 'datetime'},
                    'owner': {'sdtype': 'text'},
                },
                'branches': {
                    'id': {'sdtype': 'numerical'},
                    'name': {'sdtype': 'text'},
                }
            },
            'relationships': [{
                'parent_table_name': 'accounts',
                'parent_primary_key': 'id',
                'child_table_name': 'branches',
                'child_foreign_key': 'branch_id',
            }]
        }

        single_table_accounts = object()
        single_table_branches = object()
        mock_singletablemetadata._load_from_dict.side_effect = [
            single_table_accounts,
            single_table_branches
        ]

        # Run
        instance = MultiTableMetadata._load_from_dict(multitable_metadata)

        # Assert
        assert instance._tables == {
            'accounts': single_table_accounts,
            'branches': single_table_branches
        }

        assert instance._relationships == [{
            'parent_table_name': 'accounts',
            'parent_primary_key': 'id',
            'child_table_name': 'branches',
            'child_foreign_key': 'branch_id',
        }]

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

    def get_metadata(self):
        """Set the tables and relationships for metadata."""
        metadata = {}
        metadata['tables'] = {
            'users': {
                'columns': {
                    'id': {'sdtype': 'numerical'},
                    'country': {'sdtype': 'categorical'}
                },
                'primary_key': 'id'
            },
            'payments': {
                'columns': {
                    'payment_id': {'sdtype': 'numerical'},
                    'user_id': {'sdtype': 'numerical'},
                    'date': {'sdtype': 'datetime'}
                },
                'primary_key': 'payment_id'
            },
            'sessions': {
                'columns': {
                    'session_id': {'sdtype': 'numerical'},
                    'user_id': {'sdtype': 'numerical'},
                    'device': {'sdtype': 'categorical'}
                },
                'primary_key': 'session_id'
            },
            'transactions': {
                'columns': {
                    'transaction_id': {'sdtype': 'numerical'},
                    'session_id': {'sdtype': 'numerical'},
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

        return MultiTableMetadata._load_from_dict(metadata)

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
            '{payments|payment_id : numerical\\luser_id : numerical\\ldate : datetime\\l|'
            'Primary key: payment_id\\lForeign key (users): user_id\\l}'
        )
        expected_sessions_label = (
            '{sessions|session_id : numerical\\luser_id : numerical\\ldevice : categorical\\l|'
            'Primary key: session_id\\lForeign key (users): user_id\\l}'
        )
        expected_transactions_label = (
            '{transactions|transaction_id : numerical\\lsession_id : numerical\\ltimestamp : '
            'datetime\\l|Primary key: transaction_id\\lForeign key (sessions): session_id\\l}'
        )
        expected_nodes = {
            'users': '{users|id : numerical\\lcountry : categorical\\l|Primary key: id\\l\\l}',
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
            '{payments|payment_id : numerical\\luser_id : numerical\\ldate : datetime\\l|'
            'Primary key: payment_id\\lForeign key (users): user_id\\l}'
        )
        expected_sessions_label = (
            '{sessions|session_id : numerical\\luser_id : numerical\\ldevice : categorical\\l|'
            'Primary key: session_id\\lForeign key (users): user_id\\l}'
        )
        expected_transactions_label = (
            '{transactions|transaction_id : numerical\\lsession_id : numerical\\ltimestamp : '
            'datetime\\l|Primary key: transaction_id\\lForeign key (sessions): session_id\\l}'
        )
        expected_nodes = {
            'users': '{users|id : numerical\\lcountry : categorical\\l|Primary key: id\\l\\l}',
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
        metadata._tables = {'table': table}

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
        with pytest.raises(ValueError, match=error_message):
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
        metadata._tables = {'table': table}

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
        with pytest.raises(ValueError, match=error_message):
            metadata.update_column('table', 'column', sdtype='numerical', pii=False)

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_detect_table_from_csv(self, single_table_mock):
        """Test the ``detect_table_from_csv`` method.

        If the table does not already exist, a ``SingleTableMetadata`` instance
        should be created and call the ``detect_from_csv`` method.

        Setup:
            - Mock the ``SingleTableMetadata`` class.

        Input:
            - Table name.
            - Path.

        Side effect:
            - Table should be added to ``self._tables``.
        """
        # Setup
        metadata = MultiTableMetadata()

        # Run
        metadata.detect_table_from_csv('table', 'path.csv')

        # Assert
        single_table_mock.return_value.detect_from_csv.assert_called_once_with('path.csv')
        assert metadata._tables == {'table': single_table_mock.return_value}

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
        metadata._tables = {'table': Mock()}

        # Run
        error_message = (
            "Metadata for table 'table' already exists. Specify a new table name or "
            'create a new MultiTableMetadata object for other data sources.'
        )
        with pytest.raises(InvalidMetadataError, match=error_message):
            metadata.detect_table_from_csv('table', 'path.csv')

    @patch('sdv.metadata.multi_table.SingleTableMetadata')
    def test_detect_table_from_dataframe(self, single_table_mock):
        """Test the ``detect_table_from_dataframe`` method.

        If the table does not already exist, a ``SingleTableMetadata`` instance
        should be created and call the ``detect_from_dataframe`` method.

        Setup:
            - Mock the ``SingleTableMetadata`` class.

        Input:
            - Table name.
            - Dataframe.

        Side effect:
            - Table should be added to ``self._tables``.
        """
        # Setup
        metadata = MultiTableMetadata()
        data = pd.DataFrame()

        # Run
        metadata.detect_table_from_dataframe('table', data)

        # Assert
        single_table_mock.return_value.detect_from_dataframe.assert_called_once_with(data)
        assert metadata._tables == {'table': single_table_mock.return_value}

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
        metadata._tables = {'table': Mock()}

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
            - ``ValueError`` if the table name is not in the metadata.
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata._tables = {'table1': 'val', 'table2': 'val'}

        # Run
        metadata._validate_table_exists('table1')

        # Assert
        err_msg = re.escape("Unknown table name ('table3').")
        with pytest.raises(ValueError, match=err_msg):
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
        metadata._tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        metadata.set_primary_key('table1', 'col')

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata._tables['table1'].set_primary_key.assert_called_once_with('col')

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
        metadata._tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        warn_msg = 'Sequential modeling is not yet supported on SDV Multi Table models.'
        with pytest.warns(match=warn_msg):
            metadata.set_sequence_key('table1', 'col')

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata._tables['table1'].set_sequence_key.assert_called_once_with('col')

    def test_set_alternate_keys(self):
        """Test ``set_alternate_keys``.

        The method should validate the table exists and call
        ``SingleTableMetadata.set_alternate_keys``.

        Setup:
            - Instantiate ``MultiTableMetadata`` with some ``_tables``.
            - Mock ``_validate_table_exists``.

        Input:
            - Table name
            - List of column names
        """
        # Setup
        metadata = MultiTableMetadata()
        metadata._tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        metadata.set_alternate_keys('table1', ['col1', 'col2'])

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata._tables['table1'].set_alternate_keys.assert_called_once_with(['col1', 'col2'])

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
        metadata._tables = {'table1': Mock(), 'table2': 'val'}
        metadata._validate_table_exists = Mock()

        # Run
        warn_msg = 'Sequential modeling is not yet supported on SDV Multi Table models.'
        with pytest.warns(match=warn_msg):
            metadata.set_sequence_index('table1', 'col')

        # Assert
        metadata._validate_table_exists.assert_called_once_with('table1')
        metadata._tables['table1'].set_sequence_index.assert_called_once_with('col')
