"""Test Multi Table Metadata."""

from unittest.mock import Mock, patch

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
        """Set the tables and relationships for metadata"""
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
