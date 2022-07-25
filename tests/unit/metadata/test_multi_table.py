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
            - Mock ``SingleTableMetadata`` from ``sdv.metadata.multi_tbale``

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
            - Mock ``SingleTableMetadata`` from ``sdv.metadata.multi_tbale``

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
                'chil_foreign_key': 'branch_id',
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
            'chil_foreign_key': 'branch_id',
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
