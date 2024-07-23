import re
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sdv.metadata.metadata import Metadata
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
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
        instance = Metadata.load_from_json('filepath.json')

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
            - A dict representing a ``SingleTableMetadata``.

        Mock:
            - Mock ``SingleTableMetadata`` from ``sdv.metadata.multi_table``

        Side Effects:
            - ``SingleTableMetadata.load_from_dict`` has been called.
        """
        # Setup
        multitable_metadata = {
            'columns': {'my_column': 'value'},
            'primary_key': 'pk',
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }

        instance = Metadata()

        # Run
        instance._set_metadata_dict(multitable_metadata)

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

    @patch('sdv.metadata.metadata.read_json')
    @patch('sdv.metadata.metadata.Path')
    def test_load_from_json(self, mock_path, mock_read_json):
        """Test ``load_from_json`` on Metadata."""
        # Setup
        mock_read_json.return_value = {'tables': {}}
        filepath = 'test_path.json'
        mock_path.stem.return_value = 'test_path'

        # Run
        metadata = Metadata.load_from_json(filepath)

        # Assert
        mock_path.assert_called_once_with(filepath)
        assert isinstance(metadata, Metadata)
        mock_read_json.assert_called_once_with(filepath)

    def test_set_metadata_dict_single_table(self):
        """Test ``set_metadata_dict`` works for single tables."""
        # Setup
        metadata_dict = {'columns': {}}
        metadata = Metadata()

        # Run
        metadata._set_metadata_dict(metadata_dict, 'test_table')

        # Assert
        assert 'test_table' in metadata.tables
        assert isinstance(metadata.tables['test_table'], SingleTableMetadata)

    def test_set_metadata_dict_multi_table(self):
        """Test ``set_metadata_dict`` works for multiple tables."""
        # Setup
        metadata_dict = {'tables': {'table1': {'columns': {}}, 'table2': {'columns': {}}}}
        metadata = Metadata()

        # Run
        metadata._set_metadata_dict(metadata_dict)

        # Assert
        assert 'table1' in metadata.tables
        assert 'table2' in metadata.tables

    def test__get_table_or_default(self):
        """Test ``_get_table_or_default`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        assert metadata._get_table_or_default('table1') == metadata.tables['table1']

    def test__get_table_or_default_no_table_name(self):
        """Test ``_get_table_or_default`` method falls back to a single table."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        assert metadata._get_table_or_default() == metadata.tables['table1']

    def test__get_table_or_default_multiple_tables(self):
        """Confirm that ``_get_table_or_default_multiple_tables`` raises an error."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()
        error_msg = re.escape(
            'This metadata contains more than one table. Please provide a table name in the method.'
        )

        # Run and Assert
        with pytest.raises(ValueError, match=error_msg):
            metadata._get_table_or_default()

    def test__get_table_name_error_too_many_tables_no_arg(self):
        """Test ``_get_table_name`` with more than one table but table argument."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()
        error_msg = re.escape(
            'This metadata contains more than one table. Please provide a table name in the method.'
        )

        # Run and Assert
        with pytest.raises(ValueError, match=error_msg):
            metadata._get_table_name()

    def test_get_column_relationships(self):
        """Test ``get_column_relationships`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock.column_relationships = ['relationship1']
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_column_relationships('table1') == ['relationship1']
        assert metadata.get_column_relationships() == ['relationship1']

    def test_get_primary_key(self):
        """Test ``get_primary_key`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock.primary_key = 'id'
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_primary_key('table1') == 'id'
        assert metadata.get_primary_key() == 'id'

    def test_get_alternate_keys(self):
        """Test ``get_alternate_keys`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock.alternate_keys = ['alt_key']
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_alternate_keys('table1') == ['alt_key']
        assert metadata.get_alternate_keys() == ['alt_key']

    def test_get_columns(self):
        """Test ``get_columns`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock.columns = {'col1': 'type1'}
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_columns('table1') == {'col1': 'type1'}
        assert metadata.get_columns() == {'col1': 'type1'}

    def test_validate_data(self):
        """Test ``validate_data`` works with a dataframe."""
        # Setup
        data = pd.DataFrame({'col1': [1, 2, 3]})
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'validate_data') as mock_validate:
            metadata.validate_data(data)
            mock_validate.assert_called_once()

        assert metadata.METADATA_SPEC_VERSION == 'V1'

    def test_validate_data_dict(self):
        """Test ``validate_data`` works with a dict of dataframes."""
        # Setup
        data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3]}),
            'table2': pd.DataFrame({'col1': [4, 5, 6]}),
        }
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'validate_data') as mock_validate:
            metadata.validate_data(data)
            mock_validate.assert_called_once()

    def test_add_column_default(self):
        """Test ``add_column`` works with a single table."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'add_column') as mock_add_column:
            metadata.add_column('col1', sdtype='integer')
            mock_add_column.assert_called_once_with('table1', 'col1', sdtype='integer')

    def test_add_column(self):
        """Test ``add_column`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'add_column') as mock_add_column:
            metadata.add_column('col1', 'table2', sdtype='integer')
            mock_add_column.assert_called_once_with('table2', 'col1', sdtype='integer')

    def test_set_sequence_key(self):
        """Test ``set_sequence_key`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'set_sequence_key') as mock_set_sequence_key:
            metadata.set_sequence_key('seq_col', 'table2')
            mock_set_sequence_key.assert_called_once_with('table2', 'seq_col')

    def test_set_sequence_key_default(self):
        """Test ``set_sequence_key`` method with a single table."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'set_sequence_key') as mock_set_sequence_key:
            metadata.set_sequence_key('seq_col')
            mock_set_sequence_key.assert_called_once_with('table1', 'seq_col')

    def test_add_alternate_keys(self):
        """Test ``set_sequence_key`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'add_alternate_keys') as mock_add_alternate_keys:
            metadata.add_alternate_keys(['alt_key'], 'table2')
            mock_add_alternate_keys.assert_called_once_with('table2', ['alt_key'])

    def test_add_alternate_keys_default(self):
        """Test ``set_sequence_key`` method with a single table."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'add_alternate_keys') as mock_add_alternate_keys:
            metadata.add_alternate_keys(['alt_key'])
            mock_add_alternate_keys.assert_called_once_with('table1', ['alt_key'])

    def test_get_primary_and_alternate_keys(self):
        """Test ``get_primary_and_alternate_keys`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock._get_primary_and_alternate_keys.return_value = {'id', 'alt_key'}
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_primary_and_alternate_keys('table1') == {'id', 'alt_key'}
        assert metadata.get_primary_and_alternate_keys() == {'id', 'alt_key'}

    def test_get_set_of_sequence_keys(self):
        """Test ``get_set_of_sequence_keys`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock._get_set_of_sequence_keys.return_value = {'seq_key'}
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_set_of_sequence_keys('table1') == {'seq_key'}
        assert metadata.get_set_of_sequence_keys() == {'seq_key'}

    def test_set_primary_key_default(self):
        """Test ``set_primary_key`` method with a single table."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'set_primary_key') as mock_set_primary_key:
            metadata.set_primary_key('id')
            mock_set_primary_key.assert_called_once_with('table1', 'id')

    def test_set_primary_key(self):
        """Test ``set_primary_key`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'set_primary_key') as mock_set_primary_key:
            metadata.set_primary_key('id', 'table2')
            mock_set_primary_key.assert_called_once_with('table2', 'id')

    def test_detect_from_dataframes(self):
        """Test ``detect_from_dataframes`` method with a Dataframe."""
        # Setup
        data = {'table1': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = Metadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'detect_from_dataframes') as mock_detect:
            metadata.detect_from_dataframes(data)
            mock_detect.assert_called_once_with(data)

    def test_detect_from_dataframes_dict(self):
        """Test ``detect_from_dataframes`` method with a dict of Dataframes."""
        # Setup
        data = {
            'table1': pd.DataFrame({'col1': [1, 2, 3]}),
            'table2': pd.DataFrame({'col1': [4, 5, 6]}),
        }
        metadata = Metadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'detect_from_dataframes') as mock_detect:
            metadata.detect_from_dataframes(data)
            mock_detect.assert_called_once_with(data)

    def test_get_sequence_key(self):
        """Test ``get_sequence_key`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock.sequence_key = 'seq_key'
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_sequence_key('table1') == 'seq_key'
        assert metadata.get_sequence_key() == 'seq_key'

    def test_get_sequence_key_none(self):
        """Test ``get_sequence_key`` called with None."""
        # Setup
        metadata = Metadata()

        # Run and Assert
        assert metadata.get_sequence_key() is None

    def test_get_sequence_index(self):
        """Test ``get_sequence_index`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock.sequence_index = 'seq_index'
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_sequence_index('table1') == 'seq_index'
        assert metadata.get_sequence_index() == 'seq_index'

    def test_get_sequence_index_none(self):
        """Test ``get_sequence_index`` works with None."""
        # Setup
        metadata = Metadata()

        # Run and Assert
        assert metadata.get_sequence_index() is None

    def test_get_valid_column_relationships(self):
        """Test ``get_valid_column_relationships`` works with a table and default table."""
        # Setup
        table_mock = MagicMock()
        table_mock._valid_column_relationships = ['relationship1']
        metadata = Metadata()
        metadata.tables['table1'] = table_mock

        # Run and Assert
        assert metadata.get_valid_column_relationships('table1') == ['relationship1']
        assert metadata.get_valid_column_relationships() == ['relationship1']

    def test_add_column_relationship(self):
        """Test ``add_column_relationship`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(
            MultiTableMetadata, 'add_column_relationship'
        ) as mock_add_column_relationship:
            metadata.add_column_relationship('type', ['col1', 'col2'], 'table1')
            mock_add_column_relationship.assert_called_once_with('table1', 'type', ['col1', 'col2'])

    def test_add_column_relationship_default(self):
        """Test ``add_column_relationship`` method with a single table."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(
            MultiTableMetadata, 'add_column_relationship'
        ) as mock_add_column_relationship:
            metadata.add_column_relationship('type', ['col1', 'col2'])
            mock_add_column_relationship.assert_called_once_with('table1', 'type', ['col1', 'col2'])

    def test_update_column(self):
        """Test ``update_column`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()
        metadata.tables['table2'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'update_column') as mock_update_column:
            metadata.update_column('col1', 'table2', sdtype='integer')
            mock_update_column.assert_called_once_with('table2', 'col1', sdtype='integer')

    def test_update_column_default(self):
        """Test ``update_column`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables['table1'] = SingleTableMetadata()

        # Run and Assert
        with patch.object(MultiTableMetadata, 'update_column') as mock_update_column:
            metadata.update_column('col1', sdtype='integer')
            mock_update_column.assert_called_once_with('table1', 'col1', sdtype='integer')
