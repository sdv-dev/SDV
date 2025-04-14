import re
from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
import pytest

from sdv.errors import InvalidDataError
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata import Metadata
from sdv.metadata.single_table import SingleTableMetadata
from tests.utils import DataFrameMatcher, get_multi_table_data, get_multi_table_metadata


@pytest.fixture
def metadata_instance():
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


class TestMetadataClass:
    """Test ``Metadata`` class."""

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

    @patch('sdv.metadata.metadata.read_json')
    def test_load_from_json_single_table(self, mock_read_json):
        """Test the ``load_from_json`` method for single table metadata.

        Mock:
            - Mock the ``read_json`` function in order to return a custom json.

        Input:
            - String representing a filepath.

        Output:
            - ``SingleTableMetadata`` instance with the custom configuration from the ``json``
                file (``json.load`` return value)
        """
        # Setup
        mock_read_json.return_value = {
            'columns': {'animals': {'type': 'categorical'}},
            'primary_key': 'animals',
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        }
        warning_message = (
            'You are loading an older SingleTableMetadata object. This will be converted'
            " into the new Metadata object with a placeholder table name ('{}')."
            ' Please save this new object for future usage.'
        )

        expected_warning_with_table_name = re.escape(warning_message.format('filepath'))
        expected_warning_without_table_name = re.escape(warning_message.format('table'))

        # Run
        with pytest.warns(UserWarning, match=expected_warning_with_table_name):
            instance_with_table_name = Metadata.load_from_json('filepath.json', 'filepath')
        with pytest.warns(UserWarning, match=expected_warning_without_table_name):
            instance_without_table_name = Metadata.load_from_json('filepath.json')

        # Assert
        mock_read_json.assert_has_calls([call('filepath.json'), call('filepath.json')])
        table_name_to_instance = {
            'filepath': instance_with_table_name,
            'table': instance_without_table_name,
        }
        for table_name, instance in table_name_to_instance.items():
            assert list(instance.tables.keys()) == [table_name]
            assert instance.tables[table_name].columns == {'animals': {'type': 'categorical'}}
            assert instance.tables[table_name].primary_key == 'animals'
            assert instance.tables[table_name].sequence_key is None
            assert instance.tables[table_name].alternate_keys == []
            assert instance.tables[table_name].sequence_index is None
            assert instance.tables[table_name]._version == 'SINGLE_TABLE_V1'

    @patch('builtins.open', new_callable=mock_open)
    @patch('sdv.metadata.utils.Path')
    @patch('sdv.metadata.utils.json')
    def test_load_from_json_multi_table(self, mock_json, mock_path, mock_file):
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
        assert list(instance.tables.keys()) == ['table']
        assert instance.tables['table'].columns == {'my_column': 'value'}
        assert instance.tables['table'].primary_key == 'pk'
        assert instance.tables['table'].sequence_key is None
        assert instance.tables['table'].alternate_keys == []
        assert instance.tables['table'].sequence_index is None
        assert instance.tables['table']._version == 'SINGLE_TABLE_V1'

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
        assert list(instance.tables.keys()) == ['table']
        assert instance.tables['table'].columns == {'1': 'value'}
        assert instance.tables['table'].primary_key == 'pk'
        assert instance.tables['table'].sequence_key is None
        assert instance.tables['table'].alternate_keys == []
        assert instance.tables['table'].sequence_index is None

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

    def test__set_metadata_single_table(self, metadata_instance):
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
        assert instance.tables['table'].columns == {'my_column': 'value'}
        assert instance.tables['table'].primary_key == 'pk'
        assert instance.tables['table'].alternate_keys == []
        assert instance.tables['table'].sequence_key is None
        assert instance.tables['table'].sequence_index is None
        assert instance.tables['table'].METADATA_SPEC_VERSION == 'SINGLE_TABLE_V1'

    def test_validate(self, metadata_instance):
        """Test the method ``validate``.

        Test that when a valid ``Metadata`` has been provided no errors are being raised.

        Setup:
            - Instance of ``Metadata`` with all valid tables and relationships.
        """
        # Run
        metadata_instance.validate()

    def test_validate_no_relationships(self, metadata_instance):
        """Test the method ``validate`` without relationships.

        Test that when a valid ``Metadata`` has been provided no errors are being raised.

        Setup:
            - Instance of ``Metadata`` with all valid tables and no relationships.
        """
        # Setup
        metadata_no_relationships = metadata_instance.to_dict()
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

    def test_validate_table(self):
        """Test the ``validate_table``method."""
        # Setup
        metadata_multi_table = get_multi_table_metadata()
        metadata_single_table = Metadata.load_from_dict(
            metadata_multi_table.to_dict()['tables']['nesreca'], 'nesreca'
        )
        table = get_multi_table_data()['nesreca']

        expected_error_wrong_name = re.escape(
            'The provided data does not match the metadata:\n'
            "The provided data is missing the tables {'nesreca'}."
        )
        expected_error_mutli_table = re.escape(
            'Metadata contains more than one table, please specify the `table_name` to validate.'
        )

        # Run and Assert
        metadata_single_table.validate_table(table)
        metadata_single_table.validate_table(table, 'nesreca')
        with pytest.raises(InvalidDataError, match=expected_error_wrong_name):
            metadata_single_table.validate_table(table, 'wrong_name')
        with pytest.raises(InvalidMetadataError, match=expected_error_mutli_table):
            metadata_multi_table.validate_table(table)

    @patch('sdv.metadata.metadata.Metadata')
    def test_detect_from_dataframes(self, mock_metadata):
        """Test ``detect_from_dataframes``.

        Expected to call ``detect_table_from_dataframe`` for each table name and dataframe
        in the input.
        """
        # Setup
        mock_metadata.detect_table_from_dataframe = Mock()
        mock_metadata._detect_relationships = Mock()
        guests_table = pd.DataFrame()
        hotels_table = pd.DataFrame()
        data = {'guests': guests_table, 'hotels': hotels_table}

        # Run
        metadata = Metadata.detect_from_dataframes(data)

        # Assert
        mock_metadata.return_value.detect_table_from_dataframe.assert_any_call(
            'guests', guests_table, True, 'primary_only'
        )
        mock_metadata.return_value.detect_table_from_dataframe.assert_any_call(
            'hotels', hotels_table, True, 'primary_only'
        )
        mock_metadata.return_value._detect_relationships.assert_called_once_with(
            data, 'column_name_match'
        )
        assert metadata == mock_metadata.return_value

    @patch('sdv.metadata.metadata.Metadata')
    def test_detect_from_dataframes_infer_keys_none(self, mock_metadata):
        """Test ``detect_from_dataframes`` with infer_keys set to None."""
        # Setup
        mock_metadata.detect_table_from_dataframe = Mock()
        mock_metadata._detect_relationships = Mock()
        guests_table = pd.DataFrame()
        hotels_table = pd.DataFrame()
        data = {'guests': guests_table, 'hotels': hotels_table}

        # Run
        metadata = Metadata.detect_from_dataframes(data, infer_sdtypes=False, infer_keys=None)

        # Assert
        mock_metadata.return_value.detect_table_from_dataframe.assert_any_call(
            'guests', guests_table, False, None
        )
        mock_metadata.return_value.detect_table_from_dataframe.assert_any_call(
            'hotels', hotels_table, False, None
        )
        mock_metadata.return_value._detect_relationships.assert_not_called()
        assert metadata == mock_metadata.return_value

    def test_detect_from_dataframes_bad_foreign_key_inference_algorithm(self):
        """Test error message when 'foreign_key_inference_algorithm' is invalid."""
        # Setup
        guests_table = pd.DataFrame()
        hotels_table = pd.DataFrame()
        data = {'guests': guests_table, 'hotels': hotels_table}

        # Run and Assert
        msg = "'foreign_key_inference_algorithm' must be 'column_name_match'"
        with pytest.raises(ValueError, match=msg):
            Metadata.detect_from_dataframes(
                data=data,
                infer_sdtypes=False,
                infer_keys=None,
                foreign_key_inference_algorithm='fake',
            )

    @patch('sdv.metadata.metadata.Metadata')
    def test_detect_from_dataframes_infer_keys_primary_only(self, mock_metadata):
        """Test ``detect_from_dataframes`` with infer_keys set to 'primary_only'."""
        # Setup
        mock_metadata.detect_table_from_dataframe = Mock()
        mock_metadata._detect_relationships = Mock()
        guests_table = pd.DataFrame()
        hotels_table = pd.DataFrame()
        data = {'guests': guests_table, 'hotels': hotels_table}

        # Run
        metadata = Metadata.detect_from_dataframes(
            data, infer_sdtypes=False, infer_keys='primary_only'
        )

        # Assert
        mock_metadata.return_value.detect_table_from_dataframe.assert_any_call(
            'guests', guests_table, False, 'primary_only'
        )
        mock_metadata.return_value.detect_table_from_dataframe.assert_any_call(
            'hotels', hotels_table, False, 'primary_only'
        )
        mock_metadata.return_value._detect_relationships.assert_not_called()
        assert metadata == mock_metadata.return_value

    def test_detect_from_dataframes_bad_input_data(self):
        """Test that an error is raised if the dictionary contains something other than DataFrames.

        If the data contains values that aren't pandas.DataFrames, it should error.
        """
        # Setup
        data = {'guests': Mock(), 'hotels': Mock()}

        # Run and Assert
        expected_message = 'The provided dictionary must contain only pandas DataFrame objects.'
        with pytest.raises(ValueError, match=expected_message):
            Metadata.detect_from_dataframes(data)

    def test_detect_from_dataframes_bad_input_infer_sdtypes(self):
        """Test that an error is raised if the infer_sdtypes is not a boolean."""
        # Setup
        data = {'guests': pd.DataFrame(), 'hotels': pd.DataFrame()}
        infer_sdtypes = 'not_a_boolean'

        # Run and Assert
        expected_message = "'infer_sdtypes' must be a boolean value."
        with pytest.raises(ValueError, match=expected_message):
            Metadata.detect_from_dataframes(data, infer_sdtypes=infer_sdtypes)

    def test_detect_from_dataframes_bad_input_infer_keys(self):
        """Test that an error is raised if the infer_keys is not a correct string."""
        # Setup
        data = {'guests': pd.DataFrame(), 'hotels': pd.DataFrame()}
        infer_keys = 'incorrect_string'

        # Run and Assert
        expected_message = re.escape(
            "'infer_keys' must be one of: 'primary_and_foreign', 'primary_only', None."
        )
        with pytest.raises(ValueError, match=expected_message):
            Metadata.detect_from_dataframes(data, infer_keys=infer_keys)

    @patch('sdv.metadata.metadata.Metadata')
    def test_detect_from_dataframe(self, mock_metadata):
        """Test that the method calls the detection method and returns the metadata.

        Expected to call ``detect_table_from_dataframe`` for the dataframe.
        """
        # Setup
        mock_metadata.detect_table_from_dataframe = Mock()
        data = pd.DataFrame()

        # Run
        metadata = Metadata.detect_from_dataframe(data)

        # Assert
        mock_metadata.return_value.detect_table_from_dataframe.assert_any_call(
            Metadata.DEFAULT_SINGLE_TABLE_NAME, DataFrameMatcher(data), True, 'primary_only'
        )
        assert metadata == mock_metadata.return_value

    def test_detect_from_dataframe_raises_error_if_not_dataframe(self):
        """Test that the method raises an error if data isn't a DataFrame."""
        # Run and assert
        expected_message = 'The provided data must be a pandas DataFrame object.'
        with pytest.raises(ValueError, match=expected_message):
            Metadata.detect_from_dataframe(Mock())

    def test_detect_from_dataframe_bad_input_infer_sdtypes(self):
        """Test that an error is raised if the infer_sdtypes is not a boolean."""
        # Setup
        data = pd.DataFrame()
        infer_sdtypes = 'not_a_boolean'

        # Run and Assert
        expected_message = "'infer_sdtypes' must be a boolean value."
        with pytest.raises(ValueError, match=expected_message):
            Metadata.detect_from_dataframe(data, infer_sdtypes=infer_sdtypes)

    def test_detect_from_dataframe_bad_input_infer_keys(self):
        """Test that an error is raised if the infer_keys is not a correct string."""
        # Setup
        data = pd.DataFrame()
        infer_keys = 'primary_and_foreign'

        # Run and Assert
        expected_message = re.escape("'infer_keys' must be one of: 'primary_only', None.")
        with pytest.raises(ValueError, match=expected_message):
            Metadata.detect_from_dataframe(data, infer_keys=infer_keys)

    def test__handle_table_name(self):
        """Test the ``_handle_table_name`` method."""
        # Setup
        metadata = Metadata()
        metadata.tables = ['table_name']
        expected_error = re.escape(
            'Metadata contains more than one table, please specify the `table_name`.'
        )

        # Run
        result_none = metadata._handle_table_name(None)
        result_table_1 = metadata._handle_table_name('table_1')
        metadata.tables = ['table_1', 'table_2']
        with pytest.raises(ValueError, match=expected_error):
            metadata._handle_table_name(None)

        result_table_2 = metadata._handle_table_name('table_2')

        # Assert
        assert result_none == 'table_name'
        assert result_table_1 == 'table_1'
        assert result_table_2 == 'table_2'

    params = [
        ('update_column', ['column_name']),
        ('update_columns', ['column_names']),
        ('update_columns_metadata', ['column_metadata']),
        ('add_column', ['column_name']),
        ('set_primary_key', ['column_name']),
        ('remove_primary_key', []),
        ('add_column_relationship', ['relationship_type', 'column_names']),
        ('add_alternate_keys', ['column_names']),
        ('get_column_names', []),
    ]

    @pytest.mark.parametrize('method, args', params)
    def test_update_methods(self, method, args):
        """Test that all update methods call the superclass method with the resolved arguments."""
        # Setup
        metadata = Metadata()
        metadata._handle_table_name = Mock(return_value='table_name')
        superclass = Metadata.__bases__[0]

        with patch.object(superclass, method) as mock_super_method:
            # Run
            getattr(metadata, method)(*args, 'table_name')

            # Assert
            metadata._handle_table_name.assert_called_once_with('table_name')
            mock_super_method.assert_called_once_with('table_name', *args)

    def test_get_table_metadata(self, metadata_instance):
        """Test that `get_table_metadata` return a `Metadata` instance."""
        # Run
        metadata = metadata_instance.get_table_metadata('users')

        # Assert
        assert isinstance(metadata, Metadata)

    def test_accessing_tables_returns_single_table_metadata(self, metadata_instance):
        """Test that accessing items from `metadata.tables` returns `SingleTableMetadata`."""
        # Run
        single_table_metadata = metadata_instance.tables['users']

        # Assert
        assert isinstance(single_table_metadata, SingleTableMetadata)

    def test__handle_table_name_with_empty_tables(self):
        """Test that the proper `ValueError` is raised when there are no `tables`."""
        # Setup
        instance = Metadata()

        # Run and Assert
        error_msg = 'Metadata does not contain any tables. No columns can be added.'
        with pytest.raises(ValueError, match=error_msg):
            instance._handle_table_name(None)

    @patch('sdv.metadata.metadata.Metadata.load_from_dict')
    def test_anonymize(self, mock_load_from_dict):
        """Test that the `anonymize` method."""
        # Setup
        metadata = Metadata()
        metadata._get_anonymized_dict = Mock(return_value={})
        metadata.load_from_dict = Mock()

        # Run
        metadata.anonymize()

        # Assert
        metadata._get_anonymized_dict.assert_called_once()
        mock_load_from_dict.assert_called_once_with({})

    @patch('sdv.metadata.multi_table.visualize_graph')
    def test_visualize_with_sequence_key_and_index(self, visualize_graph_mock):
        """Test the ``visualize`` method with sequence key and index"""
        # Setup
        metadata_dict = {
            'tables': {
                'nasdaq100_2019': {
                    'columns': {
                        'Symbol': {'sdtype': 'id', 'regex_format': '[A-Z]{4}'},
                        'Date': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    },
                    'sequence_index': 'Date',
                    'sequence_key': 'Symbol',
                }
            },
            'relationships': [],
            'METADATA_SPEC_VERSION': 'V1',
        }
        metadata = Metadata.load_from_dict(metadata_dict)

        # Run
        metadata.visualize('full', True)

        # Assert
        expected_label = (
            '{nasdaq100_2019|Symbol : id\\lDate : datetime\\l|Primary key: None\\l'
            'Sequence key: Symbol\\lSequence index: Date\\l}'
        )
        expected_nodes = {
            'nasdaq100_2019': expected_label,
        }
        visualize_graph_mock.assert_called_once_with(expected_nodes, [], None)
