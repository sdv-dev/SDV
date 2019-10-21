from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.metadata import Metadata, _parse_dtypes, _read_csv_dtypes, load_csv


def test__read_csv_dtypes():
    """Test read csv data types"""
    # Run
    table_meta = {
        'fields': {
            'a_field': {
                'type': 'categorical',
                'subtype': 'categorical'
            },
            'b_field': {
                'type': 'categorical',
                'subtype': 'bool'
            },
            'c_field': {
                'type': 'datetime',
            }
        }
    }

    result = _read_csv_dtypes(table_meta)

    # Asserts
    expected = {'a_field': str}

    assert result == expected


def test__parse_dtypes():
    """Test parse data types"""
    # Run
    data = pd.DataFrame({
        'a_field': ['1996-10-17', '1965-05-23'],
        'b_field': ['7', '14']
    })
    table_meta = {
        'fields': {
            'a_field': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
            'b_field': {
                'type': 'number',
                'subtype': 'integer'
            }
        }
    }

    result = _parse_dtypes(data, table_meta)

    # Asserts
    expected = pd.DataFrame({
        'a_field': pd.to_datetime(['1996-10-17', '1965-05-23'], format='%Y-%m-%d'),
        'b_field': [7, 14]
    })

    pd.testing.assert_frame_equal(result, expected)


def test_load_csv(tmp_path):
    """Test load csv"""
    # Setup
    metadata = tmp_path / 'load_csv.csv'
    metadata.write_text('some, fake, data')

    root_path = str(tmp_path)
    table_meta = {
        'path': 'load_csv.csv'
    }

    # Run and asserts
    with patch('sdv.metadata._read_csv_dtypes') as mock_read, \
            patch('sdv.metadata._parse_dtypes') as mock_parse:

        mock_read.return_value = []

        load_csv(root_path, table_meta)

        mock_read.assert_called_once_with({'path': 'load_csv.csv'})
        assert mock_parse.call_count == 1


def test___init__default_metadata_string(tmp_path):
    """Test create Metadata instance default with a string"""
    # Run and asserts
    metadata_path = tmp_path / 'meta.json'
    metadata_path.write_text('{"some": "meta"}')

    with patch('sdv.metadata.Metadata._dict_metadata') as mock_meta, \
            patch('sdv.metadata.Metadata._get_relationships') as mock_relationships:

        metadata = Metadata(str(metadata_path), root_path=None)
        mock_meta.assert_called_once_with({'some': 'meta'})
        mock_relationships.assert_called_once_with()

    assert metadata.root_path == str(tmp_path)  # We ensure that root_path is the tmp file.
    assert metadata._hyper_transformers == dict()


class TestMetadata(TestCase):
    """Test Metadata class."""

    def test__get_relationships(self):
        """Test get relationships"""
        # Setup
        _metadata = {
            'tables': {
                'test': {
                    'use': True,
                    'name': 'test',
                    'fields': {
                        'test_field': {
                            'ref': {'table': 'table_ref', 'field': 'field_ref'},
                            'name': 'test_field'
                        }
                    }
                },
                'test_not_use': {
                    'use': False,
                    'name': 'test_not_use',
                    'fields': {
                        'test_field_not_use': {
                            'ref': {'table': 'table_ref', 'field': 'field_ref'},
                            'name': 'test_field_not_use'
                        }
                    }
                }
            }
        }

        # Run
        metadata = Mock()
        metadata._metadata = _metadata

        Metadata._get_relationships(metadata)

        # Asserts
        expected__child_map = {'table_ref': {'test'}}
        expected__parent_map = {'test': {'table_ref'}}
        expected_foreign_keys = {('test', 'table_ref'): ('field_ref', 'test_field')}

        assert metadata._child_map == expected__child_map
        assert metadata._parent_map == expected__parent_map
        assert metadata.foreign_keys == expected_foreign_keys

    def test__dict_metadata(self):
        """Test dict_metadata"""
        # Run
        metadata = {
            'tables': [{
                'name': 'test',
                'use': True,
                'fields': [{
                    'ref': {'table': 'table_ref', 'field': 'field_ref'},
                    'name': 'test_field'
                }]
            }]
        }

        result = Metadata._dict_metadata(metadata)

        # Asserts
        expected = {
            'tables': {
                'test': {
                    'use': True,
                    'name': 'test',
                    'fields': {
                        'test_field': {
                            'ref': {'table': 'table_ref', 'field': 'field_ref'},
                            'name': 'test_field'
                        }
                    }
                }
            }
        }

        assert result == expected

    @patch('sdv.metadata.Metadata._get_relationships')
    @patch('sdv.metadata.Metadata._dict_metadata')
    def test___init__default_metadata_dict(self, mock_meta, mock_relationships):
        """Test create Metadata instance default with a dict"""
        # Run
        metadata_dict = {'some': 'meta'}
        metadata = Metadata(metadata_dict)

        # Asserts
        mock_meta.assert_called_once_with({'some': 'meta'})
        mock_relationships.assert_called_once_with()
        assert metadata.root_path == '.'
        assert metadata._hyper_transformers == dict()

    def test_get_children(self):
        """Test get children"""
        # Run
        metadata = Mock()
        metadata._child_map = {
            'test': 'child_table'
        }

        table_name = 'test'

        result = Metadata.get_children(metadata, table_name)

        # Asserts
        assert result == 'child_table'

    def test_get_parents(self):
        """Test get parents"""
        # Run
        metadata = Mock()
        metadata._parent_map = {
            'test': 'parent_table'
        }

        table_name = 'test'

        result = Metadata.get_parents(metadata, table_name)

        # Asserts
        assert result == 'parent_table'

    def test_get_table_meta(self):
        """Test get table meta"""
        # Run
        metadata = Mock()
        metadata._metadata = {
            'tables': {
                'test': {'some': 'data'}
            }
        }

        table_name = 'test'

        result = Metadata.get_table_meta(metadata, table_name)

        # Asserts
        expected = {'some': 'data'}

        assert result == expected

    @patch('sdv.metadata.load_csv')
    def test_load_table(self, mock_load_csv):
        """Test load table"""
        # Setup
        root_path = '.'
        table_meta = {'some': 'data'}

        # Run
        metadata = Mock()
        metadata.root_path = root_path
        metadata.get_table_meta.return_value = table_meta
        mock_load_csv.return_value = 'data'

        table_name = 'test'

        result = Metadata.load_table(metadata, table_name)

        # Asserts
        metadata.get_table_meta.assert_called_once_with('test')
        mock_load_csv.assert_called_once_with('.', {'some': 'data'})
        assert result == 'data'

    def test__get_dtypes(self):
        """Test get data types"""
        # Setup
        fields = {
            'item 1': {'type': 'number', 'subtype': 'integer'},
            'item 2': {'type': 'number', 'subtype': 'float'},
            'item 3': {'type': 'categorical', 'subtype': 'categorical'},
            'item 4': {'type': 'categorical', 'subtype': 'bool'},
            'item 5': {'type': 'datetime'}
        }

        # Run
        metadata = Mock()
        metadata.get_fields.return_value = fields

        table_name = 'test'

        result = Metadata._get_dtypes(metadata, table_name)

        # Asserts
        expected = [int, float, np.object, bool, np.datetime64]

        assert all([item in expected for item in result])
        assert len(result) == len(expected)

    def test__get_dtypes_error_subtype_categorical(self):
        """Test get data types with an invalid categorical subtype"""
        # Setup
        fields = {
            'item': {'type': 'categorical', 'subtype': 'integer'}
        }

        # Run and asserts
        metadata = Mock()
        metadata.get_fields.return_value = fields

        table_name = 'test'

        with pytest.raises(ValueError):
            Metadata._get_dtypes(metadata, table_name)

    def test__get_dtypes_error_subtype_numerical(self):
        """Test get data types with an invalid numerical subtype"""
        # Setup
        fields = {
            'item': {'type': 'number', 'subtype': 'bool'}
        }

        # Run and asserts
        metadata = Mock()
        metadata.get_fields.return_value = fields

        table_name = 'test'

        with pytest.raises(ValueError):
            Metadata._get_dtypes(metadata, table_name)

    def test__get_pii_fields(self):
        """Test get pii fields"""
        # Setup
        fields = {
            'foo': {
                'type': 'categorical',
                'pii': True,
                'pii_category': 'email'
            },
            'bar': {
                'type': 'categorical',
                'pii_category': 'email'
            }
        }

        # Run
        metadata = Mock()
        metadata.get_fields.return_value = fields

        table_name = 'test'

        result = Metadata._get_pii_fields(metadata, table_name)

        # Asserts
        expected = {'foo': 'email'}

        assert result == expected

    @patch('sdv.metadata.HyperTransformer')
    def test__load_hyper_transformer(self, mock_ht):
        """Test load HyperTransformer"""
        # Setup
        dtypes = list()
        pii_fields = dict()

        # Run
        metadata = Mock()
        metadata._get_dtypes.return_value = dtypes
        metadata._get_pii_fields.return_value = pii_fields
        mock_ht.return_value = 'hypertransformer'

        table_name = 'test'

        result = Metadata._load_hyper_transformer(metadata, table_name)

        # Asserts
        metadata._get_dtypes.assert_called_once_with('test')
        metadata._get_pii_fields.assert_called_once_with('test')
        mock_ht.assert_called_once_with(anonymize=dict(), dtypes=list())
        assert result == 'hypertransformer'

    def test_get_table_data_transform(self):
        """Test get table data with hyper transformer and transform"""
        # Setup
        load_table = pd.DataFrame({'foo': [0, 1]})
        _hyper_transformers = dict()
        hyper_transformer = Mock()
        hyper_transformer.transform.return_value = 'transformed'

        # Run
        metadata = Mock()
        metadata.load_table.return_value = load_table
        metadata._hyper_transformers = _hyper_transformers
        metadata._load_hyper_transformer.return_value = hyper_transformer

        table_name = 'test'
        transform = True

        result = Metadata.get_table_data(metadata, table_name, transform=transform)

        # Asserts
        metadata.load_table.assert_called_once_with('test')
        metadata._load_hyper_transformer.assert_called_once_with('test')

        pd.testing.assert_frame_equal(
            hyper_transformer.fit.call_args[0][0],
            pd.DataFrame({'foo': [0, 1]})
        )

        pd.testing.assert_frame_equal(
            hyper_transformer.transform.call_args[0][0],
            pd.DataFrame({'foo': [0, 1]})
        )

        assert result == 'transformed'

    def test_get_table_data_no_transform(self):
        """Test get table data with hyper transformer and no transform"""
        # Setup
        load_table = pd.DataFrame({'foo': [0, 1]})
        _hyper_transformers = dict()
        hyper_transformer = Mock()

        # Run
        metadata = Mock()
        metadata.load_table.return_value = load_table
        metadata._hyper_transformers = _hyper_transformers
        metadata._load_hyper_transformer.return_value = hyper_transformer

        table_name = 'test'
        transform = False

        result = Metadata.get_table_data(metadata, table_name, transform=transform)

        # Asserts
        expected = pd.DataFrame({'foo': [0, 1]})

        metadata.load_table.assert_called_once_with('test')
        metadata._load_hyper_transformer.assert_called_once_with('test')

        pd.testing.assert_frame_equal(
            hyper_transformer.fit.call_args[0][0],
            expected
        )

        pd.testing.assert_frame_equal(result, expected)

        assert hyper_transformer.transform.call_count == 0

    def test_get_table_names(self):
        """Test get table names"""
        # Setup
        _metadata = {
            'tables': {
                'table 1': None,
                'table 2': None,
                'table 3': None
            }
        }

        # Run
        metadata = Mock()
        metadata._metadata = _metadata

        result = Metadata.get_table_names(metadata)

        # Asserts
        expected = ['table 1', 'table 2', 'table 3']

        assert sorted(result) == sorted(expected)

    def test_get_tables(self):
        """Test get tables"""
        # Setup
        table_data = [
            pd.DataFrame({'foo': [1, 2]}),
            pd.DataFrame({'bar': [3, 4]}),
            pd.DataFrame({'tar': [5, 6]})
        ]

        # Run
        metadata = Mock()
        metadata.get_table_data.side_effect = table_data

        tables = ['table 1', 'table 2', 'table 3']

        result = Metadata.get_tables(metadata, tables=tables)

        # Asserts
        expected = {
            'table 1': pd.DataFrame({'foo': [1, 2]}),
            'table 2': pd.DataFrame({'bar': [3, 4]}),
            'table 3': pd.DataFrame({'tar': [5, 6]})
        }

        assert result.keys() == expected.keys()

        for k, v in result.items():
            pd.testing.assert_frame_equal(v, expected[k])

    def test_get_fields(self):
        """Test get fields"""
        # Setup
        table_meta = {
            'fields': {
                'a_field': 'some data',
                'b_field': 'other data'
            }
        }

        # Run
        metadata = Mock()
        metadata.get_table_meta.return_value = table_meta

        table_name = 'test'

        result = Metadata.get_fields(metadata, table_name)

        # Asserts
        expected = {'a_field': 'some data', 'b_field': 'other data'}

        metadata.get_table_meta.assert_called_once_with('test')

        assert result == expected

    def test_get_field_names(self):
        """Test get field names"""
        # Setup
        fields = {
            'a_field': 'some data',
            'b_field': 'other data'
        }

        # Run
        metadata = Mock()
        metadata.get_fields.return_value = fields

        table_name = 'test'

        result = Metadata.get_field_names(metadata, table_name)

        # Asserts
        expected = ['a_field', 'b_field']

        metadata.get_fields.assert_called_once_with('test')

        assert sorted(result) == sorted(expected)

    def test_get_field_meta(self):
        """Test get field meta"""
        # Setup
        fields = {
            'a_field': {'some': 'data'}
        }

        # Run
        metadata = Mock()
        metadata.get_fields.return_value = fields

        table_name = 'test'
        field_name = 'a_field'

        result = Metadata.get_field_meta(metadata, table_name, field_name)

        # Asserts
        expected = {'some': 'data'}

        metadata.get_fields.assert_called_once_with('test')

        assert result == expected

    def test_get_primary_key(self):
        """Test get primary key"""
        # Setup
        table_meta = {
            'primary_key': 'pk'
        }

        # Run
        metadata = Mock()
        metadata.get_table_meta.return_value = table_meta

        table_name = 'test'

        result = Metadata.get_primary_key(metadata, table_name)

        # Asserts
        expected = 'pk'

        metadata.get_table_meta.assert_called_once_with('test')

        assert result == expected

    def test_get_foreign_key(self):
        """Test get foreign key"""
        # Setup
        primary_key = 'pk'
        fields = {
            'a_field': {
                'ref': {
                    'field': 'pk'
                },
                'name': 'a_field'
            },
            'p_field': {
                'ref': {
                    'field': 'kk'
                },
                'name': 'p_field'
            }
        }

        # Run
        metadata = Mock()
        metadata.get_primary_key.return_value = primary_key
        metadata.get_fields.return_value = fields

        parent = 'parent_table'
        child = 'child_table'

        result = Metadata.get_foreign_key(metadata, parent, child)

        # Asserts
        expected = 'a_field'

        metadata.get_primary_key.assert_called_once_with('parent_table')
        metadata.get_fields.assert_called_once_with('child_table')

        assert result == expected

    def test_reverse_transform(self):
        """Test reverse transform"""
        # Setup
        ht_mock = Mock()
        _hyper_transformers = {
            'test': ht_mock
        }

        # Run
        metadata = Mock()
        metadata._hyper_transformers = _hyper_transformers

        table_name = 'test'
        data = pd.DataFrame({'foo': [0, 1]})

        Metadata.reverse_transform(metadata, table_name, data)

        # Asserts
        expected_call = pd.DataFrame({'foo': [0, 1]})
        pd.testing.assert_frame_equal(
            ht_mock.reverse_transform.call_args[0][0],
            expected_call
        )
