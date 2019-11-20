from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.metadata import Metadata, _parse_dtypes, _read_csv_dtypes


def test__read_csv_dtypes():
    """Test read csv data types"""
    # Run
    table_meta = {
        'fields': {
            'a_field': {
                'type': 'categorical'
            },
            'b_field': {
                'type': 'boolean'
            },
            'c_field': {
                'type': 'datetime',
            },
            'd_field': {
                'type': 'id',
                'subtype': 'string'
            },
            'e_field': {
                'type': 'id',
                'subtype': 'integer'
            }
        }
    }

    result = _read_csv_dtypes(table_meta)

    # Asserts
    expected = {'a_field': str, 'd_field': str}

    assert result == expected


def test__parse_dtypes():
    """Test parse data types"""
    # Run
    data = pd.DataFrame({
        'a_field': ['1996-10-17', '1965-05-23'],
        'b_field': ['7', '14'],
        'c_field': ['1', '2'],
        'd_field': ['other', 'data']
    })
    table_meta = {
        'fields': {
            'a_field': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
            'b_field': {
                'type': 'numerical',
                'subtype': 'integer'
            },
            'c_field': {
                'type': 'id',
                'subtype': 'integer'
            },
            'd_field': {
                'type': 'other'
            }
        }
    }

    result = _parse_dtypes(data, table_meta)

    # Asserts
    expected = pd.DataFrame({
        'a_field': pd.to_datetime(['1996-10-17', '1965-05-23'], format='%Y-%m-%d'),
        'b_field': [7, 14],
        'c_field': [1, 2],
        'd_field': ['other', 'data']
    })

    pd.testing.assert_frame_equal(result, expected)


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

        assert metadata._child_map == expected__child_map
        assert metadata._parent_map == expected__parent_map

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

    @patch('sdv.metadata._load_csv')
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

    def test__get_dtypes_with_ids(self):
        """Test get data types including ids."""
        # Setup
        table_meta = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'numerical', 'subtype': 'integer'},
                'item 2': {'type': 'numerical', 'subtype': 'float'},
                'item 3': {'type': 'categorical'},
                'item 4': {'type': 'boolean'},
                'item 5': {'type': 'datetime'}
            },
            'primary_key': 'item 0'
        }

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_meta.return_value = table_meta

        result = Metadata._get_dtypes(metadata, 'test', ids=True)

        # Asserts
        expected = {
            'item 0': int,
            'item 1': int,
            'item 2': float,
            'item 3': np.object,
            'item 4': bool,
            'item 5': np.datetime64,
        }

        assert result == expected

    def test__get_dtypes_no_ids(self):
        """Test get data types excluding ids."""
        # Setup
        table_meta = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'numerical', 'subtype': 'integer'},
                'item 2': {'type': 'numerical', 'subtype': 'float'},
                'item 3': {'type': 'categorical'},
                'item 4': {'type': 'boolean'},
                'item 5': {'type': 'datetime'},
            }
        }

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_meta.return_value = table_meta

        result = Metadata._get_dtypes(metadata, 'test')

        # Asserts
        expected = {
            'item 1': int,
            'item 2': float,
            'item 3': np.object,
            'item 4': bool,
            'item 5': np.datetime64,
        }

        assert result == expected

    def test__get_dtypes_error_invalid_type(self):
        """Test get data types with an invalid type."""
        # Setup
        table_meta = {
            'fields': {
                'item': {'type': 'unknown'}
            }
        }

        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_meta.return_value = table_meta

        with pytest.raises(ValueError):
            Metadata._get_dtypes(metadata, 'test')

    def test__get_dtypes_error_id(self):
        """Test get data types with an id that is not a primary or foreign key."""
        # Setup
        table_meta = {
            'fields': {
                'item': {'type': 'id'}
            }
        }

        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_meta.return_value = table_meta

        with pytest.raises(ValueError):
            Metadata._get_dtypes(metadata, 'test', ids=True)

    def test__get_dtypes_error_subtype_numerical(self):
        """Test get data types with an invalid numerical subtype."""
        # Setup
        table_meta = {
            'fields': {
                'item': {'type': 'numerical', 'subtype': 'boolean'}
            }
        }

        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_meta.return_value = table_meta

        with pytest.raises(ValueError):
            Metadata._get_dtypes(metadata, 'test')

    def test__get_dtypes_error_subtype_id(self):
        """Test get data types with an invalid id subtype."""
        # Setup
        table_meta = {
            'fields': {
                'item': {'type': 'id', 'subtype': 'boolean'}
            }
        }

        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_meta.return_value = table_meta

        with pytest.raises(ValueError):
            Metadata._get_dtypes(metadata, 'test', ids=True)

    def test__get_pii_fields(self):
        """Test get pii fields"""
        # Setup
        table_meta = {
            'fields': {
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
        }

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_meta.return_value = table_meta

        table_name = 'test'

        result = Metadata._get_pii_fields(metadata, table_name)

        # Asserts
        expected = {'foo': 'email'}

        assert result == expected

    @patch('sdv.metadata.transformers.DatetimeTransformer')
    @patch('sdv.metadata.transformers.BooleanTransformer')
    @patch('sdv.metadata.transformers.CategoricalTransformer')
    @patch('sdv.metadata.transformers.NumericalTransformer')
    def test__get_transformers_no_error(
            self, numerical_mock, categorical_mock, boolean_mock, datetime_mock):
        """Test get transformers dict for each data type."""
        # Setup
        numerical_mock.return_value = 'NumericalTransformer'
        categorical_mock.return_value = 'CategoricalTransformer'
        boolean_mock.return_value = 'BooleanTransformer'
        datetime_mock.return_value = 'DatetimeTransformer'

        # Run
        dtypes = {
            'integer': int,
            'float': float,
            'categorical': np.object,
            'boolean': bool,
            'datetime': np.datetime64
        }

        pii_fields = {
            'categorical': 'email'
        }

        result = Metadata._get_transformers(dtypes, pii_fields)

        # Asserts
        expected = {
            'integer': 'NumericalTransformer',
            'float': 'NumericalTransformer',
            'categorical': 'CategoricalTransformer',
            'boolean': 'BooleanTransformer',
            'datetime': 'DatetimeTransformer'
        }
        expected_numerical_calls = [call(dtype=int), call(dtype=float)]

        assert result == expected
        assert len(numerical_mock.call_args_list) == len(expected_numerical_calls)
        for item in numerical_mock.call_args_list:
            assert item in expected_numerical_calls
        assert categorical_mock.call_args == call(anonymize='email')
        assert boolean_mock.call_args == call()
        assert datetime_mock.call_args == call()

    def test__get_transformers_raise_valueerror(self):
        """Test get transformers dict raise ValueError."""
        # Run
        dtypes = {
            'string': str
        }

        with pytest.raises(ValueError):
            Metadata._get_transformers(dtypes, None)

    @patch('sdv.metadata.HyperTransformer')
    def test__load_hyper_transformer(self, mock_ht):
        """Test load HyperTransformer"""
        # Run
        metadata = Mock(spec=Metadata)
        metadata._get_dtypes.return_value = {'meta': 'dtypes'}
        metadata._get_pii_fields.return_value = {'meta': 'pii_fields'}
        metadata._get_transformers.return_value = {'meta': 'transformers'}
        mock_ht.return_value = 'hypertransformer'

        table_name = 'test'

        result = Metadata._load_hyper_transformer(metadata, table_name)

        # Asserts
        metadata._get_dtypes.assert_called_once_with('test')
        metadata._get_pii_fields.assert_called_once_with('test')

        metadata._get_transformers.assert_called_once_with(
            {'meta': 'dtypes'}, {'meta': 'pii_fields'})

        mock_ht.assert_called_once_with(transformers={'meta': 'transformers'})
        assert result == 'hypertransformer'

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
        table_names = ['foo', 'bar', 'tar']
        table_data = [
            pd.DataFrame({'foo': [1, 2]}),
            pd.DataFrame({'bar': [3, 4]}),
            pd.DataFrame({'tar': [5, 6]})
        ]

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.side_effect = table_names
        metadata.load_table.side_effect = table_data

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
        data_types = {
            'item 1': int,
            'item 2': float,
            'item 3': np.object,
            'item 4': bool,
        }

        ht_mock = Mock()
        ht_mock.reverse_transform.return_value = {
            'item 1': pd.Series([1.0, 2.0, None, 4.0, 5.0]),
            'item 2': pd.Series([1.1, None, 3.3, None, 5.5]),
            'item 3': pd.Series([None, 'bbb', 'ccc', 'ddd', None]),
            'item 4': pd.Series([True, False, None, False, True])
        }

        _hyper_transformers = {
            'test': ht_mock
        }

        # Run
        metadata = Mock()
        metadata._hyper_transformers = _hyper_transformers
        metadata._get_dtypes.return_value = data_types

        table_name = 'test'
        data = pd.DataFrame({'foo': [0, 1]})

        Metadata.reverse_transform(metadata, table_name, data)

        # Asserts
        expected_call = pd.DataFrame({'foo': [0, 1]})
        pd.testing.assert_frame_equal(
            ht_mock.reverse_transform.call_args[0][0],
            expected_call
        )

    def test_add_table_already_exist(self):
        """Try to add a new table that already exist"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names

        with pytest.raises(ValueError):
            Metadata.add_table(metadata, 'a_table')

    def test_add_table_only_name(self):
        """Add table with only the name"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names
        metadata._metadata = {'tables': dict()}

        Metadata.add_table(metadata, 'x_table')

        # Asserts
        expected_table_meta = {
            'name': 'x_table',
            'fields': dict()
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.add_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_table_with_primary_key(self):
        """Add table with primary key"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names
        metadata._metadata = {'tables': dict()}

        Metadata.add_table(metadata, 'x_table', primary_key='id')

        # Asserts
        expected_table_meta = {
            'name': 'x_table',
            'fields': dict()
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.add_primary_key.assert_called_once_with('x_table', 'id')
        metadata.add_relationship.call_count == 0

    def test_add_table_with_foreign_key(self):
        """Add table with foreign key"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names
        metadata._metadata = {'tables': dict()}

        Metadata.add_table(metadata, 'x_table', parent='users')

        # Asserts
        expected_table_meta = {
            'name': 'x_table',
            'fields': dict()
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.add_primary_key.call_count == 0
        metadata.add_relationship.assert_called_once_with('x_table', 'users', None)

    def test_add_table_with_fields_dict(self):
        """Add table with fields(dict)"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names
        metadata._metadata = {'tables': dict()}

        fields = {
            'a_field': {'type': 'numerical', 'subtype': 'integer'}
        }

        Metadata.add_table(metadata, 'x_table', fields=fields)

        # Asserts
        expected_table_meta = {
            'name': 'x_table',
            'fields': {
                'a_field': {'name': 'a_field', 'type': 'numerical', 'subtype': 'integer'}
            }
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        assert metadata._validate_field.call_args_list == [
            call({'name': 'a_field', 'type': 'numerical', 'subtype': 'integer'})
        ]

        metadata.add_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_table_with_field_list_no_data(self):
        """Add table with fields(list) no data"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names
        metadata._metadata = {'tables': dict()}

        fields = ['a_field', 'b_field']

        with pytest.raises(ValueError):
            Metadata.add_table(metadata, 'x_table', fields=fields)

    def test_add_table_with_field_list_data(self):
        """Add table with fields(list) data"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names
        metadata._metadata = {'tables': dict()}

        fields = ['a_field', 'b_field']
        data = pd.DataFrame({'a_field': [0, 1], 'b_field': [True, False], 'c_field': ['a', 'b']})

        Metadata.add_table(metadata, 'x_table', fields=fields, data=data)

        # Asserts
        expected_table_meta = {
            'name': 'x_table',
            'fields': {
                'a_field': {'name': 'a_field', 'type': 'numerical', 'subtype': 'integer'},
                'b_field': {'name': 'b_field', 'type': 'boolean'}
            }
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.add_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_table_with_data_analyze(self):
        """Add table with data to analyze all"""
        # Setup
        table_names = ['a_table', 'b_table']

        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = table_names
        metadata._metadata = {'tables': dict()}

        data = pd.DataFrame({'a_field': [0, 1], 'b_field': [True, False], 'c_field': ['a', 'b']})

        Metadata.add_table(metadata, 'x_table', data=data)

        # Asserts
        expected_table_meta = {
            'name': 'x_table',
            'fields': {
                'a_field': {'name': 'a_field', 'type': 'numerical', 'subtype': 'integer'},
                'b_field': {'name': 'b_field', 'type': 'boolean'},
                'c_field': {'name': 'c_field', 'type': 'categorical'}
            }
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.add_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_relationship_table_no_exist(self):
        """Add relationship table no exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = list()

        with pytest.raises(ValueError):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_add_relationship_parent_no_exist(self):
        """Add relationship table no exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table']

        with pytest.raises(ValueError):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_add_relationship_already_exist(self):
        """Add relationship already exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table', 'b_table']
        metadata.get_parents.return_value = set(['b_table'])

        with pytest.raises(ValueError):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_add_relationship_parent_is_child_of_table(self):
        """Add relationship parent is child of table"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table', 'b_table']
        metadata.get_parents.return_value = set()
        metadata.get_children.return_value = set(['b_table'])

        with pytest.raises(ValueError):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_add_relationship_parent_no_primary_key(self):
        """Add relationship parent no primary key"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table', 'b_table']
        metadata.get_parents.return_value = set()
        metadata.get_children.return_value = set()
        metadata.get_primary_key.return_value = None

        with pytest.raises(ValueError):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_add_relationship_valid(self):
        """Add relationship valid"""
        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table', 'b_table']
        metadata.get_parents.return_value = set()
        metadata.get_children.return_value = set()
        metadata.get_primary_key.return_value = 'pk_field'

        Metadata.add_relationship(metadata, 'a_table', 'b_table')

        # Asserts
        metadata._validate_circular_relationships.assert_called_once_with('b_table', set())
        metadata.add_field.assert_called_once_with(
            'a_table', 'pk_field', 'id', None, {'ref': {'field': 'pk_field', 'table': 'b_table'}}
        )
        metadata._get_relationships.assert_called_once_with()

    def test_add_primary_key_table_no_exist(self):
        """Add primary key table no exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = list()

        with pytest.raises(ValueError):
            Metadata.add_primary_key(metadata, 'a_table', 'a_field')

    def test_add_primary_key_field_exist(self):
        """Add primary key field exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table']
        metadata.get_fields.return_value = dict()

        with pytest.raises(ValueError):
            Metadata.add_primary_key(metadata, 'a_table', 'a_field')

    def test_add_primary_key_primary_key_exist(self):
        """Add primary key primary key exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table']
        metadata.get_fields.return_value = {'a_field': dict()}
        metadata.get_primary_key.return_value = 'some_primary_key'

        with pytest.raises(ValueError):
            Metadata.add_primary_key(metadata, 'a_table', 'a_field')

    def test_add_primary_key_valid(self):
        """Add primary key valid"""
        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table']
        metadata.get_fields.return_value = dict()
        metadata.get_primary_key.return_value = None
        metadata.get_table_meta.return_value = dict()

        Metadata.add_primary_key(metadata, 'a_table', 'a_field')

        # Asserts
        metadata.get_table_names.assert_called_once_with()
        metadata.get_fields.assert_called_once_with('a_table')
        metadata.get_primary_key.assert_called_once_with('a_table')

        metadata.get_table_meta.assert_called_once_with('a_table')
        metadata.add_field.assert_called_once_with('a_table', 'a_field', 'id', None, None)

    def test_add_field_table_no_exist(self):
        """Add field table no exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = list()

        with pytest.raises(ValueError):
            Metadata.add_field(metadata, 'a_table', 'a_field', 'id', None, None)

    def test_add_field_field_exist(self):
        """Add field already exist"""
        # Run and asserts
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table']
        metadata.get_fields.return_value = {'a_field': dict()}

        with pytest.raises(ValueError):
            Metadata.add_field(metadata, 'a_table', 'a_field', 'id', None, None)

    def test_add_field_valid(self):
        """Add valid field"""
        # Run
        metadata = Mock(spec=Metadata)
        metadata.get_table_names.return_value = ['a_table']
        metadata.get_fields.return_value = dict()

        Metadata.add_field(metadata, 'a_table', 'a_field', 'numerical', 'integer', {'min': 0})

        # Asserts
        metadata.get_table_names.assert_called_once_with()
        metadata.get_fields.assert_called_once_with('a_table')
