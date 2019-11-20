import pandas as pd
import pytest
from sdv import Metadata


def get_metadata():
    return Metadata({'tables': dict()})


def test_add_fields_and_primary_key():
    metadata = get_metadata()

    metadata.add_table('a_table')

    metadata.add_field('a_table', 'categoricals', 'categorical')
    metadata.add_field('a_table', 'integers', 'numerical', 'integer', {'min': 0, 'max': 10})
    metadata.add_field('a_table', 'floats', 'numerical', 'float')
    metadata.add_field('a_table', 'booleans', 'boolean')

    metadata.add_primary_key('a_table', 'index')

    expected_metadata = {
        'tables': {
            'a_table': {
                'name': 'a_table',
                'primary_key': 'index',
                'fields': {
                    'categoricals': {
                        'name': 'categoricals',
                        'type': 'categorical'
                    },
                    'integers': {
                        'name': 'integers',
                        'type': 'numerical',
                        'subtype': 'integer',
                        'min': 0,
                        'max': 10
                    },
                    'floats': {
                        'name': 'floats',
                        'type': 'numerical',
                        'subtype': 'float'
                    },
                    'booleans': {
                        'name': 'booleans',
                        'type': 'boolean'
                    },
                    'index': {
                        'name': 'index',
                        'type': 'id'
                    }
                }
            }
        }
    }

    assert metadata._metadata == expected_metadata


def test_add_table_analyze_all():
    metadata = get_metadata()

    data = pd.DataFrame({
        'a_field': [0, 1, 2],
        'b_field': ['a', 'b', 'c'],
        'c_field': [True, False, False],
        'd_field': [0., 1., 2.]
    })

    metadata.add_table('a_table', data=data)

    expected_metadata = {
        'tables': {
            'a_table': {
                'name': 'a_table',
                'fields': {
                    'a_field': {
                        'name': 'a_field',
                        'type': 'numerical',
                        'subtype': 'integer'
                    },
                    'b_field': {
                        'name': 'b_field',
                        'type': 'categorical'
                    },
                    'c_field': {
                        'name': 'c_field',
                        'type': 'boolean'
                    },
                    'd_field': {
                        'name': 'd_field',
                        'type': 'numerical',
                        'subtype': 'float'
                    }
                }
            }
        }
    }

    assert metadata._metadata == expected_metadata


def test_add_relationships():
    metadata = get_metadata()

    metadata.add_table('foo', primary_key='index_foo')
    metadata.add_table('bar', primary_key='index_bar', parent='foo')

    assert metadata.get_children('foo') == set(['bar'])
    assert metadata.get_parents('bar') == set(['foo'])


def test_cirtular_dependence_validation():
    metadata = get_metadata()

    metadata.add_table('foo', primary_key='index_foo')
    metadata.add_table('bar', primary_key='index_bar', parent='foo')
    metadata.add_table('tar', primary_key='index_tar', parent='bar')

    with pytest.raises(ValueError):
        metadata.add_relationship('foo', 'tar')
