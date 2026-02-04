import pandas as pd
import pytest

from sdv.metadata import Metadata


@pytest.fixture()
def primary_key_to_primary_key():
    metadata = Metadata.load_from_dict({
        'tables': {
            'tableA': {
                'columns': {
                    'table_id': {'sdtype': 'id'},
                    'col1': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_id',
            },
            'tableB': {
                'columns': {
                    'table_id': {'sdtype': 'id'},
                    'col2': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'tableA',
                'parent_primary_key': 'table_id',
                'child_table_name': 'tableB',
                'child_foreign_key': 'table_id',
            }
        ],
    })
    data = {
        'tableA': pd.DataFrame({
            'table_id': range(5),
            'col1': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableB': pd.DataFrame({
            'table_id': range(5),
            'col2': ['A', 'B', 'B', 'C', 'C'],
        }),
    }
    return data, metadata
