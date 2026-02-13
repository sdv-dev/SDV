import pandas as pd
import pytest

from sdv.metadata import Metadata


@pytest.fixture()
def primary_key_to_primary_key():
    metadata = Metadata.load_from_dict({
        'tables': {
            'tableA': {
                'columns': {
                    'table_A_primary_key': {'sdtype': 'id'},
                    'column_1': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_A_primary_key',
            },
            'tableB': {
                'columns': {
                    'table_B_primary_key': {'sdtype': 'id'},
                    'column_2': {'sdtype': 'categorical'},
                },
                'primary_key': 'table_B_primary_key',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'tableA',
                'parent_primary_key': 'table_A_primary_key',
                'child_table_name': 'tableB',
                'child_foreign_key': 'table_B_primary_key',
            }
        ],
    })
    data = {
        'tableA': pd.DataFrame({
            'table_A_primary_key': range(5),
            'column_1': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableB': pd.DataFrame({
            'table_B_primary_key': range(5),
            'column_2': ['A', 'B', 'B', 'C', 'C'],
        }),
    }
    return data, metadata
