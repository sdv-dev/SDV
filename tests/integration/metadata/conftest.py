import pandas as pd
import pytest

from sdv.metadata import Metadata


@pytest.fixture()
def primary_key_to_primary_key():
    metadata = Metadata.load_from_dict({
        'tables': {
            'tableA': {
                'columns': {
<<<<<<< HEAD
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
=======
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
>>>>>>> cf5ffe7f60030955f315c372e10606675cb7330e
            },
        },
        'relationships': [
            {
                'parent_table_name': 'tableA',
<<<<<<< HEAD
                'parent_primary_key': 'table_id',
                'child_table_name': 'tableB',
                'child_foreign_key': 'table_id',
=======
                'parent_primary_key': 'table_A_primary_key',
                'child_table_name': 'tableB',
                'child_foreign_key': 'table_B_primary_key',
>>>>>>> cf5ffe7f60030955f315c372e10606675cb7330e
            }
        ],
    })
    data = {
        'tableA': pd.DataFrame({
<<<<<<< HEAD
            'table_id': range(5),
            'col1': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableB': pd.DataFrame({
            'table_id': range(5),
            'col2': ['A', 'B', 'B', 'C', 'C'],
=======
            'table_A_primary_key': range(5),
            'column_1': ['A', 'B', 'B', 'C', 'C'],
        }),
        'tableB': pd.DataFrame({
            'table_B_primary_key': range(5),
            'column_2': ['A', 'B', 'B', 'C', 'C'],
>>>>>>> cf5ffe7f60030955f315c372e10606675cb7330e
        }),
    }
    return data, metadata
