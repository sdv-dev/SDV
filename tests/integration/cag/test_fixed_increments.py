"""Integration tests for FixedIncrements CAG pattern."""

import numpy as np
import pandas as pd
import pytest

from sdv.cag import FixedIncrements
from sdv.metadata import Metadata
from tests.utils import run_pattern


@pytest.mark.parametrize(
    'dtype',
    [
        'int8',
        'int16',
        'int32',
        'int64',
        'Int8',
        'Int16',
        'Int32',
        'Int64',
        'float16',
        'float32',
        'float64',
        'Float32',
        'Float64',
    ],
)
def test_fixed_increments_integers(dtype):
    # Setup
    increment_value = 5
    values = np.random.randint(low=1, high=10, size=10) * increment_value
    series = pd.Series(values, dtype=dtype)
    if dtype.startswith('I') or dtype.startswith('F'):
        series.iloc[-2:] = pd.NA
    elif dtype.startswith('f'):
        series.iloc[-2:] = np.nan
    data = pd.DataFrame({dtype: series})
    metadata = Metadata.load_from_dict({
        'columns': {
            dtype: {'sdtype': 'numerical', 'computer_representation': dtype},
        }
    })
    pattern = FixedIncrements(dtype, increment_value=increment_value)

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {f'{dtype}#increment': {'sdtype': 'numerical'}}
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == [f'{dtype}#increment']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_fixed_incremements_with_multi_table():
    """Test that FixedIncrements constraint works with multi-table data."""
    # Setup
    increment_value = 1000
    A_values = np.random.randint(low=1, high=10, size=10)
    B_values = np.random.randint(low=1, high=100, size=10)
    A_values *= increment_value
    B_values *= increment_value

    data = {
        'table1': pd.DataFrame({
            'A': pd.Series(A_values, dtype='int64'),
            'B': pd.Series(B_values, dtype='Int64'),
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A': {'sdtype': 'numerical'},
                    'B': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })
    pattern = FixedIncrements(column_name='A', table_name='table1', increment_value=increment_value)

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A#increment': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed['table1'].columns) == ['B', 'A#increment']
    assert set(data.keys()) == set(reverse_transformed.keys())
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, reverse_transformed[table_name])
