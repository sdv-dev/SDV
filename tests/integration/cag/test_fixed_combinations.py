"""Integration tests for FixedCombinations CAG pattern."""

import numpy as np
import pandas as pd

from sdv.cag import FixedCombinations
from sdv.metadata import Metadata
from tests.utils import run_pattern


def test_fixed_combinations_integers():
    """Test that FixedCombinations constraint works with integer columns."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
        }
    })
    pattern = FixedCombinations(['A', 'B'])

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A#B': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A#B']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_fixed_combinations_with_nans():
    """Test that FixedCombinations constraint works with NaNs."""
    # Setup
    data = pd.DataFrame({
        'A': pd.Categorical([1, 2, np.nan, 1, 2, 1]),
        'B': pd.Categorical([10, 20, 30, 10, 20, None]),
    })
    metadata = Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
        }
    })

    pattern = FixedCombinations(['A', 'B'])

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {
            'A#B': {'sdtype': 'categorical'},
        }
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == ['A#B']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_fixed_null_combinations_with_multi_table():
    """Test that FixedCombinations constraint works with multi-table data."""
    # Setup
    data = {
        'table1': pd.DataFrame({
            'A': [1, 2, 3, 1, 2, 1],
            'B': [10, 20, 30, 10, 20, 10],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A': {'sdtype': 'categorical'},
                    'B': {'sdtype': 'categorical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })
    pattern = FixedCombinations(['A', 'B'], table_name='table1')

    # Run
    updated_metadata, transformed, reverse_transformed = run_pattern(pattern, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A#B': {'sdtype': 'categorical'},
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
    assert list(transformed['table1'].columns) == ['A#B']
    assert set(data.keys()) == set(reverse_transformed.keys())
    for table_name, table in data.items():
        pd.testing.assert_frame_equal(table, reverse_transformed[table_name])
