"""Integration tests for FixedIncrements CAG constraint."""

import re

import numpy as np
import pandas as pd
import pytest

from sdv.cag import FixedIncrements
from sdv.cag._errors import constraintNotMetError
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import run_constraint


@pytest.fixture()
def data():
    return pd.DataFrame({'A': [10, 20, 30, 40, 50]})


@pytest.fixture()
def metadata():
    return Metadata.load_from_dict({
        'columns': {
            'A': {'sdtype': 'numerical'},
        }
    })


@pytest.fixture()
def constraint():
    return FixedIncrements(column_name='A', increment_value=10)


@pytest.fixture()
def data_multi():
    increment_value = 1000
    A_values = np.random.randint(low=1, high=10, size=10)
    B_values = np.random.randint(low=1, high=100, size=10)
    A_values *= increment_value
    B_values *= increment_value

    return {
        'table1': pd.DataFrame({
            'A': pd.Series(A_values, dtype='int64'),
            'B': pd.Series(B_values, dtype='Int64'),
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }


@pytest.fixture()
def metadata_multi():
    return Metadata.load_from_dict({
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


@pytest.fixture()
def constraint_multi():
    return FixedIncrements(column_name='A', table_name='table1', increment_value=1000)


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
    constraint = FixedIncrements(dtype, increment_value=increment_value)

    # Run
    updated_metadata, transformed, reverse_transformed = run_constraint(constraint, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'columns': {f'{dtype}#increment': {'sdtype': 'numerical'}}
    }).to_dict()
    assert expected_updated_metadata == updated_metadata.to_dict()
    assert list(transformed.columns) == [f'{dtype}#increment']
    pd.testing.assert_frame_equal(data, reverse_transformed)


def test_fixed_incremements_with_multi_table(data_multi, metadata_multi, constraint_multi):
    """Test that FixedIncrements constraint works with multi-table data."""
    # Run
    data = data_multi
    metadata = metadata_multi
    constraint = constraint_multi
    updated_metadata, transformed, reverse_transformed = run_constraint(constraint, data, metadata)

    # Assert
    expected_updated_metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A#increment': {'sdtype': 'numerical'},
                    'B': {'sdtype': 'numerical'},
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


def run_copula(data, metadata, constraint):
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(constraints=[constraint])
    synthesizer.fit(data)
    return synthesizer


def test_validate_cag(data, metadata, constraint):
    """Test validate_cag works with synthetic data generated with FixedIncrements."""
    # Setup
    synthesizer = run_copula(data, metadata, constraint)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(data['A'] % constraint.increment_value == 0)
    assert all(synthetic_data['A'] % constraint.increment_value == 0)


def test_validate_cag_raises(data, metadata, constraint):
    """Test validate_cag raises an error with bad multitable data with FixedIncrements."""
    # Setup
    synthesizer = run_copula(data, metadata, constraint)
    synthetic_data = pd.DataFrame({'A': [1, 3, 5, 7, 9, 12]})
    msg = re.escape('The fixed increments requirement is not met for row indices: 0, 1, 2, 3, 4')

    # Run and Assert
    with pytest.raises(constraintNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)


def test_validate_cag_multi(data_multi, metadata_multi, constraint_multi):
    """Test validate_cag works with multitable data generated with FixedIncrements."""
    # Setup
    data = data_multi
    metadata = metadata_multi
    constraint = constraint_multi
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_cag(constraints=[constraint])
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    assert all(data['table1']['A'] % constraint.increment_value == 0)
    assert all(synthetic_data['table1']['A'] % constraint.increment_value == 0)


def test_validate_cag_multi_raises():
    """Test validate_cag raises an error with bad multitable data with FixedIncrements."""
    # Setup
    data = {
        'table1': pd.DataFrame({'A': [2, 4, 6, 8, 10]}),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'A': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })
    constraint = FixedIncrements(column_name='A', table_name='table1', increment_value=2)
    synthetic_data = {
        'table1': pd.DataFrame({'A': [1, 3, 5, 7, 9, 12]}),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_cag(constraints=[constraint])
    synthesizer.fit(data)
    msg = re.escape(
        "Table 'table1': The fixed increments requirement is "
        'not met for row indices: 0, 1, 2, 3, 4.'
    )

    # Run and Assert
    with pytest.raises(constraintNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)
