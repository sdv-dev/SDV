import numpy as np
import pandas as pd
import pytest

from sdv.cag import Inequality, OneHotEncoding
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer


def test_overlapping_single_table_constraints():
    """Test overlapping single-table constraints work as expected."""
    # Setup
    parent_table = pd.DataFrame({
        'id': [i for i in range(20)],
        'colA': np.random.randint(low=0, high=100, size=20),
    })
    parent_table['colB'] = parent_table['colA'] + np.random.randint(low=1, high=10, size=20)
    parent_table['colC'] = parent_table['colB'] + np.random.randint(low=1, high=10, size=20)

    child_table = pd.DataFrame({
        'parent_id': np.random.randint(low=0, high=20, size=100),
        'colD': np.random.randint(low=100, high=200, size=100),
    })
    data = {'parent_table': parent_table, 'child_table': child_table}

    metadata = Metadata()
    metadata = Metadata.detect_from_dataframes(data)

    constraint1 = Inequality(
        low_column_name='colA',
        high_column_name='colB',
        table_name='parent_table',
        strict_boundaries=True,
    )
    constraint2 = Inequality(
        low_column_name='colB',
        high_column_name='colC',
        table_name='parent_table',
        strict_boundaries=True,
    )
    synthesizer = HMASynthesizer(metadata)

    # Run
    synthesizer.add_constraints(constraints=[constraint1, constraint2])
    synthesizer.fit(data)
    sampled = synthesizer.sample(10)

    # Assert
    assert all(sampled['parent_table']['colA'] < sampled['parent_table']['colB'])
    assert all(sampled['parent_table']['colB'] < sampled['parent_table']['colC'])


def test_add_constraint_iteratively():
    """Test adding constraints in multiple steps."""
    # Setup
    parent_table = pd.DataFrame({
        'id': [i for i in range(20)],
        'colA': np.random.randint(low=0, high=100, size=20),
    })
    parent_table['colB'] = parent_table['colA'] + np.random.randint(low=1, high=10, size=20)
    parent_table['colC'] = parent_table['colB'] + np.random.randint(low=1, high=10, size=20)

    child_table = pd.DataFrame({
        'parent_id': np.random.randint(low=0, high=20, size=100),
        'colD': np.random.randint(low=100, high=200, size=100),
    })
    data = {'parent_table': parent_table, 'child_table': child_table}

    metadata = Metadata()
    metadata = Metadata.detect_from_dataframes(data)

    constraint1 = Inequality(
        low_column_name='colA',
        high_column_name='colB',
        table_name='parent_table',
        strict_boundaries=True,
    )
    constraint2 = Inequality(
        low_column_name='colB',
        high_column_name='colC',
        table_name='parent_table',
        strict_boundaries=True,
    )
    synthesizer = HMASynthesizer(metadata)

    # Run
    synthesizer.add_constraints([constraint1])
    synthesizer.add_constraints([constraint2])
    synthesizer.fit(data)
    sampled = synthesizer.sample(10)

    # Assert
    assert all(sampled['parent_table']['colA'] < sampled['parent_table']['colB'])
    assert all(sampled['parent_table']['colB'] < sampled['parent_table']['colC'])


@pytest.mark.parametrize('computer_representation, dtype', [('Int64', 'int64'), ('Int8', 'int8')])
def test_ohe_with_computer_representation(computer_representation, dtype):
    """Test OneHotEncoding constraint with integer columns and computer representation"""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'a': {
                        'sdtype': 'numerical',
                        'computer_representation': computer_representation,
                    },
                    'b': {
                        'sdtype': 'numerical',
                        'computer_representation': computer_representation,
                    },
                },
            },
        }
    })
    data = {
        'table1': pd.DataFrame({
            'a': pd.Series([1, 1, 0], dtype=dtype),
            'b': pd.Series([0, 0, 1], dtype=dtype),
        })
    }
    synthesizer = HMASynthesizer(metadata)
    constraint = OneHotEncoding(column_names=['a', 'b'], table_name='table1')
    synthesizer.add_constraints([constraint])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(scale=2.0)

    # Assert
    synthesizer.validate(synthetic_data)
    assert (synthetic_data['table1'].sum(axis=1) == 1).all()
    assert set(synthetic_data['table1']['a'].unique()).issubset({0, 1})
    assert set(synthetic_data['table1']['b'].unique()).issubset({0, 1})
