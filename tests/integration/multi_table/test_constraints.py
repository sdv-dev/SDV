import numpy as np
import pandas as pd

from sdv.cag import Inequality
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
