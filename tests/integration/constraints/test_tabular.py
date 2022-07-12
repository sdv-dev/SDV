import numpy as np
import pandas as pd

from sdv.constraints import (
    FixedIncrements, Inequality, Range, ScalarInequality, ScalarRange, create_custom_constraint)
from sdv.sampling.tabular import Condition
from sdv.tabular import GaussianCopula


def test_create_custom_constraint():
    """Test the ``create_custom_constraint`` method end to end."""
    # Setup
    custom_constraint = create_custom_constraint(
        lambda _, x: pd.Series([True if x_i > 0 else False for x_i in x['col']]),
        lambda _, x: pd.DataFrame({'col': x['col'] ** 2}),
        lambda _, x: pd.DataFrame({'col': x['col'] ** .5})
    )('col')

    data = pd.DataFrame({'col': np.random.randint(1, 10, size=100)})
    gc = GaussianCopula(constraints=[custom_constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(100)

    # Assert
    assert all(sampled > 0)


def test_invalid_create_custom_constraint():
    """Test the an invalid ``create_custom_constraint`` method end to end.

    It should correctly sample the synthetic data through reject sample.
    """
    # Setup
    custom_constraint = create_custom_constraint(
        lambda _, x: pd.Series([True if x_i > 0 else False for x_i in x['col']]),
        lambda _: pd.DataFrame({'col': [10 / 0] * 100}),
        lambda _, x: pd.DataFrame({'col': x['col'] ** .5})
    )('col')

    data = pd.DataFrame({'col': np.random.randint(1, 10, size=100)})
    gc = GaussianCopula(constraints=[custom_constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(100)

    # Assert
    assert all(sampled > 0)


def test_FixedIncrements():
    """Test the ``FixedIncrements`` constraint end to end."""
    # Setup
    values = np.random.randint(1, 10, size=20) * 5
    data = pd.DataFrame({'column': values})
    constraint = FixedIncrements(column_name='column', increment_value=5)
    gc = GaussianCopula(constraints=[constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(100)

    # Assert
    assert all(sampled % 5 == 0)


def test_Inequality():
    """Test the ``Inequality`` constraint end to end."""
    # Setup
    data = pd.DataFrame({
        'low': np.random.randint(1, 10, size=20),
        'high': np.random.randint(10, 20, size=20)
    })
    constraint = Inequality('low', 'high')
    gc = GaussianCopula(constraints=[constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(10)

    # Assert
    assert all(sampled['low'] <= sampled['high'])


def test_ScalarInequality():
    """Test the ``ScalarInequality`` constraint end to end."""
    # Setup
    data = pd.DataFrame({
        'low': np.random.randint(1, 10, size=20),
    })
    constraint = ScalarInequality(column_name='low', value=11, relation='<')
    gc = GaussianCopula(constraints=[constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(10)

    # Assert
    assert all(sampled['low'] < 11)


def test_Range():
    """Test the ``Range`` constraint end to end."""
    # Setup
    data = pd.DataFrame({
        'low_column': np.random.randint(1, 5, size=20),
        'middle_column': np.random.randint(6, 10, size=20),
        'high_column': np.random.randint(11, 20, size=20),
    })
    constraint = Range(
        low_column_name='low_column',
        middle_column_name='middle_column',
        high_column_name='high_column',
        strict_boundaries=True
    )

    gc = GaussianCopula(constraints=[constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(100)

    # Assert
    assert sampled.middle_column.min() >= sampled.low_column.min()
    assert sampled.middle_column.max() <= sampled.high_column.max()


def test_ScalarRange():
    """Test the ``ScalarRange`` constraint end to end."""
    # Setup
    data = pd.DataFrame({
        'column': np.random.randint(6, 10, size=20),
    })
    constraint = ScalarRange(
        column_name='column',
        low_value=5,
        high_value=11,
        strict_boundaries=True
    )

    gc = GaussianCopula(constraints=[constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(100)

    # Assert
    assert sampled.column.min() >= 5
    assert sampled.column.max() <= 11


def test_ScalarRange_conditions():
    """Test ``ScalarRange`` with conditions.

    This test ensures that the conditions are not altered by the constraint transformation.
    """
    # Setup
    constraint = ScalarRange(column_name='input', low_value=49, high_value=100)
    data = pd.DataFrame({
        'input': [_ for _ in range(50, 80)],
        'output': [np.random.rand() for _ in range(30)]
    })
    condition = Condition({'input': 88}, num_rows=10)
    model = GaussianCopula(
        field_names=['input', 'output'],
        field_transformers={'input': 'integer', 'output': 'float'},
        constraints=[constraint],
    )

    # Run
    model.fit(data)
    sampled = model.sample_conditions([condition])

    # Assert
    assert all(sampled['input'] == 88)
    assert all(sampled['output'] > 0)
    assert all(sampled['output'] < 1)
