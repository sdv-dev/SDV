import numpy as np
import pandas as pd

from sdv.constraints import FixedIncrements, Inequality, Range, ScalarInequality, ScalarRange
from sdv.tabular import GaussianCopula


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
