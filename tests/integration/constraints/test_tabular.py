import numpy as np
import pandas as pd

from sdv.constraints import FixedIncrements, Inequality, ScalarInequality
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
    sampled = gc.sample(10)

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
