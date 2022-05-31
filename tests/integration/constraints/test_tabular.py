import numpy as np
import pandas as pd

from sdv.constraints import FixedIncrements
from sdv.tabular import GaussianCopula


def test_FixedIncrements():
    # Setup
    values = np.random.randint(1, 10, size=20) * 5
    data = pd.DataFrame({'column': values})
    constraint = FixedIncrements(column_name='column', increment_value=5)
    gc = GaussianCopula(constraints=[constraint])
    gc.fit(data)

    # Run
    sampled = gc.sample(10)
    assert all(sampled % 5 == 0)
