import pandas as pd
import pytest

from sdv.tabular.ctgan import TVAE


@pytest.mark.xfail(reason="not implemented")
def test_conditional_sampling_tvae_fails():
    data = pd.DataFrame({
        "column1": [1.0, 0.5, 2.5] * 10,
        "column2": ["a", "b", "c"] * 10,
        "column3": ["d", "e", "f"] * 10
    })

    model = TVAE(epochs=1)
    model.fit(data)
    conditions = {
        "column2": "b",
    }
    with pytest.raises(NotImplementedError):
        model.sample(30, conditions=conditions)
