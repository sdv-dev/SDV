from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.tabular.copulagan import CopulaGAN
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.ctgan import CTGAN, TVAE

MODELS = [
    CTGAN(epochs=1),
    TVAE(epochs=1),
    GaussianCopula(),
    CopulaGAN(epochs=1),
]


@pytest.mark.parametrize('model', MODELS)
def test_conditional_sampling_graceful_reject_sampling(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    conditions = {
        'column1': "this is not used"
    }

    model._sample_batch = Mock()
    model._sample_batch.return_value = pd.DataFrame({
        "column1": [28, 28],
        "column2": [37, 37],
        "column3": [93, 93],
    })

    model.fit(data)
    output = model.sample(5, conditions=conditions, graceful_reject_sampling=True)
    assert len(output) == 2, "Only expected 2 valid rows."
    with pytest.raises(ValueError):
        model.sample(5, conditions=conditions, graceful_reject_sampling=False)
