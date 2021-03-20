import pandas as pd
import pytest


from sdv.demo import load_demo
from sdv.tabular.ctgan import CTGAN
from sdv.tabular.ctgan import TVAE
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.copulagan import CopulaGAN

MODELS = [
    CTGAN(epochs=1),
    TVAE(epochs=1),
    GaussianCopula(),
    CopulaGAN(epochs=1),
]


@pytest.mark.parametrize("model", MODELS)
def test_conditional_sampling_graceful_reject_sampling_True_dict(model):
    data = pd.DataFrame({
        "column1": list(range(100)) * 2,
        "column2": list(range(100)) * 2,
        "column3": list(range(100)) * 2
    })

    model.fit(data)
    conditions = {
        "column1": 28,
        "column2": 37,
        "column3": 93
    }

    samples = model.sample(1, conditions=conditions)
    assert len(samples) == 0


@pytest.mark.parametrize("model", MODELS)
def test_conditional_sampling_graceful_reject_sampling_True_dataframe(model):
    data = pd.DataFrame({
        "column1": list(range(100)),
        "column2": list(range(100)),
        "column3": list(range(100))
    })

    model.fit(data)
    conditions = pd.DataFrame({
        "column1": [28],
        "column2": [37],
        "column3": [93]
    })

    samples = model.sample(conditions=conditions)
    assert len(samples) == 0


@pytest.mark.parametrize("model", MODELS)
def test_conditional_sampling_graceful_reject_sampling_False_dict(model):
    data = pd.DataFrame({
        "column1": list(range(100)),
        "column2": list(range(100)),
        "column3": list(range(100))
    })

    model.fit(data)
    conditions = {
        "column1": 28,
        "column2": 37,
        "column3": 93
    }

    with pytest.raises(ValueError):
        model.sample(1, conditions=conditions, graceful_reject_sampling=False)


@pytest.mark.parametrize("model", MODELS)
def test_conditional_sampling_graceful_reject_sampling_False_dataframe(model):
    data = pd.DataFrame({
        "column1": list(range(100)),
        "column2": list(range(100)),
        "column3": list(range(100))
    })

    model.fit(data)
    conditions = pd.DataFrame({
        "column1": [28],
        "column2": [37],
        "column3": [93]
    })

    with pytest.raises(ValueError):
        model.sample(conditions=conditions, graceful_reject_sampling=False)
