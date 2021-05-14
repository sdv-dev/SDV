from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.constraints import UniqueCombinations
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
def test_conditional_sampling_graceful_reject_sampling_True_dict(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)
    conditions = {
        'column1': 28,
        'column2': 37,
        'column3': 93
    }

    with pytest.raises(ValueError):
        model.sample(1, conditions=conditions, graceful_reject_sampling=True)


@pytest.mark.parametrize('model', MODELS)
def test_conditional_sampling_graceful_reject_sampling_True_dataframe(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)
    conditions = pd.DataFrame({
        'column1': [28],
        'column2': [37],
        'column3': [93]
    })

    with pytest.raises(ValueError):
        model.sample(conditions=conditions, graceful_reject_sampling=True)


@pytest.mark.parametrize('model', MODELS)
def test_conditional_sampling_graceful_reject_sampling_False_dict(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)
    conditions = {
        'column1': 28,
        'column2': 37,
        'column3': 93
    }

    with pytest.raises(ValueError):
        model.sample(1, conditions=conditions)


@pytest.mark.parametrize('model', MODELS)
def test_conditional_sampling_graceful_reject_sampling_False_dataframe(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    model.fit(data)
    conditions = pd.DataFrame({
        'column1': [28],
        'column2': [37],
        'column3': [93]
    })

    with pytest.raises(ValueError):
        model.sample(conditions=conditions)

def test_conditional_sampling_properly_handles_constraints():
    """Test that the ``sample`` method handles constraints with conditions.

    The ``sample`` method is expected to properly apply constraint
    transformations by dropping columns that cannot be conditonally sampled
    on due to them being part of a constraint.

    Input:
    - Conditions
    Side Effects:
    - Correct columns to condition on are passed to underlying sample method
    """
    # Setup
    constraint = UniqueCombinations(
        columns=['city', 'state'],
        handling_strategy='transform'
    )
    data = pd.DataFrame({
        'city': ['LA', 'SF', 'CHI', 'LA', 'LA'],
        'state': ['CA', 'CA', 'IL', 'CA', 'CA'],
        'age': [27, 28, 26, 21, 30]
    })
    model = GaussianCopula(constraints=[constraint])
    conditions = {'age': 30, 'state': 'CA'}
    reverse_transformed_data = pd.DataFrame({
        'city': ['LA', 'SF', 'SF', 'LA', 'LA'],
        'state': ['CA', 'CA', 'CA', 'CA', 'CA'],
        'age': [30, 30, 30, 30, 30]
    })
    expected_transformed_conditions = {'age': 30}
    model.fit(data)
    model._model.sample = Mock()
    model._model.sample.return_value = pd.DataFrame()
    model._metadata.reverse_transform = Mock()
    model._metadata.reverse_transform.return_value = reverse_transformed_data

    # Run
    sampled_data = model.sample(5, conditions=conditions)

    # Assert
    model._model.sample.assert_called_once_with(5, conditions=expected_transformed_conditions)
