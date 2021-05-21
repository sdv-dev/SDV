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


def test_sample_empty_transformed_conditions():
    """Test that None is passed to ``_sample_batch`` if transformed conditions are empty.

    The ``Sample`` method is expected to:
    - Return sampled data and pass None to ``sample_batch`` as the
    ``transformed_conditions``.

    Input:
    - Number of rows to sample
    - Conditions

    Output:
    - Sampled data
    """
    # Setup
    model = GaussianCopula()
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    conditions = {
        'column1': 25
    }
    conditions_df = pd.DataFrame([
        [0, 25], [1, 25], [2, 25], [3, 25], [4, 25]
    ], columns=['__condition_idx__', 'column1'])
    model._sample_batch = Mock()
    expected_output = pd.DataFrame({
        'column1': [28, 28],
        'column2': [37, 37],
        'column3': [93, 93],
    })
    model._sample_batch.return_value = expected_output
    model.fit(data)
    model._metadata = Mock()
    model._metadata.get_fields.return_value = ['column1', 'column2', 'column3']
    model._metadata.transform.return_value = pd.DataFrame()

    # Run
    output = model.sample(5, conditions=conditions, graceful_reject_sampling=True)

    # Assert
    _, args, kwargs = model._metadata.transform.mock_calls[0]
    assert args[0].equals(conditions_df)
    assert kwargs['on_missing_column'] == 'drop'
    model._metadata.transform.assert_called_once()
    model._sample_batch.assert_called_with(5, 100, 10, conditions, None, 0.01)
    assert output.equals(expected_output)


def test_sample_batches_transform_conditions_correctly():
    """Test that transformed conditions are batched correctly.

    The ``Sample`` method is expected to:
    - Return sampled data and call ``_sample_batch`` for every grouped condition
    value with the correct transformed condition.

    Input:
    - Number of rows to sample
    - Conditions

    Output:
    - Sampled data
    """
    # Setup
    model = GaussianCopula()
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    conditions = {
        'column1': [25, 25, 25, 30, 30]
    }
    conditions_df = pd.DataFrame([
        [0, 25], [1, 25], [2, 25], [3, 30], [4, 30]
    ], columns=['__condition_idx__', 'column1'])
    model._sample_batch = Mock()
    expected_output = pd.DataFrame({
        'column1': [28, 28],
        'column2': [37, 37],
        'column3': [93, 93],
    })
    model._sample_batch.return_value = expected_output
    model.fit(data)
    model._metadata = Mock()
    model._metadata.get_fields.return_value = ['column1', 'column2', 'column3']
    model._metadata.transform.return_value = pd.DataFrame([
        [50], [50], [50], [60], [70]
    ], columns=['transformed_column'])

    # Run
    output = model.sample(5, conditions=conditions, graceful_reject_sampling=True)

    # Assert
    _, args, kwargs = model._metadata.transform.mock_calls[0]
    assert args[0].equals(conditions_df)
    assert kwargs['on_missing_column'] == 'drop'
    model._metadata.transform.assert_called_once()
    model._sample_batch.assert_any_call(
        3, 100, 10, {'column1': 25}, {'transformed_column': 50}, 0.01
    )
    model._sample_batch.assert_any_call(
        2, 100, 10, {'column1': 30}, {'transformed_column': 60}, 0.01
    )
    assert output.equals(expected_output)
