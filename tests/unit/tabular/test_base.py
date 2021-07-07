from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdv.metadata.table import Table
from sdv.tabular.copulagan import CopulaGAN
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.ctgan import CTGAN, TVAE

MODELS = [
    CTGAN(epochs=1),
    TVAE(epochs=1),
    GaussianCopula(),
    CopulaGAN(epochs=1),
]


@patch('sdv.tabular.base.Table', spec_set=Table)
def test__init__passes_correct_parameters(metadata_mock):
    """
    Tests the ``BaseTabularModel.__init__`` method.

    The method should pass the parameters to the ``Table``
    class.

    Input:
    - rounding set to an int
    - max_value set to an int
    - min_value set to an int
    Side Effects:
    - ``instance._metadata`` should receive the correct parameters
    """
    # Run
    GaussianCopula(rounding=-1, max_value=100, min_value=-50)
    CTGAN(epochs=1, rounding=-1, max_value=100, min_value=-50)
    TVAE(epochs=1, rounding=-1, max_value=100, min_value=-50)
    CopulaGAN(epochs=1, rounding=-1, max_value=100, min_value=-50)

    # Asserts
    assert len(metadata_mock.mock_calls) == 5
    expected_calls = [
        call(field_names=None, primary_key=None, field_types=None, field_transformers=None,
             anonymize_fields=None, constraints=None, dtype_transformers={'O': 'one_hot_encoding'},
             rounding=-1, max_value=100, min_value=-50),
        call(field_names=None, primary_key=None, field_types=None, field_transformers=None,
             anonymize_fields=None, constraints=None, dtype_transformers={'O': 'label_encoding'},
             rounding=-1, max_value=100, min_value=-50),
        call(field_names=None, primary_key=None, field_types=None, field_transformers=None,
             anonymize_fields=None, constraints=None, dtype_transformers={'O': 'label_encoding'},
             rounding=-1, max_value=100, min_value=-50),
        call(field_names=None, primary_key=None, field_types=None, field_transformers=None,
             anonymize_fields=None, constraints=None, dtype_transformers={'O': 'label_encoding'},
             rounding=-1, max_value=100, min_value=-50)
    ]
    metadata_mock.assert_has_calls(expected_calls, any_order=True)


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
    conditions_series = pd.Series([25, 25, 25, 25, 25], name='column1')
    model._sample_batch = Mock()
    sampled = pd.DataFrame({
        'column1': [28, 28],
        'column2': [37, 37],
        'column3': [93, 93],
    })
    model._sample_batch.return_value = sampled
    model.fit(data)
    model._metadata = Mock()
    model._metadata.get_fields.return_value = ['column1', 'column2', 'column3']
    model._metadata.transform.return_value = pd.DataFrame()
    model._metadata.make_ids_unique.side_effect = lambda x: x

    # Run
    output = model.sample(5, conditions=conditions, graceful_reject_sampling=True)

    # Assert
    expected_output = pd.DataFrame({
        'column1': [28, 28],
        'column2': [37, 37],
        'column3': [93, 93],
    })
    _, args, kwargs = model._metadata.transform.mock_calls[0]
    pd.testing.assert_series_equal(args[0]['column1'], conditions_series)
    assert kwargs['on_missing_column'] == 'drop'
    model._metadata.transform.assert_called_once()
    model._sample_batch.assert_called_with(5, 100, 10, conditions, None, 0.01)
    pd.testing.assert_frame_equal(output, expected_output)


def test_sample_batches_transform_conditions_correctly():
    """Test that transformed conditions are batched correctly.

    The ``Sample`` method is expected to:
    - Return sampled data and call ``_sample_batch`` for every unique transformed
    condition group.

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
    conditions_series = pd.Series([25, 25, 25, 30, 30], name='column1')
    model._sample_batch = Mock()
    expected_outputs = [
        pd.DataFrame({
            'column1': [25, 25, 25],
            'column2': [37, 37, 37],
            'column3': [93, 93, 93],
        }), pd.DataFrame({
            'column1': [30],
            'column2': [37],
            'column3': [93],
        }), pd.DataFrame({
            'column1': [30],
            'column2': [37],
            'column3': [93],
        })
    ]
    model._sample_batch.side_effect = expected_outputs
    model.fit(data)
    model._metadata = Mock()
    model._metadata.get_fields.return_value = ['column1', 'column2', 'column3']
    model._metadata.transform.return_value = pd.DataFrame([
        [50], [50], [50], [60], [70]
    ], columns=['transformed_column'])

    # Run
    model.sample(5, conditions=conditions, graceful_reject_sampling=True)

    # Assert
    _, args, kwargs = model._metadata.transform.mock_calls[0]
    pd.testing.assert_series_equal(args[0]['column1'], conditions_series)
    assert kwargs['on_missing_column'] == 'drop'
    model._metadata.transform.assert_called_once()
    model._sample_batch.assert_any_call(
        3, 100, 10, {'column1': 25}, {'transformed_column': 50}, 0.01
    )
    model._sample_batch.assert_any_call(
        1, 100, 10, {'column1': 30}, {'transformed_column': 60}, 0.01
    )
    model._sample_batch.assert_any_call(
        1, 100, 10, {'column1': 30}, {'transformed_column': 70}, 0.01
    )
