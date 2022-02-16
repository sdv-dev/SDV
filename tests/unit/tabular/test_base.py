from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdv.metadata.table import Table
from sdv.tabular.base import BaseTabularModel
from sdv.tabular.copulagan import CopulaGAN
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.ctgan import CTGAN, TVAE

MODELS = [
    CTGAN(epochs=1),
    TVAE(epochs=1),
    GaussianCopula(),
    CopulaGAN(epochs=1),
]


class TestBaseTabularModel:

    def test__sample_batch_zero_valid(self):
        """Test the `BaseTabularModel._sample_batch` method with zero valid rows.

        Expect that the requested number of rows are returned, if the first `_sample_rows` call
        returns zero valid rows, and the second one returns enough valid rows.
        See https://github.com/sdv-dev/SDV/issues/285.

        Input:
            - num_rows = 5
            - condition on `column1` = 2
        Output:
            - The requested number of sampled rows (5).
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        valid_sampled_data = pd.DataFrame({
            "column1": [28, 28, 21, 1, 2],
            "column2": [37, 37, 1, 4, 5],
            "column3": [93, 93, 6, 4, 12],
        })
        gaussian_copula._sample_rows.side_effect = [(pd.DataFrame({}), 0), (valid_sampled_data, 5)]

        conditions = {
            'column1': 2,
            'column1': 2,
            'column1': 2,
            'column1': 2,
            'column1': 2,
        }

        # Run
        output = GaussianCopula._sample_batch(gaussian_copula, num_rows=5, conditions=conditions)

        # Assert
        assert gaussian_copula._sample_rows.call_count == 2
        assert len(output) == 5

    def test_sample_valid_num_rows(self):
        """Test the `BaseTabularModel.sample` method with a valid `num_rows` argument.

        Expect that the expected call to `_sample_batch` is made.

        Input:
            - num_rows = 5
        Output:
            - The requested number of sampled rows.
        Side Effect:
            - Call `_sample_batch` method with the expected number of rows.
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        valid_sampled_data = pd.DataFrame({
            'column1': [28, 28, 21, 1, 2],
            'column2': [37, 37, 1, 4, 5],
            'column3': [93, 93, 6, 4, 12],
        })
        gaussian_copula._sample_batch.return_value = valid_sampled_data

        # Run
        output = BaseTabularModel.sample(gaussian_copula, 5)

        # Assert
        assert gaussian_copula._sample_batch.called_once_with(5)
        assert len(output) == 5

    def test_sample_no_num_rows(self):
        """Test the `BaseTabularModel.sample` method with no `num_rows` input.

        Expect that an error is thrown.
        """
        # Setup
        model = BaseTabularModel()

        # Run and assert
        with pytest.raises(
                TypeError,
                match=r'sample\(\) missing 1 required positional argument: \'num_rows\''):
            model.sample()

    def test_sample_num_rows_none(self):
        """Test the `BaseTabularModel.sample` method with a `num_rows` input of `None`.

        Expect that a `ValueError` is thrown.

        Input:
            - num_rows = None
        Side Effect:
            - ValueError
        """
        # Setup
        model = BaseTabularModel()
        num_rows = None

        # Run and assert
        with pytest.raises(
                ValueError,
                match=r'You must specify the number of rows to sample \(e.g. num_rows=100\)'):
            model.sample(num_rows)


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
             anonymize_fields=None, constraints=None, dtype_transformers={'O': None},
             rounding=-1, max_value=100, min_value=-50),
        call(field_names=None, primary_key=None, field_types=None, field_transformers=None,
             anonymize_fields=None, constraints=None, dtype_transformers={'O': None},
             rounding=-1, max_value=100, min_value=-50),
        call(field_names=None, primary_key=None, field_types=None, field_transformers=None,
             anonymize_fields=None, constraints=None, dtype_transformers={'O': None},
             rounding=-1, max_value=100, min_value=-50)
    ]
    metadata_mock.assert_has_calls(expected_calls, any_order=True)


def test__sample_rows_previous_rows_appended_correctly():
    """Test the ``BaseTabularModel._sample_rows`` method.

    If ``_sample_rows`` is passed ``previous_rows``, then it
    should reset the index when appending them to the new
    sampled rows.

    Input:
    - num_rows is 5
    - previous_rows is a DataFrame of 3 existing rows.

    Output:
    - 5 sampled rows with index set to [0, 1, 2, 3, 4]
    """
    # Setup
    model = GaussianCopula()
    previous_data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': [4, 5, 6],
        'column3': [7, 8, 9]
    })
    new_data = pd.DataFrame({
        'column1': [4, 5],
        'column2': [7, 8],
        'column3': [10, 11]
    })
    model._metadata = Mock()
    model._sample = Mock()
    model._sample.return_value = new_data
    model._metadata.reverse_transform.return_value = new_data
    model._metadata.filter_valid = lambda x: x

    # Run
    sampled, num_valid = model._sample_rows(5, previous_rows=previous_data)

    # Assert
    expected = pd.DataFrame({
        'column1': [1, 2, 3, 4, 5],
        'column2': [4, 5, 6, 7, 8],
        'column3': [7, 8, 9, 10, 11]
    })
    assert num_valid == 5
    pd.testing.assert_frame_equal(sampled, expected)


@pytest.mark.parametrize('model', MODELS)
def test_fit_sets_num_rows(model):
    """Test ``fit`` sets ``_num_rows`` to the length of the data passed.

    The ``fit`` method is expected to:
        - Save the length of the data passed to the ``fit`` method in the ``_num_rows`` attribute.

    Input:
        - DataFrame

    Side Effects:
        - ``_num_rows`` is set to the length of the data passed to ``fit``.
    """
    # Setup
    _N_DATA_ROWS = 100
    data = pd.DataFrame({
        'column1': list(range(_N_DATA_ROWS)),
        'column2': list(range(_N_DATA_ROWS)),
        'column3': list(range(_N_DATA_ROWS))
    })

    # Run
    model.fit(data)

    # Assert
    assert model._num_rows == _N_DATA_ROWS


@pytest.mark.parametrize('model', MODELS)
def test__make_conditions_df_without_num_rows(model):
    """Test ``_make_conditions_df`` works correctly when ``num_rows`` is not passed.

    The ``_make_conditions_df`` method is expected to:
        - Return conditions as a ``DataFrame`` for every row in the data.

    Input:
        - Conditions

    Output:
        - Conditions as ``DataFrame``
    """
    # Setup
    _N_DATA_ROWS = 100
    conditions = {'column2': 'M'}
    expected_conditions = pd.DataFrame([conditions] * _N_DATA_ROWS)

    model._num_rows = _N_DATA_ROWS

    # Run
    result_conditions = model._make_conditions_df(conditions=conditions, num_rows=None)

    # Assert
    assert isinstance(result_conditions, pd.DataFrame)
    assert len(result_conditions) == _N_DATA_ROWS
    assert all(result_conditions == expected_conditions)


@pytest.mark.parametrize('model', MODELS)
def test__make_conditions_df_specifying_num_rows(model):
    """Test ``_make_conditions_df`` works correctly when ``num_rows`` is passed.

    The ``_make_conditions_df`` method is expected to:
    - Return as many condition rows as specified with ``num_rows`` as a ``DataFrame``.

    Input:
        - Conditions
        - Num_rows

    Output:
        - Conditions as ``DataFrame``
    """
    # Setup
    _N_DATA_ROWS = 100
    _NUM_ROWS = 10

    conditions = {'column2': 'M'}
    expected_conditions = pd.DataFrame([conditions] * _NUM_ROWS)

    model._num_rows = _N_DATA_ROWS

    # Run
    result_conditions = model._make_conditions_df(conditions=conditions, num_rows=_NUM_ROWS)

    # Assert
    assert isinstance(result_conditions, pd.DataFrame)
    assert len(result_conditions) == _NUM_ROWS
    assert all(result_conditions == expected_conditions)
