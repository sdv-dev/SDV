from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdv.metadata.table import Table
from sdv.sampling import Condition
from sdv.tabular.base import COND_IDX, BaseTabularModel
from sdv.tabular.copulagan import CopulaGAN
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.ctgan import CTGAN, TVAE
from tests.utils import DataFrameMatcher

MODELS = [
    CTGAN(epochs=1),
    TVAE(epochs=1),
    GaussianCopula(),
    CopulaGAN(epochs=1),
]


class TestBaseTabularModel:

    def test__sample_with_conditions_no_transformed_columns(self):
        """Test the ``BaseTabularModel.sample`` method with no transformed columns.

        When the transformed conditions DataFrame has no columns, expect that sample
        does not pass through any conditions when conditionally sampling.

        Setup:
            - Mock the ``_make_condition_dfs`` method to return a dataframe representing
              the expected conditions, and the ``get_fields`` method to return metadata
              fields containing the expected conditioned column.
            - Mock the ``_metadata.transform`` method to return an empty transformed
              conditions dataframe.
            - Mock the ``_conditionally_sample_rows`` method to return the expected
              sampled rows.
            - Mock the `make_ids_unique` to return the expected sampled rows.
        Input:
            - number of rows
            - one set of conditions
        Output:
            - the expected sampled rows
        Side Effects:
            - Expect ``_conditionally_sample_rows`` to be called with the given condition
              and a transformed_condition of None.
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        expected = pd.DataFrame(['a', 'a', 'a'])

        condition_dataframe = pd.DataFrame({'a': ['a', 'a', 'a']})
        gaussian_copula._make_condition_dfs.return_value = condition_dataframe
        gaussian_copula._metadata.get_fields.return_value = ['a']
        gaussian_copula._metadata.transform.return_value = pd.DataFrame({}, index=[0, 1, 2])
        gaussian_copula._conditionally_sample_rows.return_value = pd.DataFrame({
            'a': ['a', 'a', 'a'],
            COND_IDX: [0, 1, 2]})
        gaussian_copula._metadata.make_ids_unique.return_value = expected

        # Run
        out = GaussianCopula._sample_with_conditions(
            gaussian_copula, condition_dataframe, 100, None)

        # Asserts
        gaussian_copula._conditionally_sample_rows.assert_called_once_with(
            DataFrameMatcher(pd.DataFrame({COND_IDX: [0, 1, 2], 'a': ['a', 'a', 'a']})),
            {'a': 'a'},
            None,
            100,
            None,
        )
        pd.testing.assert_frame_equal(out, expected)

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

    def test_sample_conditions_with_multiple_conditions(self):
        """Test the `BaseTabularModel.sample_conditions` method with multiple condtions.

        When multiple condition dataframes are returned by `_make_condition_dfs`,
        expect `_sample_with_conditions` is called for each condition dataframe.

        Input:
            - 2 conditions with different columns
        Output:
            - The expected sampled rows
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)

        condition_values1 = {'cola': 'a'}
        condition1 = Condition(condition_values1, num_rows=2)
        sampled1 = pd.DataFrame({'a': ['a', 'a'], 'b': [1, 2]})

        condition_values2 = {'colb': 1}
        condition2 = Condition(condition_values2, num_rows=3)
        sampled2 = pd.DataFrame({'a': ['b', 'c', 'a'], 'b': [1, 1, 1]})

        expected = pd.DataFrame({
            'a': ['a', 'a', 'b', 'c', 'a'],
            'b': [1, 2, 1, 1, 1],
        })

        gaussian_copula._make_condition_dfs.return_value = [
            pd.DataFrame([condition_values1] * 2),
            pd.DataFrame([condition_values2] * 3),
        ]
        gaussian_copula._sample_with_conditions.side_effect = [
            sampled1,
            sampled2,
        ]

        # Run
        out = GaussianCopula.sample_conditions(gaussian_copula, [condition1, condition2])

        # Asserts
        gaussian_copula._sample_with_conditions.assert_has_calls([
            call(DataFrameMatcher(pd.DataFrame([condition_values1] * 2)), 100, None),
            call(DataFrameMatcher(pd.DataFrame([condition_values2] * 3)), 100, None),
        ])
        pd.testing.assert_frame_equal(out, expected)

    def test_sample_remaining_columns(self):
        """Test the `BaseTabularModel.sample_remaining_colmns` method.

        When a valid DataFrame is given, expect `_sample_with_conditions` to be called
        with the input DataFrame.

        Input:
            - DataFrame with condition column values populated.
        Output:
            - The expected sampled rows.
        Side Effects:
            - `_sample_with_conditions` is called once.
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)

        conditions = pd.DataFrame([{'cola': 'a'}] * 5)

        sampled = pd.DataFrame({
            'cola': ['a', 'a', 'a', 'a', 'a'],
            'colb': [1, 2, 1, 1, 1],
        })
        gaussian_copula._sample_with_conditions.return_value = sampled

        # Run
        out = GaussianCopula.sample_remaining_columns(gaussian_copula, conditions)

        # Asserts
        gaussian_copula._sample_with_conditions.assert_called_once_with(
            DataFrameMatcher(conditions), 100, None)
        pd.testing.assert_frame_equal(out, sampled)

    def test__sample_with_conditions_invalid_column(self):
        """Test the `BaseTabularModel._sample_with_conditions` method with an invalid column.

        When a condition has an invalid column, expect a ValueError.

        Setup:
            - Conditions DataFrame contains `colb` which is not present in the metadata.
        Input:
            - Conditions DataFrame with an invalid column.
        Side Effects:
            - A ValueError is thrown.
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        metadata_mock = Mock()
        metadata_mock.get_fields.return_value = {'cola': {}}
        gaussian_copula._metadata = metadata_mock

        conditions = pd.DataFrame([{'colb': 'a'}] * 5)

        # Run and Assert
        with pytest.raises(ValueError, match=(
                'Unexpected column name `colb`. '
                'Use a column name that was present in the original data.')):
            GaussianCopula._sample_with_conditions(gaussian_copula, conditions, 100, None)


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


@pytest.mark.parametrize('model', MODELS)
def test_sample_conditions_graceful_reject_sampling(model):
    data = pd.DataFrame({
        'column1': list(range(100)),
        'column2': list(range(100)),
        'column3': list(range(100))
    })

    conditions = [
        Condition(
            {'column1': "this is not used"},
            num_rows=5,
        )
    ]

    model._sample_batch = Mock()
    model._sample_batch.return_value = pd.DataFrame({
        "column1": [28, 28],
        "column2": [37, 37],
        "column3": [93, 93],
    })

    model.fit(data)
    output = model.sample_conditions(conditions)
    assert len(output) == 2, "Only expected 2 valid rows."


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


def test__sample_with_conditions_empty_transformed_conditions():
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
    output = model._sample_with_conditions(pd.DataFrame([conditions] * 5), 100, None)

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
    model._sample_batch.assert_called_with(5, 100, None, conditions, None, 0.01)
    pd.testing.assert_frame_equal(output, expected_output)


def test__sample_with_conditions_transform_conditions_correctly():
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

    condition_values = [25, 25, 25, 30, 30]
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
    model._sample_with_conditions(
        pd.DataFrame({'column1': condition_values}), 100, None)

    # Assert
    _, args, kwargs = model._metadata.transform.mock_calls[0]
    pd.testing.assert_series_equal(args[0]['column1'], conditions_series)
    assert kwargs['on_missing_column'] == 'drop'
    model._metadata.transform.assert_called_once()
    model._sample_batch.assert_any_call(
        3, 100, None, {'column1': 25}, {'transformed_column': 50}, 0.01
    )
    model._sample_batch.assert_any_call(
        1, 100, None, {'column1': 30}, {'transformed_column': 60}, 0.01
    )
    model._sample_batch.assert_any_call(
        1, 100, None, {'column1': 30}, {'transformed_column': 70}, 0.01
    )


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
def test__make_condition_dfs_without_num_rows(model):
    """Test ``_make_condition_dfs`` works correctly when ``num_rows`` is not passed.

    The ``_make_condition_dfs`` method is expected to:
        - Return conditions as a ``DataFrame`` for one row.

    Input:
        - Conditions

    Output:
        - Conditions as ``[DataFrame]``
    """
    # Setup
    column_values = {'column2': 'M'}
    conditions = [Condition(column_values=column_values)]
    expected_conditions = pd.DataFrame([column_values])

    # Run
    result_conditions_list = model._make_condition_dfs(conditions=conditions)

    # Assert
    assert len(result_conditions_list) == 1
    result_conditions = result_conditions_list[0]
    assert isinstance(result_conditions, pd.DataFrame)
    assert len(result_conditions) == 1
    assert all(result_conditions == expected_conditions)


@pytest.mark.parametrize('model', MODELS)
def test__make_condition_dfs_specifying_num_rows(model):
    """Test ``_make_condition_dfs`` works correctly when ``num_rows`` is passed.

    The ``_make_condition_dfs`` method is expected to:
    - Return as many condition rows as specified with ``num_rows`` as a ``DataFrame``.

    Input:
        - Conditions
        - Num_rows

    Output:
        - Conditions as ``[DataFrame]``
    """
    # Setup
    _NUM_ROWS = 10
    column_values = {'column2': 'M'}
    conditions = [Condition(column_values=column_values, num_rows=_NUM_ROWS)]
    expected_conditions = pd.DataFrame([column_values] * _NUM_ROWS)

    # Run
    result_conditions_list = model._make_condition_dfs(conditions=conditions)

    # Assert
    assert len(result_conditions_list) == 1
    result_conditions = result_conditions_list[0]
    assert isinstance(result_conditions, pd.DataFrame)
    assert len(result_conditions) == _NUM_ROWS
    assert all(result_conditions == expected_conditions)


@pytest.mark.parametrize('model', MODELS)
def test__make_condition_dfs_with_multiple_conditions_same_column(model):
    """Test ``_make_condition_dfs`` works correctly with multiple conditions.

    The ``_make_condition_dfs`` method is expected to:
        - Combine conditions for conditions with the same columns.

    Input:
        - Conditions

    Output:
        - Conditions as ``[DataFrame]``
    """
    # Setup
    column_values1 = {'column2': 'M'}
    column_values2 = {'column2': 'N'}
    conditions = [
        Condition(column_values=column_values1, num_rows=2),
        Condition(column_values=column_values2, num_rows=3),
    ]
    expected_conditions = pd.DataFrame([column_values1] * 2 + [column_values2] * 3)

    # Run
    result_conditions_list = model._make_condition_dfs(conditions=conditions)

    # Assert
    assert len(result_conditions_list) == 1
    result_conditions = result_conditions_list[0]
    assert isinstance(result_conditions, pd.DataFrame)
    assert len(result_conditions) == 5
    assert all(result_conditions == expected_conditions)


@pytest.mark.parametrize('model', MODELS)
def test__make_condition_dfs_with_multiple_conditions_different_columns(model):
    """Test ``_make_condition_dfs`` works correctly with multiple conditions.

    The ``_make_condition_dfs`` method is expected to:
        - Return multiple DataFrames if conditions are not able to be combined.

    Input:
        - Conditions

    Output:
        - Conditions as ``[DataFrame]``
    """
    # Setup
    column_values1 = {'column2': 'M'}
    column_values2 = {'column3': 'N'}
    conditions = [
        Condition(column_values=column_values1, num_rows=2),
        Condition(column_values=column_values2, num_rows=3),
    ]
    expected_conditions1 = pd.DataFrame([column_values1] * 2)
    expected_conditions2 = pd.DataFrame([column_values2] * 3)

    # Run
    result_conditions_list = model._make_condition_dfs(conditions=conditions)

    # Assert
    assert len(result_conditions_list) == 2

    result_conditions1 = result_conditions_list[0]
    assert isinstance(result_conditions1, pd.DataFrame)
    assert len(result_conditions1) == 2
    assert all(result_conditions1 == expected_conditions1)

    result_conditions2 = result_conditions_list[1]
    assert isinstance(result_conditions2, pd.DataFrame)
    assert len(result_conditions2) == 3
    assert all(result_conditions2 == expected_conditions2)
