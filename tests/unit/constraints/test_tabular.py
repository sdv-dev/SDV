"""Tests for the sdv.constraints.tabular module."""

import operator
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.constraints.errors import (
    FunctionError, InvalidFunctionError, MissingConstraintColumnError)
from sdv.constraints.tabular import (
    FixedCombinations, FixedIncrements, Inequality, Negative, OneHotEncoding, Positive, Range,
    ScalarInequality, ScalarRange, Unique, _validate_inputs_custom_constraint,
    create_custom_constraint)


def dummy_transform_table(table_data):
    return table_data


def dummy_reverse_transform_table(table_data):
    return table_data


def dummy_is_valid_table(table_data):
    return [True] * len(table_data)


def dummy_transform_table_column(table_data, column):
    return table_data


def dummy_reverse_transform_table_column(table_data, column):
    return table_data


def dummy_is_valid_table_column(table_data, column):
    return [True] * len(table_data[column])


def dummy_transform_column(column_data):
    return column_data


def dummy_reverse_transform_column(column_data):
    return column_data


def dummy_is_valid_column(column_data):
    return [True] * len(column_data)


class TestCreateCustomConstraint():

    def test__validate_inputs_custom_constraint_is_valid_incorrect(self):
        """Test validation when ``is_valid_fn`` is not callable.

        Input:
        - ``is_valid_fn`` as a non-callable object
        Side effects:
        - Raise ValueError
        """
        err_msg = '`is_valid` must be a function.'
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            _validate_inputs_custom_constraint(is_valid_fn=10)

    def test__validate_inputs_custom_constraint_transform_none(self):
        """Test validation when ``transform_fn`` not passed.

        Input:
        - ``is_valid`` as callable
        - ``reverse_transform_fn`` as callable
        Side effects:
        - Raise ValueError
        """
        err_msg = 'Missing parameter `transform_fn`.'
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            _validate_inputs_custom_constraint(is_valid_fn=sorted, reverse_transform_fn=sorted)

    def test__validate_inputs_custom_constraint_reverse_transform_none(self):
        """Test validation when ``reverse_transform_fn`` is not passed.

        Input:
        - ``is_valid`` as callable
        - ``transform_fn`` as callable
        Side effects:
        - Raise ValueError
        """
        err_msg = 'Missing parameter `reverse_transform_fn`.'
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            _validate_inputs_custom_constraint(is_valid_fn=sorted, transform_fn=sorted)

    def test__validate_inputs_custom_constraint_transform_not_callable(self):
        """Test validation when ``transform_fn`` is not callable.

        Input:
        - ``is_valid`` as callable
        - ``transform_fn`` as non-callable
        - ``reverse_transform_fn`` as callable
        Side effects:
        - Raise ValueError
        """
        err_msg = '`transform_fn` must be a function.'
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            _validate_inputs_custom_constraint(
                is_valid_fn=sorted, transform_fn='a', reverse_transform_fn=sorted)

    def test__validate_inputs_custom_constraint_reverse_transform_not_callable(self):
        """Test validation when ``reverse_transform_fn`` is not callable.

        Input:
        - ``is_valid`` as callable
        - ``transform_fn`` as callable
        - ``reverse_transform_fn`` as non-callable
        Side effects:
        - Raise ValueError
        """
        err_msg = '`reverse_transform_fn` must be a function.'
        # Run / Assert
        with pytest.raises(ValueError, match=err_msg):
            _validate_inputs_custom_constraint(
                is_valid_fn=sorted, transform_fn=sorted, reverse_transform_fn=10)

    @patch('sdv.constraints.tabular._validate_inputs_custom_constraint')
    def test_create_custom_constraint(self, mock_validate):
        """Test ``CustomConstraint`` object is correctly created.

        Input:
        - ``is_valid`` as callable
        - ``transform_fn`` as callable
        - ``reverse_transform_fn`` as callable
        Side effects:
        - call ``_validate_inputs_custom_constraint``
        Output:
        - ``CustomConstraint`` object
        """
        # Run
        out = create_custom_constraint(sorted, sorted, sorted)

        # Assert
        mock_validate.assert_called_once_with(sorted, sorted, sorted)
        assert hasattr(out, 'is_valid')
        assert hasattr(out, 'transform')
        assert hasattr(out, 'reverse_transform')
        assert hasattr(out, '_transform')

    def test_create_custom_constraint_is_valid(self):
        """Test ``is_valid`` method of ``CustomConstraint``.

        Call ``create_custom_constraint`` on a ``is_valid`` function and confirm
        the produced ``CustomConstraint`` correctly applied ``is_valid``.

        Input:
        - pd.DataFrame
        Output:
        - pd.Series of booleans, describing whether the values of the input are valid
        """
        # Setup
        custom_constraint = create_custom_constraint(
            lambda _, x: pd.Series([True if x_i >= 0 else False for x_i in x['col']])
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        valid_out = custom_constraint.is_valid(data)

        # Assert
        expected_out = pd.Series([False, True, True, True, False])
        pd.testing.assert_series_equal(valid_out, expected_out)

    def test_create_custom_constraint_is_valid_wrong_shape(self):
        """Test ``is_valid`` method of ``CustomConstraint`` which produces data of wrong shape.

        Call ``create_custom_constraint`` on an invalid ``is_valid`` function.

        Input:
        - pd.DataFrame
        Raises:
        - InvalidFunctionError
        """
        # Setup
        custom_constraint = create_custom_constraint(
            lambda _, x: pd.Series([True, True, True])
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        err_msg = '`is_valid_fn` did not produce exactly 1 True/False value for each row.'
        with pytest.raises(InvalidFunctionError, match=err_msg):
            custom_constraint.is_valid(data)

    def test_create_custom_constraint_is_valid_not_a_series(self):
        """Test ``is_valid`` method of ``CustomConstraint`` which produces a list.

        Call ``create_custom_constraint`` on an ``is_valid`` function that returns
        a list instead of a ``pandas.Series``.

        Input:
        - pd.DataFrame
        Raises:
        - ValueError
        """
        # Setup
        custom_constraint = create_custom_constraint(
            lambda _, x: list([True if x_i >= 0 else False for x_i in x['col']])
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        err_msg = (
            "The custom 'is_valid' function returned an unsupported type. "
            'The returned object must be a pandas.Series'
        )
        with pytest.raises(ValueError, match=err_msg):
            custom_constraint.is_valid(data)

    def test_create_custom_constraint_transform(self):
        """Test ``transform`` method of ``CustomConstraint``.

        Call ``create_custom_constraint`` on a ``transform`` function and confirm
        the produced ``CustomConstraint`` correctly applied ``transform``.

        Input:
        - pd.DataFrame
        Output:
        - pd.DataFrame of transformed values
        """
        # Setup
        custom_constraint = create_custom_constraint(
            lambda _, x: pd.Series([True] * 5),
            lambda _, x: pd.DataFrame({'col': x['col'] ** 2}),
            lambda _, x: pd.DataFrame({'col': x['col'] ** 1 / 2}),
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        transform_out = custom_constraint.transform(data)

        # Assert
        expected_out = pd.DataFrame({'col': [100, 1, 0, 9, .25]})
        pd.testing.assert_frame_equal(transform_out, expected_out)

    def test_create_custom_constraint_transform_not_defined(self):
        """Test ``transform`` method of ``CustomConstraint`` when it wasn't defined.

        Call ``create_custom_constraint`` on a not defined ``transform`` function.

        Input:
        - pd.DataFrame
        Raises:
        - Original data
        """
        # Setup
        custom_constraint = create_custom_constraint(
            lambda _, x: pd.Series([True] * 5)
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        out = custom_constraint.transform(data)

        # Assert
        pd.testing.assert_frame_equal(data, out)

    def test_create_custom_constraint_transform_wrong_shape(self):
        """Test ``transform`` method of ``CustomConstraint`` which produces data of wrong shape.

        Call ``create_custom_constraint`` on an invalid ``transform`` function.

        Input:
        - pd.DataFrame
        Raises:
        - InvalidFunctionError
        """
        # Setup
        custom_constraint = create_custom_constraint(
            lambda _, x: pd.Series([True] * 5),
            lambda _, x: pd.DataFrame({'col': [1, 2, 3]}),
            sorted
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        err_msg = 'Transformation did not produce the same number of rows as the original'
        with pytest.raises(InvalidFunctionError, match=err_msg):
            custom_constraint.transform(data)

    def test_create_custom_constraint_incorrect_transform(self):
        """Test ``transform`` method of ``CustomConstraint`` with incorrect transform.

        Call ``create_custom_constraint`` on an incorrect ``transform`` function, such as
        only accepting 1 argument instead of the required 2.

        Input:
        - pd.DataFrame
        Raises:
        - FunctionError
        """
        # Setup
        custom_constraint = create_custom_constraint(
            lambda _, x: pd.Series([True] * 5),
            lambda _: pd.DataFrame({'col': [1, 2, 3]}),
            sorted
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        with pytest.raises(FunctionError):
            custom_constraint.transform(data)

    def test_create_custom_constraint_reverse_transform(self):
        """Test ``reverse_transform`` method of ``CustomConstraint``.

        Call ``create_custom_constraint`` on a ``reverse_transform`` function and confirm
        the produced ``CustomConstraint`` correctly applied ``reverse_transform``.

        Input:
        - pd.DataFrame
        Output:
        - pd.DataFrame of transformed values
        """
        # Setup
        custom_constraint = create_custom_constraint(
            sorted,
            sorted,
            lambda _, x: pd.DataFrame({'col': x['col'] ** 2})
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        transformed_out = custom_constraint.reverse_transform(data)

        # Assert
        expected_out = pd.DataFrame({'col': [100, 1, 0, 9, .25]})
        pd.testing.assert_frame_equal(transformed_out, expected_out)

    def test_create_custom_constraint_reverse_transform_not_defined(self):
        """Test ``reverse_transform`` method of ``CustomConstraint`` when it wasn't defined.

        Call ``create_custom_constraint`` on a not defined ``reverse_transform`` function.

        Input:
        - pd.DataFrame
        Output:
        - Original data
        """
        # Setup
        custom_constraint = create_custom_constraint(sorted)('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        out = custom_constraint.reverse_transform(data)

        # Assert
        pd.testing.assert_frame_equal(data, out)

    def test_create_custom_constraint_reverse_transform_wrong_shape(self):
        """Test invalid ``reverse_transform`` method of ``CustomConstraint``

        Call ``create_custom_constraint`` on a ``reverse_transform`` function
        which produces data of the wrong shape.

        Input:
        - pd.DataFrame
        Raises:
        - InvalidFunctionError
        """
        # Setup
        custom_constraint = create_custom_constraint(
            sorted,
            sorted,
            lambda _, x: pd.DataFrame({'col': [1, 2, 3]})
        )('col')
        data = pd.DataFrame({'col': [-10, 1, 0, 3, -.5]})

        # Run
        err_msg = 'Reverse transform did not produce the same number of rows as the original.'
        with pytest.raises(InvalidFunctionError, match=err_msg):
            custom_constraint.reverse_transform(data)


class TestFixedCombinations():

    def test___init__(self):
        """Test the ``FixedCombinations.__init__`` method.

        It is expected to create a new Constraint instance and receiving the names of
        the columns that need to produce fixed combinations.

        Side effects:
        - instance._colums == columns
        """
        # Setup
        columns = ['b', 'c']

        # Run
        instance = FixedCombinations(column_names=columns)

        # Assert
        assert instance._columns == columns
        assert instance.constraint_columns == tuple(columns)

    def test___init__with_one_column(self):
        """Test the ``FixedCombinations.__init__`` method with only one constraint column.

        Expect a ``ValueError`` because FixedCombinations requires at least two
        constraint columns.

        Side effects:
        - A ValueError is raised
        """
        # Setup
        columns = ['c']

        # Run and assert
        with pytest.raises(ValueError):
            FixedCombinations(column_names=columns)

    def test__fit(self):
        """Test the ``FixedCombinations._fit`` method.

        The ``FixedCombinations.fit`` method is expected to:
        - Call ``FixedCombinations._valid_separator``.
        - Find a valid separator for the data and generate the joint column name.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)

        # Run
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i'],
            'b#c': ['1', '2', '3'],
        })
        instance._fit(table_data)

        # Asserts
        expected_combinations = pd.DataFrame({
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        assert instance._separator == '##'
        assert instance._joint_column == 'b##c'
        pd.testing.assert_frame_equal(instance._combinations, expected_combinations)

    def test_is_valid_true(self):
        """Test the ``FixedCombinations.is_valid`` method.

        If the input data satisfies the constraint, result is a series of ``True`` values.

        Input:
        - Table data (pandas.DataFrame), satisfying the constraint.
        Output:
        - Series of ``True`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        out = instance.is_valid(table_data)

        expected_out = pd.Series([True, True, True], name='b#c')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false(self):
        """Test the ``FixedCombinations.is_valid`` method.

        If the input data doesn't satisfy the constraint, result is a series of ``False`` values.

        Input:
        - Table data (pandas.DataFrame), which does not satisfy the constraint.
        Output:
        - Series of ``False`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        incorrect_table = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['D', 'E', 'F'],
            'c': ['g', 'h', 'i']
        })
        out = instance.is_valid(incorrect_table)

        # Assert
        expected_out = pd.Series([False, False, False], name='b#c')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_non_string_true(self):
        """Test the ``FixedCombinations.is_valid`` method with non string columns.

        If the input data satisfies the constraint, result is a series of ``True`` values.

        Input:
        - Table data (pandas.DataFrame), satisfying the constraint.
        Output:
        - Series of ``True`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        out = instance.is_valid(table_data)

        expected_out = pd.Series([True, True, True], name='b#c#d')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_non_string_false(self):
        """Test the ``FixedCombinations.is_valid`` method with non string columns.

        If the input data doesn't satisfy the constraint, result is a series of ``False`` values.

        Input:
        - Table data (pandas.DataFrame), which does not satisfy the constraint.
        Output:
        - Series of ``False`` values (pandas.Series)
        Side effects:
        - Since the ``is_valid`` method needs ``self._combinations``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        incorrect_table = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [6, 7, 8],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        out = instance.is_valid(incorrect_table)

        # Assert
        expected_out = pd.Series([False, False, False], name='b#c#d')
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform(self):
        """Test the ``FixedCombinations.transform`` method.

        It is expected to return a Table data with the columns concatenated by the separator.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data transformed, with the columns concatenated (pandas.DataFrame)
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        out = instance.transform(table_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out_a = pd.Series(['a', 'b', 'c'], name='a')
        pd.testing.assert_series_equal(expected_out_a, out['a'])
        try:
            [uuid.UUID(u) for c, u in out['b#c'].items()]
        except ValueError:
            assert False

    def test_transform_non_string(self):
        """Test the ``FixedCombinations.transform`` method with non strings.

        It is expected to return a Table data with the columns concatenated by the separator.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data transformed, with the columns as UUIDs.
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        out = instance.transform(table_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out_a = pd.Series(['a', 'b', 'c'], name='a')
        pd.testing.assert_series_equal(expected_out_a, out['a'])
        try:
            [uuid.UUID(u) for c, u in out['b#c#d'].items()]
        except ValueError:
            assert False

    def test_transform_not_all_columns_provided(self):
        """Test the ``FixedCombinations.transform`` method.

        If some of the columns needed for the transform are missing, and it will raise a
        ``MissingConstraintColumnError``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Raises ``MissingConstraintColumnError``.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run/Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame({'a': ['a', 'b', 'c']}))

    def test_reverse_transform(self):
        """Test the ``FixedCombinations.reverse_transform`` method.

        It is expected to return the original data separating the concatenated columns.

        Input:
        - Table data transformed (pandas.DataFrame)
        Output:
        - Original table data, with the concatenated columns separated (pandas.DataFrame)
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        transformed_data = instance.transform(table_data)
        out = instance.reverse_transform(transformed_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_non_string(self):
        """Test the ``FixedCombinations.reverse_transform`` method with a non string column.

        It is expected to return the original data separating the concatenated columns.

        Input:
        - Table data transformed (pandas.DataFrame)
        Output:
        - Original table data, with the concatenated columns separated (pandas.DataFrame)
        Side effects:
        - Since the ``transform`` method needs ``self._joint_column``, method ``fit``
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        columns = ['b', 'c', 'd']
        instance = FixedCombinations(column_names=columns)
        instance.fit(table_data)

        # Run
        transformed_data = instance.transform(table_data)
        out = instance.reverse_transform(transformed_data)

        # Assert
        assert instance._combinations_to_uuids is not None
        assert instance._uuids_to_combinations is not None
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6]
        })
        pd.testing.assert_frame_equal(expected_out, out)


class TestInequality():

    def test__validate_inputs_incorrect_column(self):
        """Test the ``_validate_inputs`` method.

        Ensure the method crashes when one of the passed columns is not a string.

        Input:
        - a string
        - a non-string
        - a bool
        Side effect:
        - Raise ``ValueError`` because column names must be strings
        """
        # Run / Assert
        with pytest.raises(ValueError):
            Inequality._validate_inputs(
                low_column_name='a', high_column_name=['b', 'c'], strict_boundaries=True
            )

    def test__validate_inputs_incorrect_strict_boundaries(self):
        """Test the ``_validate_inputs`` method.

        Ensure the method crashes when ``strict_boundaries`` is not a bool.

        Input:
        - a string
        - a string
        - a non-bool
        Side effect:
        - Raise ``ValueError`` because ``strict_boundaries`` must be a boolean
        """
        # Run / Assert
        with pytest.raises(ValueError):
            Inequality._validate_inputs(
                low_column_name='a', high_column_name='b', strict_boundaries=None
            )

    @patch('sdv.constraints.tabular.Inequality._validate_inputs')
    def test___init___(self, mock_validate):
        """Test the ``Inequality.__init__`` method.

        The passed arguments should be stored as attributes.

        Input:
        - low_column_name and high_column_name should be two column names
        Side effects:
        - _low_column_name and _high_column_name are set to the input column names
        - _diff_column_name is set to '_low_column_name#_high_column_name'
        - _operator is set to the default np.greater_equal
        - _dtype and _is_datetime are None
        - _validate_inputs is called once
        """
        # Run
        instance = Inequality(low_column_name='a', high_column_name='b')

        # Asserts
        assert instance._low_column_name == 'a'
        assert instance._high_column_name == 'b'
        assert instance._diff_column_name == 'a#b'
        assert instance._operator == np.greater_equal
        assert instance._dtype is None
        assert instance._is_datetime is None
        mock_validate.assert_called_once_with('a', 'b', False)

    def test___init___strict_boundaries_true(self):
        """Test the ``Inequality.__init__`` method.

        Ensure that ``_operator`` is set to ``np.greater``
        when ``strict_boundaries`` is set to ``True``.

        Input:
        - low = 'a'
        - high = 'b'
        - strict_boundaries = True
        """
        # Run
        instance = Inequality(low_column_name='a', high_column_name='b', strict_boundaries=True)

        # Assert
        assert instance._operator == np.greater

    def test__get_is_datetime_incorrect_data(self):
        """Test the ``Inequality._get_is_datetime`` method.

        Ensure that if one of the low/high columns is datetime, both of them are.

        Input:
        - Table data.
        Side Effect:
        - Raises ``ValueError`` if only one of the low/high columns is datetime.
        """
        # Setupy
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01']),
            'b': ['a']
        })
        instance = Inequality(low_column_name='a', high_column_name='b')

        # Run / Assert
        err_msg = 'Both high and low must be datetime.'
        with pytest.raises(ValueError, match=err_msg):
            instance._get_is_datetime(table_data)

    def test__validate_columns_exist_incorrect_columns(self):
        """Test the ``Inequality._validate_columns_exist`` method.

        This method raises an error if ``low_column_name`` or ``high_column_name`` do not exist.

        Input:
        - Table with given data.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6]
        })
        instance = Inequality(low_column_name='a', high_column_name='c')

        # Run / Assert
        with pytest.raises(KeyError):
            instance._validate_columns_exist(table_data)

    def test__fit(self):
        """Test the ``Inequality._fit`` method.

        The method should learn the ``dtype`` of ``_column_name`` and ``_is_datetime``.

        Input:
        - Table data with integers.
        Side Effect:
        - _validate_columns_exist should be called once
        - _get_is_datetime should be called once
        - _is_datetime should receive the output of _get_is_datetime
        - _dtype should be a list of integer dtypes.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6]
        })
        instance = Inequality(low_column_name='a', high_column_name='b')
        instance._validate_columns_exist = Mock()
        instance._get_is_datetime = Mock(return_value='abc')

        # Run
        instance._fit(table_data)

        # Assert
        instance._validate_columns_exist.assert_called_once_with(table_data)
        instance._get_is_datetime.assert_called_once_with(table_data)
        assert instance._is_datetime == 'abc'
        assert instance._dtype == pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS

    def test__fit_floats(self):
        """Test the ``Inequality._fit`` method.

        The attribute ``_dtype`` should be float when ``high_column_name`` contains floats.

        Input:
        - Table data with floats.
        Side Effect:
        - _dtype should be a float dtype.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4., 5., 6.]
        })
        instance = Inequality(low_column_name='a', high_column_name='b')

        # Run
        instance._fit(table_data)

        # Assert
        assert instance._dtype == np.dtype('float')

    def test__fit_datetime(self):
        """Test the ``Inequality._fit`` method.

        The attribute ``_dtype`` should be datetime when ``high_column_name`` contains datetimes.

        Input:
        - Table data with datetimes.
        Side Effect:
        - _dtype should be a list of datetime dtypes.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01']),
            'b': pd.to_datetime(['2020-01-02'])
        })
        instance = Inequality(low_column_name='a', high_column_name='b')

        # Run
        instance._fit(table_data)

        # Assert
        assert instance._dtype == np.dtype('<M8[ns]')

    def test_is_valid(self):
        """Test the ``Inequality.is_valid`` method.

        The method should return True when ``high_column_name`` column is greater or equal to
        ``low_column_name`` or the row contains nan, otherwise return False.

        Input:
        - Table with a mixture of valid and invalid rows, as well as np.nans.
        Output:
        - False should be returned for the strictly invalid rows and True for the rest.
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, np.nan, 3, 4, None, 6, 8, 0],
            'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
            'c': [7, 8, 9, 10, 11, 12, 13, 14]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, True, False, True, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_strict_boundaries_True(self):
        """Test the ``Inequality.is_valid`` method with ``strict_boundaries = True``.

        The method should return True when ``high_column_name`` column is greater than
        ``low_column_name`` or the row contains nan, otherwise return False.

        Input:
        - Table with a mixture of valid and invalid rows, as well as np.nans.
        Output:
        - False should be returned for the non-strictly invalid rows and True for the rest.
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b', strict_boundaries=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, np.nan, 3, 4, None, 6, 8, 0],
            'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
            'c': [7, 8, 9, 10, 11, 12, 13, 14]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, True, False, False, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_datetimes(self):
        """Test the ``Inequality.is_valid`` method with datetimes.

        The method should return True when ``high_column_name`` column is greater or equal to
        ``low_column_name`` or the row contains nan, otherwise return False.

        Input:
        - Table with datetimes and np.nans.
        Output:
        - False should be returned for the strictly invalid rows and True for the rest.
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b')

        # Run
        table_data = pd.DataFrame({
            'a': [datetime(2020, 5, 17), datetime(2021, 9, 1), np.nan],
            'b': [datetime(2020, 5, 18), datetime(2020, 9, 2), datetime(2020, 9, 2)],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, False, True]
        np.testing.assert_array_equal(expected_out, out)

    def test__transform(self):
        """Test the ``Inequality._transform`` method.

        The method is expected to compute the distance between the high and low columns
        and create a diff column with the logarithm of the distance + 1.

        Setup:
        - ``_diff_column_name`` is set to ``'a#b'``.
        Input:
        - Table with two columns at a constant distance of 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the log of distances + 1, which is np.log(4).
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b')
        instance._diff_column_name = 'a#b'

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_datetime(self):
        """Test the ``Inequality._transform`` method.

        The method is expected to compute the distance between the high and low columns
        and create a diff column with the logarithm of the distance + 1.

        Setup:
        - ``_diff_column_name`` is set to ``'a#b'``.
        Input:
        - Table with two datetime columns at a distance of 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the log of distances + 1, which is np.log(4).
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b')
        instance._diff_column_name = 'a#b'
        instance._is_datetime = True

        # Run
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform(self):
        """Test the ``Inequality.reverse_transform`` method.

        The method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to integers
            - add back the dropped column

        Setup:
        - ``_diff_column_name = 'a#b'``
        - ``_dtype`` as integer
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3 with diff column dropped.
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b')
        instance._dtype = pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS
        instance._diff_column_name = 'a#b'

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_floats(self):
        """Test the ``Inequality.reverse_transform`` method.

        The method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to floats
            - add back the dropped column

        Setup:
        - ``_diff_column_name = 'a#b'``
        - ``_dtype`` as float
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3 with diff column dropped.
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b')
        instance._dtype = np.dtype('float')
        instance._diff_column_name = 'a#b'

        # Run
        transformed = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'c': [7, 8, 9],
            'b': [4.1, 5.2, 6.3],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime(self):
        """Test the ``Inequality.reverse_transform`` method.

        The method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to datetime
            - add back the dropped column

        Setup:
        - ``_diff_column_name = 'a#b'``
        - ``_dtype`` as datetime
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 1sec with diff column dropped.
        """
        # Setup
        instance = Inequality(low_column_name='a', high_column_name='b')
        instance._dtype = np.dtype('<M8[ns]')
        instance._diff_column_name = 'a#b'
        instance._is_datetime = True

        # Run
        transformed = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'c': [1, 2],
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01'])
        })
        pd.testing.assert_frame_equal(out, expected_out)


class TestScalarInequality():

    def test__validate_inputs_incorrect_column(self):
        """Test the ``_validate_inputs`` method.

        Ensure the method raises an error when the column name is not a string.

        Input:
        - a non-string
        - a number
        - an inequality
        Side effect:
        - Raise ``ValueError`` because the column name must be a string
        """
        # Run / Assert
        err_msg = '`column_name` must be a string.'
        with pytest.raises(ValueError, match=err_msg):
            ScalarInequality._validate_inputs(column_name=['a'], value=1, relation='>')

    def test__validate_inputs_incorrect_value_datetime(self):
        """Test the ``_validate_inputs`` method.

        Ensure the method raises an error when the ``value`` is not a ``Datetime`` represented
        as string.

        Input:
        - a string
        - a non string datetime
        - an inequality
        Side effect:
        - Raise ``ValueError`` because the datetime has to be represented as a string.
        """
        # Run / Assert
        value = pd.to_datetime('2021-02-01')
        err_msg = 'Datetime must be represented as a string.'
        with pytest.raises(ValueError, match=err_msg):
            ScalarInequality._validate_inputs(column_name='a', value=value, relation='>')

    def test__validate_inputs_incorrect_value(self):
        """Test the ``_validate_inputs`` method.

        Ensure the method raises an error when the value is not numerical.

        Input:
        - a string
        - a non-number
        - an inequality
        Side effect:
        - Raise ``ValueError`` because the value must be a numerical
        """
        # Run / Assert
        err_msg = '`value` must be a number or a string that represents a datetime.'
        with pytest.raises(ValueError, match=err_msg):
            ScalarInequality._validate_inputs(column_name='a', value='b', relation='>')

    def test__validate_inputs_incorrect_relation(self):
        """Test the ``_validate_inputs`` method.

        Raise an error when the relation is not valid.

        Input:
        - a string
        - a number
        - a non-inequality
        Side effect:
        - Raise ``ValueError`` because the relation must be an inequality
        """
        # Run / Assert
        err_msg = '`relation` must be one of the following: `>`, `>=`, `<`, `<=`'
        with pytest.raises(ValueError, match=err_msg):
            ScalarInequality._validate_inputs(column_name='a', value=1, relation='=')

    @patch('sdv.constraints.ScalarInequality._validate_inputs')
    def test___init___(self, mock_validate):
        """Test the ``ScalarInequality.__init__`` method.

        The passed arguments should be stored as attributes.

        Input:
        - column_name should be a column name
        - value should be a number
        - relation should be an inequality symbol
        Side effects:
        - _column_name is set to column_name
        - _value is set to value
        - _diff_column_name is set to 'column_name#diff'
        - _operator is set to the numpy operation corresponding to the input relation
        - _dtype and _is_datetime are None
        - _validate_inputs is called once
        """
        # Run
        instance = ScalarInequality(column_name='a', value=1, relation='>')

        # Asserts
        assert instance._column_name == 'a'
        assert instance._value == 1
        assert instance._diff_column_name == 'a#diff'
        assert instance._operator == np.greater
        assert instance._dtype is None
        assert instance._is_datetime is None
        assert instance._datetime_format is None
        mock_validate.assert_called_once_with('a', 1, '>')

    def test__get_is_datetime_incorrect_data(self):
        """Test the ``ScalarInequality._get_is_datetime`` method.

        Ensure that if one of column/value is datetime, both of them are.

        Input:
        - Table data.
        Side Effect:
        - Raises ``ValueError`` if only one of column/value is datetime.
        """
        # Setupy
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01']),
        })
        instance = ScalarInequality(column_name='a', value=1, relation='<')

        # Run / Assert
        err_msg = 'Both column and value must be datetime.'
        with pytest.raises(ValueError, match=err_msg):
            instance._get_is_datetime(table_data)

    def test__validate_columns_exist_incorrect_columns(self):
        """Test the ``ScalarInequality._validate_columns_exist`` method.

        This method raises an error if ``column_name`` does not exist.

        Input:
        - Table with given data.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6]
        })
        instance = ScalarInequality(column_name='c', value=5, relation='>')

        # Run / Assert
        with pytest.raises(KeyError):
            instance._validate_columns_exist(table_data)

    def test__fit(self):
        """Test the ``ScalarInequality._fit`` method.

        The method should learn the ``dtype`` of ``column_name`` and ``_is_datetime``.

        Input:
        - Table data with integers.
        Side Effect:
        - _validate_columns_exist should be called once
        - _get_is_datetime should be called once
        - _is_datetime should receive the output of _get_is_datetime
        - _dtype should be a list of integer dtypes.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6]
        })
        instance = ScalarInequality(column_name='b', value=3, relation='>')
        instance._validate_columns_exist = Mock()
        instance._get_is_datetime = Mock(return_value=False)

        # Run
        instance._fit(table_data)

        # Assert
        instance._validate_columns_exist.assert_called_once_with(table_data)
        instance._get_is_datetime.assert_called_once_with(table_data)
        assert not instance._is_datetime
        assert instance._dtype == pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS

    def test__fit_floats(self):
        """Test the ``ScalarInequality._fit`` method.

        The attribute ``_dtype`` should be float when ``column_name`` contains floats.

        Input:
        - Table data with floats.
        Side Effect:
        - _dtype should be a float dtype.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4., 5., 6.]
        })
        instance = ScalarInequality(column_name='b', value=10, relation='>')

        # Run
        instance._fit(table_data)

        # Assert
        assert instance._dtype == np.dtype('float')

    def test__fit_datetime(self):
        """Test the ``ScalarInequality._fit`` method.

        The attribute ``_dtype`` should be datetime when ``column_name`` contains datetimes.

        Input:
        - Table data with datetimes.
        Side Effect:
        - _dtype should be a list of datetime dtypes.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01']),
            'b': pd.to_datetime(['2020-01-02'])
        })
        instance = ScalarInequality(column_name='b', value='2020-01-01', relation='>')

        # Run
        instance._fit(table_data)

        # Assert
        assert instance._dtype == np.dtype('<M8[ns]')
        assert instance._datetime_format == '%Y-%m-%d'

    def test_is_valid(self):
        """Test the ``ScalarInequality.is_valid`` method with ``relation = '>'``.

        The method should return True when ``column_name`` is greater than
        ``value`` or the row contains nan, otherwise return False.

        Input:
        - Table with a mixture of valid and invalid rows, as well as np.nans.
        Output:
        - False should be returned for the strictly invalid rows and True for the rest.
        """
        # Setup
        instance = ScalarInequality(column_name='b', value=2, relation='>')

        # Run
        table_data = pd.DataFrame({
            'a': [1, np.nan, 3, 4, None],
            'b': [4, 2, np.nan, -6, None],
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, False, True, False, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_datetimes(self):
        """Test the ``ScalarInequality.is_valid`` method with datetimes and ``relation = '<='``.

        The method should return True when ``column_name`` is greater or equal to
        ``value`` or the row contains nan, otherwise return False.

        Input:
        - Table with datetimes and np.nans.
        Output:
        - False should be returned for the strictly invalid rows and True for the rest.
        """
        # Setup
        instance = ScalarInequality(
            column_name='b',
            value='8/31/2021',
            relation='>=')

        # Run
        table_data = pd.DataFrame({
            'b': [datetime(2021, 8, 30), datetime(2021, 8, 31), datetime(2021, 9, 2), np.nan],
            'c': [7, 8, 9, 10]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [False, True, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test__transform(self):
        """Test the ``ScalarInequality._transform`` method.

        The method is expected to compute the distance between the ``column_name`` and ``value``
        and create a diff column with the logarithm of the distance + 1.

        Setup:
        - ``_diff_column_name`` is set to ``'a#'``.
        Input:
        - Table data.
        Output:
        - Same table with a diff column of the log of the distances + 1
          in the ``column_name``'s place.
        """
        # Setup
        instance = ScalarInequality(column_name='a', value=1, relation='>=')
        instance._diff_column_name = 'a#'

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'c': [7, 8, 9],
            'a#': [np.log(1), np.log(2), np.log(3)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_datetime(self):
        """Test the ``ScalarInequality._transform`` method.

        The method is expected to compute the distance between the ``column_name`` and ``value``
        and create a diff column with the logarithm of the distance + 1.

        Setup:
        - ``_diff_column_name`` is set to ``'a#'``.
        Input:
        - Table data with datetimes.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1
          in the ``column_name``'s place.
        """
        # Setup
        instance = ScalarInequality(
            column_name='a',
            value='2020-01-01T00:00:00',
            relation='>')
        instance._diff_column_name = 'a#'
        instance._is_datetime = True

        # Run
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-01T00:00:01']),
            'c': [1, 2],
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'c': [1, 2],
            'a#': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform(self):
        """Test the ``ScalarInequality.reverse_transform`` method.

        The method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the ``column_name``
            - convert the output to integers
            - add back the dropped column

        Setup:
        - ``_diff_column_name = 'a#'``
        - ``_dtype`` as integer
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as int
        and the diff column dropped.
        """
        # Setup
        instance = ScalarInequality(column_name='a', value=1, relation='>=')
        instance._dtype = pd.Series([1]).dtype  # exact dtype (32 or 64) depends on OS
        instance._diff_column_name = 'a#'

        # Run
        transformed = pd.DataFrame({
            'a#': [np.log(1), np.log(2), np.log(3)],
            'c': [7, 8, 9],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'c': [7, 8, 9],
            'a': [1, 2, 3],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_floats(self):
        """Test the ``ScalarInequality.reverse_transform`` method.

        The method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the ``value``
            - convert the output to float
            - add back the dropped column

        Setup:
        - ``_diff_column_name = 'a#'``
        - ``_dtype`` as float
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 1, as float
        and the diff column dropped.
        """
        # Setup
        instance = ScalarInequality(column_name='a', value=1, relation='>=')
        instance._dtype = np.dtype('float')
        instance._diff_column_name = 'a#'

        # Run
        transformed = pd.DataFrame({
            'a#': [np.log(1.1), np.log(2.1), np.log(3.3)],
            'c': [7, 8, 9],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'c': [7, 8, 9],
            'a': [1.1, 2.1, 3.3],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime(self):
        """Test the ``ScalarInequality.reverse_transform`` method.

        The method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the ``column_name``
            - convert the output to datetime
            - add back the dropped column

        Setup:
        - ``_diff_column_name = 'a#'``
        - ``_dtype`` as datetime
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as datetime
        and the diff column dropped.
        """
        # Setup
        instance = ScalarInequality(
            column_name='a',
            value='2020-01-01T00:00:00',
            relation='>=')
        instance._dtype = np.dtype('<M8[ns]')
        instance._diff_column_name = 'a#'
        instance._is_datetime = True
        instance._datetime_format = '%Y-%m-%dT%H:%M:%S'

        # Run
        transformed = pd.DataFrame({
            'a#': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'c': [1, 2],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'c': [1, 2],
            'a': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-01T00:00:01']),
        })
        pd.testing.assert_frame_equal(out, expected_out)


class TestPositive():

    def test__init__(self):
        """Test the ``Positive.__init__`` method.

        Ensure the attributes are correctly set.
        """
        # Run
        instance = Positive(column_name='abc')

        # Asserts
        assert instance._value == 0
        assert instance._column_name == 'abc'
        assert instance._operator == np.greater_equal

    def test__init__strict_True(self):
        """Test the ``Positive.__init__`` method.

        Ensure the attributes are correctly set when ``strict`` is True.
        """
        # Run
        instance = Positive(column_name='abc', strict=True)

        # Asserts
        assert instance._value == 0
        assert instance._column_name == 'abc'
        assert instance._operator == np.greater


class TestNegative():

    def test__init__(self):
        """Test the ``Negative.__init__`` method.

        Ensure the attributes are correctly set.
        """
        # Run
        instance = Negative(column_name='abc')

        # Asserts
        assert instance._value == 0
        assert instance._column_name == 'abc'
        assert instance._operator == np.less_equal

    def test__init__strict_True(self):
        """Test the ``Negative.__init__`` method.

        Ensure the attributes are correctly set when ``strict`` is True.
        """
        # Run
        instance = Negative(column_name='abc', strict=True)

        # Asserts
        assert instance._value == 0
        assert instance._column_name == 'abc'
        assert instance._operator == np.less


class TestRange():

    def test___init__(self):
        """Test the ``Range.__init__`` method.

        The instance should contain the name of the three passed columns and also
        use an operator ``<`` if ``strict_boundaries`` is ``True`` by default.
        Input:
            - Three column names.
        """
        # Run
        instance = Range('age_when_joined', 'current_age', 'retirement_age')

        # Assert
        assert instance.low_column_name == 'age_when_joined'
        assert instance.middle_column_name == 'current_age'
        assert instance.high_column_name == 'retirement_age'
        assert instance._operator == operator.lt

    def test___init__strict_boundaries_false(self):
        """Test the ``Range.__init__`` method.

        Test the ``__init__`` method when ``strict_boundaries`` is ``False``.

        Input:
            - Three column names.
            - ``strict_boundaries=False``.

        Side Effect:
            - ``instance._operator`` should be ``operator.le``
        """
        # Run
        instance = Range(
            'age_when_joined',
            'current_age',
            'retirement_age',
            strict_boundaries=False
        )

        # Assert
        assert instance.low_column_name == 'age_when_joined'
        assert instance.middle_column_name == 'current_age'
        assert instance.high_column_name == 'retirement_age'
        assert instance._operator == operator.le

    def test__get_diff_column_name(self):
        """Test the ``Range._get_diff_column_name`` method.

        This method should return the name for the new ``transform_column``.

        Setup:
            - Create a pd.DataFrame with three columns.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``age_when_joined``, ``current_age``, ``retirement_age`` columns.

        Output:
            - The column names concatenated with ``#`` where the order is
              the ``middle_column_name`` followed by ``lower_column_name`` and ends with the
              ``higher_column_name``.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'current_age': [21, 22, 25],
            'retirement_age': [65, 68, 75]
        })
        instance = Range('age_when_joined', 'current_age', 'retirement_age')

        # Run
        transformed_column_name = instance._get_diff_column_name(table_data)

        # Assert
        assert transformed_column_name == 'current_age#age_when_joined#retirement_age'

    def test__get_is_datetime(self):
        """Test that the method detects whether or not the ``data`` is ``datetime``.

        This method should detect whether or not the data is ``datetime`` and if in case some is
        but other is not, raises an error.

        Setup:
            - Create a pd.DataFrame with three columns that are datetime data.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``join_date``, ``promotion_date``, ``retirement_date`` columns.

        Output:
            - The output should be True.
        """
        # Setup
        table_data = pd.DataFrame({
            'join_date': pd.to_datetime(['2021-02-10', '2021-05-10', '2021-08-11']),
            'promotion_date': pd.to_datetime(['2022-05-10', '2022-06-10', '2022-11-17']),
            'retirement_date': pd.to_datetime(['2050-10-11', '2058-10-04', '2075-11-14'])
        })
        instance = Range('join_date', 'promotion_date', 'retirement_date')

        # Run
        is_datetime = instance._get_is_datetime(table_data)

        # Assert
        assert is_datetime

    def test__get_is_datetime_no_datetimes(self):
        """Test that the method detects whether or not the ``data`` is ``datetime``.

        This method should detect whether or not the data is ``datetime`` and if in case some is
        but other is not, raises an error.

        Setup:
            - Create a pd.DataFrame with three columns that do not contain datetime data.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``age_when_joined``, ``current_age``, ``retirement_age`` columns.

        Output:
            - The output should be false since all the data is ``int``.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'current_age': [21, 22, 25],
            'retirement_age': [65, 68, 75]
        })
        instance = Range('age_when_joined', 'current_age', 'retirement_age')

        # Run
        is_datetime = instance._get_is_datetime(table_data)

        # Assert
        assert not is_datetime

    def test__get_is_datetime_raises_an_error(self):
        """Test that the method detects whether or not the ``data`` is ``datetime``.

        This method should detect whether or not the data is ``datetime`` and if in case some is
        but other is not, raises an error.

        Setup:
            - Create a pd.DataFrame with three columns that contain both datetime and non
              datetime data.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``join_date``, ``promotion_date``, ``current_age`` columns.

        Side Effect:
            - Value error with the expected message should be raised.
        """
        # Setup
        table_data = pd.DataFrame({
            'join_date': pd.to_datetime(['2021-02-10', '2021-05-10', '2021-08-11']),
            'promotion_date': pd.to_datetime(['2022-05-10', '2022-06-10', '2022-11-17']),
            'current_age': [21, 22, 25],
        })
        instance = Range('join_date', 'promotion_date', 'current_age')
        expected_text = 'The constraint column and bounds must all be datetime.'

        # Run
        with pytest.raises(ValueError, match=expected_text):
            instance._get_is_datetime(table_data)

    def test__fit(self):
        """Test the ``_fit`` method of ``Range``.

        Test that the ``_fit`` method stores the proper transformation column name and
        learns whether the data is ``datetime``or not and it's ``dtype``.

        Setup:
            - Create a pd.DataFrame with three columns.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``join_date``, ``promotion_date``, ``current_age`` columns.

        Side Effect:
            - instance has ``_transformed_column`` and ``_is_datetime`` and ``_dtype``.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'current_age': [21, 22, 25],
            'retirement_age': [65, 68, 75],
            'current_age#age_when_joined#retirement_age': [1, 2, 3]
        }, dtype=np.int64)
        instance = Range('age_when_joined', 'current_age', 'retirement_age')

        # Run
        instance._fit(table_data)

        # Assert
        assert instance._transformed_column == 'current_age##age_when_joined##retirement_age'
        assert instance._dtype == np.int64
        assert not instance._is_datetime

    def test_is_valid_lt(self):
        """Test the ``Range.is_valid``.

        This test ensures that the ``is_valid`` method works with the operator ``<``
        (``<``) and validates that the ``middle_column_name`` is between ``low_column_name`` and
        ``high_column_name``.

        Setup:
            - Create a pd.DataFrame with three columns.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``age_when_joined``, ``current_age``, ``retirement_age`` columns.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'current_age': [21, 22, 25],
            'retirement_age': [65, 68, 75]
        })
        instance = Range('age_when_joined', 'current_age', 'retirement_age')

        # Run
        result = instance.is_valid(table_data)

        # Assert
        assert all(result)

    def test_is_valid_le(self):
        """Test the ``Range.is_valid``.

        This test ensures that the ``is_valid`` method works with the operator
        ``<=`` and validates that the ``middle_column_name`` is
        between ``low_column_name`` and ``high_column_name``.

        Setup:
            - Create a pd.DataFrame with three columns.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``age_when_joined``, ``current_age``, ``retirement_age`` columns.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [21, 19, 20],
            'current_age': [21, 22, 25],
            'retirement_age': [65, 68, 75]
        })
        instance = Range(
            'age_when_joined',
            'current_age',
            'retirement_age',
            strict_boundaries=False
        )

        # Run
        result = instance.is_valid(table_data)

        # Assert
        assert all(result)

    def test_is_valid_invalid(self):
        """Test the ``Range.is_valid``.

        This test ensures that the ``is_valid`` fails when the data is not in the range.

        Setup:
            - Create a pd.DataFrame with three columns.
            - Instance of ``Range`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``age_when_joined``, ``current_age``, ``retirement_age`` columns.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [70, 19, 20],
            'current_age': [21, 22, 25],
            'retirement_age': [65, 68, 75]
        })
        instance = Range(
            'age_when_joined',
            'current_age',
            'retirement_age',
            strict_boundaries=False
        )

        # Run
        result = instance.is_valid(table_data)

        # Assert
        assert not all(result)

    @patch('sdv.constraints.tabular.logit')
    def test__transform(self, mock_logit):
        """Test the ``Range._transform`` method.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Mock:
            - Mock the ``logit`` function.
        Input:
            - pd.DataFrame with ``age_when_joined``, ``current_age``, ``retirement_age`` columns.

        Output:
            - pd.DataFrame with an extra column containing the transformed ``column``.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'current_age': [21, 22, 25],
            'retirement_age': [65, 68, 75]
        })
        mock_logit.return_value = [1, 2, 3]
        instance = Range('age_when_joined', 'current_age', 'retirement_age')

        # Run
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'retirement_age': [65, 68, 75],
            'current_age#age_when_joined#retirement_age': [1, 2, 3]
        })
        mock_logit.assert_called_once()
        pd.testing.assert_frame_equal(expected_out, out)

    @patch('sdv.constraints.tabular.sigmoid')
    def test_reverse_transform(self, mock_sigmoid):
        """Test the ``reverse_transform`` method for ``Range``.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Mock:
            - Mock the sigomid function.
        Setup:
            - Original table data.
            - An expected transformed data with the expected column name.
            - Instance of Range constraint.

        Output:
            - A pd.DataFrame containing the original data.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'retirement_age': [65, 68, 75],
            'current_age': [21, 22, 25],
        })
        mock_sigmoid.return_value = pd.Series([21, 22, 25])
        transformed_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'retirement_age': [65, 68, 75],
            'current_age#age_when_joined#retirement_age': [1, 2, 3]
        })
        instance = Range('age_when_joined', 'current_age', 'retirement_age')

        # Run
        instance.fit(table_data)
        out = instance.reverse_transform(transformed_data)

        # Assert
        mock_sigmoid.assert_called_once()
        pd.testing.assert_frame_equal(table_data, out)

    @patch('sdv.constraints.tabular.pd')
    @patch('sdv.constraints.tabular.sigmoid')
    def test_reverse_transform_is_datetime(self, mock_sigmoid, mock_pd):
        """Test the ``reverse_transform`` method for ``Range``.

        When ``instance._is_datetime`` is ``True``, the data should be converted
        to ``pandas.to_datetime``.

        Mock:
            - Mock the sigmoid function.
            - Mock pandas.

        Setup:
            - Original table data.
            - An expected transformed data.
            - Instance of ScalarRange constraint.

        Output:
            - A pd.DataFrame containing the original data.

        Side Effects:
            - ``mock_pd`` has to be called once.
        """
        # Setup
        table_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'retirement_age': [65, 68, 75],
            'current_age': [21, 22, 25],
        })
        mock_sigmoid.return_value = pd.Series([21, 22, 25])
        transformed_data = pd.DataFrame({
            'age_when_joined': [18, 19, 20],
            'retirement_age': [65, 68, 75],
            'current_age#age_when_joined#retirement_age': [1, 2, 3]
        })
        mock_pd.to_datetime.side_effect = lambda x: x

        instance = Range('age_when_joined', 'current_age', 'retirement_age')
        instance._transformed_column = 'current_age#age_when_joined#retirement_age'
        instance._is_datetime = True

        # Run
        out = instance.reverse_transform(transformed_data)

        # Assert
        pd.testing.assert_frame_equal(table_data, out)
        mock_sigmoid.assert_called_once()
        mock_pd.to_datetime.assert_called_once()


class TestScalarRange():

    def test__validate_inputs(self):
        """Test the ``_validate_inputs`` method.

        The method should raise an error if the inputs are not valid.

        Setup:
            - low_value as a string not representing a datetime.
            - high_value as a int.

        Side Effects:
            A value error should be raised.
        """
        # Setup
        low_value = 'abc'
        high_value = 2

        # Run / Assert
        error_msg = (
            '``low_value`` and ``high_value`` must be a number or a string that '
            'represents a datetime.'
        )
        with pytest.raises(ValueError, match=error_msg):
            ScalarRange._validate_inputs(low_value, high_value)

    def test__validate_inputs_datetimes_not_strings(self):
        """Test the ``_validate_inputs`` method.

        The method should raise an error if the datetimes are not represented as string.

        Setup:
            - low_value as ``pd.datetime``
            - high_value as ``pd.datetime``

        Side Effects:
            A value error should be raised.
        """
        # Setup
        low_value = pd.to_datetime('2021-02-02')
        high_value = pd.to_datetime('2021-02-02')

        # Run / Assert
        with pytest.raises(ValueError, match='Datetime must be represented as a string.'):
            ScalarRange._validate_inputs(low_value, high_value)

    def test___init__(self):
        """Test the ``ScalarRange.__init__`` method.

        The instance should contain the name of the three passed columns and also
        use an operator ``<`` if ``strict_boundaries`` is ``True`` by default.
        Input:
            - Column name.
            - Lower value.
            - High value.
        """
        # Run
        instance = ScalarRange(column_name='age_when_joined', low_value=18, high_value=28)

        # Assert
        assert instance._column_name == 'age_when_joined'
        assert instance._low_value == 18
        assert instance._high_value == 28
        assert instance._operator == operator.lt

    def test___init__strict_boundaries_false(self):
        """Test the ``ScalarRange.__init__`` method.

        Test the ``__init__`` method when ``strict_boundaries`` is ``False``.

        Input:
            - Three column names.
            - ``strict_boundaries=False``.

        Side Effect:
            - ``instance._operator`` should be ``operator.le``
        """
        # Run
        instance = ScalarRange('age_when_joined', 18, 28, strict_boundaries=False)

        # Assert
        assert instance._column_name == 'age_when_joined'
        assert instance._low_value == 18
        assert instance._high_value == 28
        assert instance._operator == operator.le

    def test__get_is_datetime(self):
        """Test that the method detects whether or not the ``data`` is ``datetime``.

        This method should detect whether or not the data is ``datetime`` and if in case some is
        but other is not, raises an error.

        Setup:
            - Create a pd.DataFrame with a column that contains datetime data.
            - Instance of ``ScalarRange`` constraint with low and high as datetime (timestamps).

        Input:
            - pd.DataFrame with ``promotion_date`` column.

        Output:
            - The output should be True.
        """
        # Setup
        table_data = pd.DataFrame({
            'promotion_date': pd.to_datetime(['2022-05-10', '2022-06-10', '2022-11-17']),
        })
        instance = ScalarRange('promotion_date', '2021-02-10', '2050-10-11')

        # Run
        is_datetime = instance._get_is_datetime(table_data)

        # Assert
        assert is_datetime

    def test__get_is_datetime_no_datetimes(self):
        """Test that the method detects whether or not the ``data`` is ``datetime``.

        This method should detect whether or not the data is ``datetime`` and if in case some is
        but other is not, raises an error.

        Setup:
            - Create a pd.DataFrame with a column that is not datatime.
            - Instance of ``ScalarRange`` constraint with low and high as integers.

        Input:
            - pd.DataFrame with ``current_age``

        Output:
            - The output should be false since all the data is ``int``.
        """
        # Setup
        table_data = pd.DataFrame({'current_age': [21, 22, 25]})
        instance = ScalarRange('current_age', 21, 30)

        # Run
        is_datetime = instance._get_is_datetime(table_data)

        # Assert
        assert not is_datetime

    def test__get_is_datetime_raises_an_error(self):
        """Test that the method detects whether or not the ``data`` is ``datetime``.

        This method should detect whether or not the data is ``datetime`` and if in case some is
        but other is not, raises an error.

        Setup:
            - Create a pd.DataFrame with a column that contains datetime data.
            - Instance of ``ScalarRange`` constraint with low and high as integers.

        Input:
            - pd.DataFrame with  ``promotion_date``.

        Output:
            - The output should be false since all the data is ``int``.
        """
        # Setup
        table_data = pd.DataFrame({
            'promotion_date': pd.to_datetime(['2022-05-10', '2022-06-10', '2022-11-17']),
        })
        instance = ScalarRange('promotion_date', 18, 25)
        expected_text = 'The constraint column and bounds must all be datetime.'

        # Run
        with pytest.raises(ValueError, match=expected_text):
            instance._get_is_datetime(table_data)

    def test__get_diff_column_name(self):
        """Test the ``ScalarRange._get_diff_column_name`` method.

        This method should return the name for the new ``transform_column``.

        Setup:
            - Create a pd.DataFrame.
            - Instance of ``ScalarRange`` constraint.

        Input:
            - pd.DataFrame with ``current_age`` column.

        Output:
            - The column name concatenated with ``#`` followed by the ``low`` and ``high`` values.
        """
        # Setup
        table_data = pd.DataFrame({
            'current_age': [21, 22, 25],
            'current_age#18#35': [1, 2, 3]
        })
        instance = ScalarRange('current_age', 18, 35)

        # Run
        transformed_column_name = instance._get_diff_column_name(table_data)

        # Assert
        assert transformed_column_name == 'current_age##18##35'

    def test__fit(self):
        """Test the ``_fit`` method of ``ScalarRange``.

        Test that the ``_fit`` method stores the proper transformation column name and
        learns whether the data is ``datetime``or not and it's ``dtype``.

        Setup:
            - Create a pd.DataFrame with a column.
            - Instance of ``ScalarRange`` constraint with those three column names.

        Input:
            - pd.DataFrame with ``current_age`` columns.

        Side Effect:
            - instance has ``_transformed_column`` and ``_is_datetime`` and ``_dtype``.
        """
        # Setup
        table_data = pd.DataFrame({'current_age': [21, 22, 25]})
        instance = ScalarRange('current_age', 18, 20)

        # Run
        instance._fit(table_data)

        # Assert
        assert not instance._is_datetime
        assert instance._transformed_column == 'current_age#18#20'

    def test__fit_datetime(self):
        """Test the ``_fit`` method of ``ScalarRange`` when fitting with ``datetime`` data.

        Setup:
            - table_data with datetimes.

        Side Effects:
            - The ``instance._low_value`` and ``ìnstance._high_value`` have been converted
              to ``pd.datetime``
            - The ``instance._datetime_format`` equals to the ``table_data``.
        """
        # Setup
        table_data = pd.DataFrame({
            'checkin': [
                pd.to_datetime('2022-05-06'),
                pd.to_datetime('2022-05-07'),
                pd.to_datetime('2022-05-08'),
                pd.to_datetime('2022-05-09'),
            ]
        })
        instance = ScalarRange('checkin', '2022-05-05', '2022-06-01')

        # Run
        instance._fit(table_data)

        # Assert
        assert instance._is_datetime
        assert instance._transformed_column == 'checkin#2022-05-05#2022-06-01'
        assert instance._datetime_format == '%Y-%m-%d'
        assert instance._low_value == pd.to_datetime('2022-05-05')
        assert instance._high_value == pd.to_datetime('2022-06-01')

    def test_is_valid_lt(self):
        """Test the ``ScalarRange.is_valid``.

        This test ensures that the ``is_valid`` method works with the operator ``<``
        (``<``) and validates that the ``column_name`` is between ``low_value`` and
        ``high_value``.

        Setup:
            - Create a pd.DataFrame.
            - Instance of ``ScalarRange`` constraint with low and high values that are within the
              expected range.

        Input:
            - pd.DataFrame with ``current_age`` column.
        """
        # Setup
        table_data = pd.DataFrame({'current_age': [21, 22, 25]})
        instance = ScalarRange('current_age', 20, 26)

        # Run
        result = instance.is_valid(table_data)

        # Assert
        assert all(result)

    def test_is_valid_le(self):
        """Test the ``ScalarRange.is_valid``.

        This test ensures that the ``is_valid`` method works with the operator
        ``<=`` and validates that the ``column_name`` is
        between ``low_value`` and ``high_value``.

        Setup:
            - Create a pd.DataFrame.
            - Instance of ``ScalarRange`` constraint with low and high values that are within the
              expected range and using ``strict_boundaries`` as ``False``.

        Input:
            - pd.DataFrame with ``current_age`` column.
        """
        # Setup
        table_data = pd.DataFrame({'current_age': [21, 22, 25]})
        instance = ScalarRange('current_age', 21, 25, strict_boundaries=False)

        # Run
        result = instance.is_valid(table_data)

        # Assert
        assert all(result)

    def test_is_valid_invalid(self):
        """Test the ``ScalarRange.is_valid``.

        This test ensures that the ``is_valid`` fails when the data is not in the range.

        Setup:
            - Create a pd.DataFrame with ``current_age`` column.
            - Instance of ``ScalarRange`` constraint with ``low_value`` higher than the
              low value in ``current_age`` and high value lower than the higher value in the
              ``current_age``.

        Input:
            - pd.DataFrame with ``current_age`` column.
        """
        # Setup
        table_data = pd.DataFrame({'current_age': [21, 22, 25]})
        instance = ScalarRange('current_age', 28, 18, strict_boundaries=False)

        # Run
        result = instance.is_valid(table_data)

        # Assert
        assert not all(result)

    @patch('sdv.constraints.tabular.logit')
    def test__transform(self, mock_logit):
        """Test the ``ScalarRange._transform`` method.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Mock:
            - mock the logit function.
        Input:
            - pd.DataFrame with ``current_age`` column.

        Output:
            - pd.DataFrame with the transformed ``current_age``.
        """
        # Setup
        table_data = pd.DataFrame({'current_age': [21, 22, 25]})
        instance = ScalarRange('current_age', 20, 28)
        mock_logit.return_value = [1, 2, 3]

        # Run
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({'current_age#20#28': [1, 2, 3]})
        pd.testing.assert_frame_equal(expected_out, out)

    @patch('sdv.constraints.tabular.sigmoid')
    def test_reverse_transform(self, mock_sigmoid):
        """Test the ``reverse_transform`` method for ``ScalarRange``.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Mock:
            - Mock the sigmoid function.

        Setup:
            - Original table data.
            - An expected transformed data.
            - Instance of ScalarRange constraint.

        Output:
            - A pd.DataFrame containing the original data.
        """
        # Setup
        table_data = pd.DataFrame({'current_age': [21, 22, 25]})
        transformed_data = pd.DataFrame({'current_age#20#28': [1, 2, 3]})
        mock_sigmoid.return_value = pd.Series([21, 22, 25])
        instance = ScalarRange('current_age', 20, 28)

        # Run
        instance.fit(table_data)
        out = instance.reverse_transform(transformed_data)

        # Assert
        pd.testing.assert_frame_equal(table_data, out)
        mock_sigmoid.assert_called_once()

    @patch('sdv.constraints.tabular.pd')
    @patch('sdv.constraints.tabular.sigmoid')
    def test_reverse_transform_is_datetime(self, mock_sigmoid, mock_pd):
        """Test the ``reverse_transform`` method for ``ScalarRange``.

        When ``instance._is_datetime`` is ``True``, the data should be converted
        to ``pandas.to_datetime``.

        Mock:
            - Mock the sigmoid function.
            - Mock pandas.

        Setup:
            - Original table data.
            - An expected transformed data.
            - Instance of ScalarRange constraint.

        Output:
            - A pd.DataFrame containing the original data.

        Side Effects:
            - ``mock_pd`` has to be called once.
        """
        # Setup
        transformed_data = pd.DataFrame({'current_age#20#28': [1, 2, 3]})
        mock_sigmoid.return_value = pd.Series([21, 22, 25])
        instance = ScalarRange('current_age', 20, 28)
        instance._transformed_column = 'current_age#20#28'
        instance._is_datetime = True
        mock_pd.to_datetime.side_effect = lambda x: pd.to_datetime('2021-02-02 10:10:59')
        instance._datetime_format = '%Y-%m-%d'

        # Run
        output = instance.reverse_transform(transformed_data)

        # Assert
        expected_output = pd.DataFrame({
            'current_age': [
                pd.to_datetime('2021-02-02 10:10:59'),
                pd.to_datetime('2021-02-02 10:10:59'),
                pd.to_datetime('2021-02-02 10:10:59'),
            ]
        })
        pd.testing.assert_frame_equal(expected_output, output)
        mock_sigmoid.assert_called_once()
        assert mock_pd.to_datetime.call_count == 2


class TestOneHotEncoding():

    def test_reverse_transform(self):
        """Test the ``OneHotEncoding.reverse_transform`` method.

        It is expected to, for each of the appropriate rows, set the column
        with the largest value to one and set all other columns to zero.

        Input:
        - Table data with any numbers (pandas.DataFrame)
        Output:
        - Table data where the appropriate rows are one hot (pandas.DataFrame)
        """
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b'])

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.8],
            'b': [0.8, 0.1, 0.9],
            'c': [1, 2, 3]
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [0.0, 1.0, 0.0],
            'b': [1.0, 0.0, 1.0],
            'c': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_is_valid(self):
        """Test the ``OneHotEncoding.is_valid`` method.

        ``True`` when for the rows where the data is one hot, ``False`` otherwise.

        Input:
        - Table data (pandas.DataFrame) containing one valid column, one column with a sum less
        than 1, one column with a sum greater than 1, one column with halves adding to one and one
        column with nans.
        Output:
        - Series of ``True`` and ``False`` values (pandas.Series)
        """
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b', 'c'])

        # Run
        table_data = pd.DataFrame({
            'a': [1.0, 1.0, 0.0, 0.5, 1.0],
            'b': [0.0, 1.0, 0.0, 0.5, 0.0],
            'c': [0.0, 2.0, 0.0, 0.0, np.nan],
            'd': [1, 2, 3, 4, 5]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False, False, False])
        pd.testing.assert_series_equal(expected_out, out)


class TestUnique():

    def test___init__(self):
        """Test the ``Unique.__init__`` method.

        The ``column_names`` should be set to those provided.

        Input:
        - column names to keep unique.
        Output:
        - Instance with ``column_names`` set.
        """
        # Run
        instance = Unique(column_names=['a', 'b'])

        # Assert
        assert instance.column_names == ['a', 'b']

    def test_is_valid(self):
        """Test the ``Unique.is_valid`` method.

        This method should return a pd.Series where the index
        of the first occurence of a unique combination of ``instance.column_names``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple of the same combinations of columns.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(column_names=['a', 'b', 'c'])

        # Run
        data = pd.DataFrame({
            'a': [1, 1, 2, 2, 3, 4],
            'b': [5, 5, 6, 6, 7, 8],
            'c': [9, 9, 10, 10, 12, 13]
        })
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, False, True, False, True, True])
        pd.testing.assert_series_equal(valid, expected)

    def test_is_valid_custom_index_same_values(self):
        """Test the ``Unique.is_valid`` method.

        This method should return a pd.Series where the index
        of the first occurence of a unique combination of ``instance.column_names``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple of the same combinations of columns.
        - DataFrame has a custom index column which is set to 0 for rows.
        Output:
        - Series with the index of the first occurences set to ``True``.
        Github Issue:
        - Problem is described in: https://github.com/sdv-dev/SDV/issues/616
        """
        # Setup
        instance = Unique(column_names=['a', 'b', 'c'])

        # Run
        data = pd.DataFrame({
            'a': [1, 1, 2, 2, 3],
            'b': [5, 5, 6, 6, 7],
            'c': [8, 8, 9, 9, 10]
        }, index=[0, 0, 0, 0, 0])
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, False, True, False, True], index=[0, 0, 0, 0, 0])
        pd.testing.assert_series_equal(valid, expected)

    def test_is_valid_custom_index_not_sorted(self):
        """Test the ``Unique.is_valid`` method.

        This method should return a pd.Series where the index
        of the first occurence of a unique combination of ``instance.column_names``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple of the same combinations of columns.
        - DataFrame has a custom index column which is set in an unsorted way.
        Output:
        - Series with the index of the first occurences set to ``True``.
        Github Issue:
        - Problem is described in: https://github.com/sdv-dev/SDV/issues/617
        """
        # Setup
        instance = Unique(column_names=['a', 'b', 'c'])

        # Run
        data = pd.DataFrame({
            'a': [1, 1, 2, 2, 3],
            'b': [5, 5, 6, 6, 7],
            'c': [8, 8, 9, 9, 10]
        }, index=[2, 1, 3, 5, 4])
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, False, True, False, True], index=[2, 1, 3, 5, 4])
        pd.testing.assert_series_equal(valid, expected)

    def test_is_valid_one_column_custom_index_not_sorted(self):
        """Test the ``Unique.is_valid`` method.

        This method should return a pd.Series where the index
        of the first occurence of a unique value of ``self.column_names``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple occurences of the same value of the
        one column in ``instance.columns``.
        - DataFrame has a custom index column which is set in an unsorted way.
        Output:
        - Series with the index of the first occurences set to ``True``.
        Github Issue:
        - Problem is described in: https://github.com/sdv-dev/SDV/issues/617
        """
        # Setup
        instance = Unique(column_names='a')

        # Run
        data = pd.DataFrame({
            'a': [1, 1, 1, 2, 3, 2],
            'b': [1, 2, 3, 4, 5, 6],
            'c': [False, False, True, False, False, True]
        }, index=[2, 1, 3, 5, 4, 6])
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, False, False, True, True, False], index=[2, 1, 3, 5, 4, 6])
        pd.testing.assert_series_equal(valid, expected)

    def test_is_valid_one_column_custom_index_same_values(self):
        """Test the ``Unique.is_valid`` method.

        This method should return a pd.Series where the index
        of the first occurence of a unique value of ``self.column_names``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple occurences of the same value of the
        one column in ``instance.columns``.
        - DataFrame has a custom index column which is set to 0 for rows.
        Output:
        - Series with the index of the first occurences set to ``True``.
        Github Issue:
        - Problem is described in: https://github.com/sdv-dev/SDV/issues/616
        """
        # Setup
        instance = Unique(column_names='a')

        # Run
        data = pd.DataFrame({
            'a': [1, 1, 1, 2, 3, 2],
            'b': [1, 2, 3, 4, 5, 6],
            'c': [False, False, True, False, False, True]
        }, index=[0, 0, 0, 0, 0, 0])
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, False, False, True, True, False], index=[0, 0, 0, 0, 0, 0])
        pd.testing.assert_series_equal(valid, expected)

    def test_is_valid_one_column(self):
        """Test the ``Unique.is_valid`` method.

        This method should return a pd.Series where the index
        of the first occurence of a unique value of ``self.column_names``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple occurences of the same value of the
        one column in ``instance.columns``.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(column_names='a')

        # Run
        data = pd.DataFrame({
            'a': [1, 1, 1, 2, 3, 2],
            'b': [1, 2, 3, 4, 5, 6],
            'c': [False, False, True, False, False, True]
        })
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, False, False, True, True, False])
        pd.testing.assert_series_equal(valid, expected)

    def test_is_valid_one_column_nans(self):
        """Test the ``Unique.is_valid`` method for one column with nans.

        This method should return a pd.Series where the index
        of the first occurence of a unique value of ``instance.column_names``
        is set to ``True``, and every other occurence is set to ``False``.
        ``None``, ``np.nan`` and ``float('nan')`` should be treated as the same category.

        Input:
        - DataFrame with some repeated values, some of which are nan's.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(column_names=['a'])

        # Run
        data = pd.DataFrame({
            'a': [1, None, 2, np.nan, float('nan'), 1],
            'b': [np.nan, 1, None, float('nan'), float('nan'), 1],
        })
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, True, True, False, False, False])
        pd.testing.assert_series_equal(valid, expected)

    def test_is_valid_multiple_columns_nans(self):
        """Test the ``Unique.is_valid`` method for multiple columns with nans.

        This method should return a pd.Series where the index
        of the first occurence of a unique combination of ``instance.column_names``
        is set to ``True``, and every other occurence is set to ``False``.
        ``None``, ``np.nan`` and ``float('nan')`` should be treated as the same category.

        Input:
        - DataFrame with multiple of the same combinations of columns, some of which are nan's.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(column_names=['a', 'b'])

        # Run
        data = pd.DataFrame({
            'a': [1, None, 1, np.nan, float('nan'), 1],
            'b': [np.nan, 1, None, float('nan'), float('nan'), 1],
        })
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, True, False, True, False, True])
        pd.testing.assert_series_equal(valid, expected)


class TestFixedIncrements():

    def test___init__(self):
        """Test the ``FixedIncrements.__init__`` method.

        The ``column_name`` and ``increment_value`` instance variables should be set.

        Input:
            - column name as a string.
            - increment_value as an int.

        Expected behavior:
            - Instance with ``column_name`` and ``increment_value`` set.
        """
        # Run
        instance = FixedIncrements(column_name='column', increment_value=5)

        # Assert
        assert instance.column_name == 'column'
        assert instance.increment_value == 5
        assert instance.constraint_columns == ('column', )

    def test___init___increment_value_is_negative_number(self):
        """Test the ``FixedIncrements.__init__ method with a negative increment.

        If the ``increment_value`` is less than or equal to 0, then an error should be raised.

        Input:
            - column name as a string.
            - increment_value as -1

        Expected behavior:
            - ``ValueError`` should be raised.
        """
        # Run / Assert
        error_message = "The increment_value must be greater than 0."
        with pytest.raises(ValueError, match=error_message):
            FixedIncrements(column_name='column', increment_value=-1)

    def test___init___increment_value_is_decimal(self):
        """Test the ``FixedIncrements.__init__ method with a decimal as an increment.

        If the ``increment_value`` is not a whole number, then an error should be raised.

        Input:
            - column name as a string.
            - increment_value as 1.5

        Expected behavior:
            - ``ValueError`` should be raised.
        """
        # Run / Assert
        error_message = "The increment_value must be a whole number."
        with pytest.raises(ValueError, match=error_message):
            FixedIncrements(column_name='column', increment_value=1.5)

    def test__fit(self):
        """Test the ``FixedIncrements._fit`` method.

        The ``fit`` method should store the dtype of the DataFrame.

        Input:
            - A ``pandas.DataFrame`` with a float dtype.

        Expected behavior:
            - The ``instance._dtype`` should be set to float.
        """
        # Setup
        data = pd.DataFrame({'column': [7, 14, 21]}, dtype=float)
        instance = FixedIncrements(column_name='column', increment_value=7)

        # Run
        instance._fit(data)

        # Assert
        assert instance._dtype == float

    def test_is_valid(self):
        """Test the ``FixedIncrements.is_valid`` method.

        The ``is_valid`` method should return ``True`` for rows that are NaN or evenly divisible
        by the increment.

        Input:
            - A ``pandas.DataFrame`` with one column containing some NaNs, some numbers that are
            divisible by the increment and some numbers that are not.

        Output:
            - A ``pandas.Series`` where all the rows that are NaN or divisible by the increment are
            ``True``.
        """
        # Setup
        data = pd.DataFrame({'column': [7, 14, np.nan, 20, 8, 35]}, dtype=float)
        instance = FixedIncrements(column_name='column', increment_value=7)

        # Run
        is_valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, True, True, False, False, True], name='column')
        pd.testing.assert_series_equal(is_valid, expected)

    def test__transform(self):
        """Test the ``FixedIncrements._transform`` method.

        The ``_transform`` method should divide all values in the data by the ``increment_value``.

        Input:
            - A ``pd.DataFrame`` with one column containing NaNs and values divisible by the
            ``increment_value``.

        Output:
            - A ``pd.DataFrame`` with all the values in that column divided by the
            ``increment_value`` and the NaNs left alone.
        """
        # Setup
        data = pd.DataFrame({'column': [7, 14, np.nan, 35]})
        instance = FixedIncrements(column_name='column', increment_value=7)

        # Run
        transformed = instance._transform(data)

        # Assert
        expected = pd.DataFrame({'column': [1, 2, np.nan, 5]})
        pd.testing.assert_frame_equal(transformed, expected)

    def test_reverse_transform(self):
        """Test the ``FixedIncrements.reverse_transform`` method.

        The ``reverse_transform`` method should round all sampled values to the nearest int,
        and then multiply them by the ``increment_value`` and convert them to the ``_dtype``.

        Setup:
            - Set the ``_dtype`` to int64.

        Input:
            - A ``pandas.DataFrame`` with floats.

        Output:
            - A ``pandas.DataFrame`` with the values multiplied by the ``increment_value`` and
            converted to ints.
        """
        # Setup
        data = pd.DataFrame({'column': [1.3, 3.5, 4.2, 2.1]})
        instance = FixedIncrements(column_name='column', increment_value=7)
        instance._dtype = np.int64

        # Run
        reverse_transformed = instance.reverse_transform(data)

        # Assert
        expected = pd.DataFrame({'column': [7, 28, 28, 14]})
        pd.testing.assert_frame_equal(expected, reverse_transformed)

    def test_reverse_transform_nans(self):
        """Test the ``FixedIncrements.reverse_transform`` method with NaNs.

        The ``reverse_transform`` method should ignore the NaN values.

        Setup:
            - Set the ``_dtype`` to float64.

        Input:
            - A ``pandas.DataFrame`` with floats.

        Output:
            - A ``pandas.DataFrame`` with the values multiplied by the ``increment_value`` and
            the NaNs ignored.
        """
        # Setup
        data = pd.DataFrame({'column': [1.3, 3.5, np.nan, 4.2, np.nan, 2.1]})
        instance = FixedIncrements(column_name='column', increment_value=7)
        instance._dtype = np.float64

        # Run
        reverse_transformed = instance.reverse_transform(data)

        # Assert
        expected = pd.DataFrame({'column': [7, 28, np.nan, 28, np.nan, 14]})
        pd.testing.assert_frame_equal(expected, reverse_transformed)
