"""Tests for the sdv.constraints.tabular module."""

import uuid
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.tabular import (
    Between, ColumnFormula, CustomConstraint, FixedCombinations, GreaterThan, Negative,
    OneHotEncoding, Positive, Rounding, Unique)


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


class TestCustomConstraint():

    def test___init__(self):
        """Test the ``CustomConstraint.__init__`` method.

        The ``transform``, ``reverse_transform`` and ``is_valid`` methods
        should be replaced by the given ones, importing them if necessary.

        Setup:
        - Create dummy functions (created above this class).

        Input:
        - dummy transform and revert_transform + is_valid FQN
        Output:
        - Instance with all the methods replaced by the dummy versions.
        """
        is_valid_fqn = __name__ + '.dummy_is_valid_table'

        # Run
        instance = CustomConstraint(
            transform=dummy_transform_table,
            reverse_transform=dummy_reverse_transform_table,
            is_valid=is_valid_fqn
        )

        # Assert
        assert instance._transform == dummy_transform_table
        assert instance._reverse_transform == dummy_reverse_transform_table
        assert instance._is_valid == dummy_is_valid_table

    def test__run_transform_table(self):
        """Test the ``CustomConstraint._run`` method.

        The ``_run`` method excutes ``transform`` and ``reverse_transform``
        based on the signature of the functions. In this test, we evaluate
        the execution of "table" based functions.

        Setup:
        - Pass dummy transform function with ``table_data`` argument.
        Side Effects:
        - Run transform function once with ``table_data`` as input.
        Output:
        - applied identity transformation "table_data = transformed".
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_transform_mock = Mock(side_effect=dummy_transform_table,
                                    return_value=table_data)
        # Run
        instance = CustomConstraint(transform=dummy_transform_mock)
        transformed = instance.transform(table_data)

        # Asserts
        called = dummy_transform_mock.call_args
        dummy_transform_mock.assert_called_once()
        pd.testing.assert_frame_equal(called[0][0], table_data)
        pd.testing.assert_frame_equal(transformed, dummy_transform_mock.return_value)

    def test__run_reverse_transform_table(self):
        """Test the ``CustomConstraint._run`` method.

        The ``_run`` method excutes ``transform`` and ``reverse_transform``
        based on the signature of the functions. In this test, we evaluate
        the execution of "table" based functions.

        Setup:
        - Pass dummy reverse transform function with ``table_data`` argument.
        Side Effects:
        - Run reverse transform function once with ``table_data`` as input.
        Output:
        - applied identity transformation "table_data = reverse_transformed".
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_reverse_transform_mock = Mock(side_effect=dummy_reverse_transform_table,
                                            return_value=table_data)
        # Run
        instance = CustomConstraint(reverse_transform=dummy_reverse_transform_mock)
        reverse_transformed = instance.reverse_transform(table_data)

        # Asserts
        called = dummy_reverse_transform_mock.call_args
        dummy_reverse_transform_mock.assert_called_once()
        pd.testing.assert_frame_equal(called[0][0], table_data)
        pd.testing.assert_frame_equal(
            reverse_transformed, dummy_reverse_transform_mock.return_value)

    def test__run_is_valid_table(self):
        """Test the ``CustomConstraint._run_is_valid`` method.

        The ``_run_is_valid`` method excutes ``is_valid`` based on
        the signature of the functions. In this test, we evaluate
        the execution of "table" based functions.

        Setup:
        - Pass dummy is valid function with ``table_data`` argument.
        Side Effects:
        - Run is valid function once with ``table_data`` as input.
        Output:
        - Return a list of [True] of length ``table_data``.
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_is_valid_mock = Mock(side_effect=dummy_is_valid_table)

        # Run
        instance = CustomConstraint(is_valid=dummy_is_valid_mock)
        is_valid = instance.is_valid(table_data)

        # Asserts
        expected_out = [True] * len(table_data)
        called = dummy_is_valid_mock.call_args
        dummy_is_valid_mock.assert_called_once()
        pd.testing.assert_frame_equal(called[0][0], table_data)
        np.testing.assert_array_equal(is_valid, expected_out)

    def test__run_transform_table_column(self):
        """Test the ``CustomConstraint._run`` method.

        The ``_run`` method excutes ``transform`` and ``reverse_transform``
        based on the signature of the functions. In this test, we evaluate
        the execution of "table" and "column" based functions.

        Setup:
        - Pass dummy transform function with ``table_data`` and ``column`` arguments.
        Side Effects:
        - Run transform function once with ``table_data`` and ``column`` as input.
        Output:
        - applied identity transformation "table_data = transformed".
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_transform_mock = Mock(side_effect=dummy_transform_table_column,
                                    return_value=table_data)
        # Run
        instance = CustomConstraint(columns='a', transform=dummy_transform_mock)
        transformed = instance.transform(table_data)

        # Asserts
        called = dummy_transform_mock.call_args
        assert called[0][1] == 'a'
        dummy_transform_mock.assert_called_once()
        pd.testing.assert_frame_equal(called[0][0], table_data)
        pd.testing.assert_frame_equal(transformed, dummy_transform_mock.return_value)

    def test__run_transform_missing_column(self):
        """Test the ``CustomConstraint._run`` method.

        The ``_run`` method excutes ``transform`` and ``reverse_transform``
        based on the signature of the functions. In this test, we evaluate
        the execution of "table" that is missing the constraint column.

        Setup:
        - Pass dummy transform function with ``table_data`` and ``column`` arguments.
        Side Effects:
        - MissingConstraintColumnError is thrown.
        """
        # Setup
        table_data = pd.DataFrame({'b': [1, 2, 3]})
        dummy_transform_mock = Mock(side_effect=dummy_transform_table_column,
                                    return_value=table_data)
        # Run and assert
        instance = CustomConstraint(columns='a', transform=dummy_transform_mock)
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(table_data)

    def test__run_reverse_transform_table_column(self):
        """Test the ``CustomConstraint._run`` method.

        The ``_run`` method excutes ``transform`` and ``reverse_transform``
        based on the signature of the functions. In this test, we evaluate
        the execution of "table" and "column" based functions.

        Setup:
        - Pass dummy reverse transform function with ``table_data`` and ``column`` arguments.
        Side Effects:
        - Run reverse transform function once with ``table_data`` and ``column`` as input.
        Output:
        - applied identity transformation "table_data = transformed".
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_reverse_transform_mock = Mock(side_effect=dummy_reverse_transform_table_column,
                                            return_value=table_data)
        # Run
        instance = CustomConstraint(columns='a', reverse_transform=dummy_reverse_transform_mock)
        reverse_transformed = instance.reverse_transform(table_data)

        # Asserts
        called = dummy_reverse_transform_mock.call_args
        assert called[0][1] == 'a'
        dummy_reverse_transform_mock.assert_called_once()
        pd.testing.assert_frame_equal(called[0][0], table_data)
        pd.testing.assert_frame_equal(
            reverse_transformed, dummy_reverse_transform_mock.return_value)

    def test__run_is_valid_table_column(self):
        """Test the ``CustomConstraint._run_is_valid`` method.

        The ``_run_is_valid`` method excutes ``is_valid`` based on
        the signature of the functions. In this test, we evaluate
        the execution of "table" and "column" based functions.

        Setup:
        - Pass dummy is valid function with ``table_data`` and ``column`` argument.
        Side Effects:
        - Run is valid function once with ``table_data`` and ``column`` as input.
        Output:
        - Return a list of [True] of length ``table_data``.
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_is_valid_mock = Mock(side_effect=dummy_is_valid_table_column)

        # Run
        instance = CustomConstraint(columns='a', is_valid=dummy_is_valid_mock)
        is_valid = instance.is_valid(table_data)

        # Asserts
        expected_out = [True] * len(table_data)
        called = dummy_is_valid_mock.call_args
        assert called[0][1] == 'a'
        dummy_is_valid_mock.assert_called_once()
        pd.testing.assert_frame_equal(called[0][0], table_data)
        np.testing.assert_array_equal(is_valid, expected_out)

    def test__run_transform_column(self):
        """Test the ``CustomConstraint._run`` method.

        The ``_run`` method excutes ``transform`` and ``reverse_transform``
        based on the signature of the functions. In this test, we evaluate
        the execution of "column" based functions.

        Setup:
        - Pass dummy transform function with ``column_data`` argument.
        Side Effects:
        - Run transform function twice, once with the attempt of
        ``table_data`` and ``column`` and second with ``column_data`` as input.
        Output:
        - applied identity transformation "table_data = transformed".
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_transform_mock = Mock(side_effect=dummy_transform_column,
                                    return_value=table_data)
        # Run
        instance = CustomConstraint(columns='a', transform=dummy_transform_mock)
        transformed = instance.transform(table_data)

        # Asserts
        called = dummy_transform_mock.call_args_list
        assert len(called) == 2
        # call 1 (try)
        assert called[0][0][1] == 'a'
        pd.testing.assert_frame_equal(called[0][0][0], table_data)
        # call 2 (catch TypeError)
        pd.testing.assert_series_equal(called[1][0][0], table_data['a'])
        pd.testing.assert_frame_equal(transformed, dummy_transform_mock.return_value)

    def test__run_reverse_transform_column(self):
        """Test the ``CustomConstraint._run`` method.

        The ``_run`` method excutes ``transform`` and ``reverse_transform``
        based on the signature of the functions. In this test, we evaluate
        the execution of "column" based functions.

        Setup:
        - Pass dummy reverse transform function with ``column_data`` argument.
        Side Effects:
        - Run reverse transform function twice, once with the attempt of
        ``table_data`` and ``column`` and second with ``column_data`` as input.
        Output:
        - Applied identity transformation "table_data = transformed".
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_reverse_transform_mock = Mock(side_effect=dummy_reverse_transform_column,
                                            return_value=table_data)
        # Run
        instance = CustomConstraint(columns='a', reverse_transform=dummy_reverse_transform_mock)
        reverse_transformed = instance.reverse_transform(table_data)

        # Asserts
        called = dummy_reverse_transform_mock.call_args_list
        assert len(called) == 2
        # call 1 (try)
        assert called[0][0][1] == 'a'
        pd.testing.assert_frame_equal(called[0][0][0], table_data)
        # call 2 (catch TypeError)
        pd.testing.assert_series_equal(called[1][0][0], table_data['a'])
        pd.testing.assert_frame_equal(
            reverse_transformed, dummy_reverse_transform_mock.return_value)

    def test__run_is_valid_column(self):
        """Test the ``CustomConstraint._run_is_valid`` method.

        The ``_run_is_valid`` method excutes ``is_valid`` based on
        the signature of the functions. In this test, we evaluate
        the execution of "column" based functions.

        Setup:
        - Pass dummy is valid function with ``column_data`` argument.
        Side Effects:
        - Run is valid function twice, once with the attempt of
        ``table_data`` and ``column`` and second with ``column_data`` as input.
        Output:
        - Return a list of [True] of length ``table_data``.
        """
        # Setup
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        dummy_is_valid_mock = Mock(side_effect=dummy_is_valid_column)

        # Run
        instance = CustomConstraint(columns='a', is_valid=dummy_is_valid_mock)
        is_valid = instance.is_valid(table_data)

        # Asserts
        expected_out = [True] * len(table_data)
        called = dummy_is_valid_mock.call_args_list
        assert len(called) == 2
        # call 1 (try)
        assert called[0][0][1] == 'a'
        pd.testing.assert_frame_equal(called[0][0][0], table_data)
        # call 2 (catch TypeError)
        pd.testing.assert_series_equal(called[1][0][0], table_data['a'])
        np.testing.assert_array_equal(is_valid, expected_out)


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

    def test___init__sets_rebuild_columns_if_not_reject_sampling(self):
        """Test the ``FixedCombinations.__init__`` method.

        The rebuild columns should only be set if the ``handling_strategy``
        is not ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are set
        """
        # Setup
        columns = ['b', 'c']

        # Run
        instance = FixedCombinations(column_names=columns, handling_strategy='transform')

        # Assert
        assert instance.rebuild_columns == tuple(columns)

    def test___init__does_not_set_rebuild_columns_reject_sampling(self):
        """Test the ``FixedCombinations.__init__`` method.

        The rebuild columns should not be set if the ``handling_strategy``
        is ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are empty
        """
        # Setup
        columns = ['b', 'c']

        # Run
        instance = FixedCombinations(column_names=columns, handling_strategy='reject_sampling')

        # Assert
        assert instance.rebuild_columns == ()

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

    def test_fit(self):
        """Test the ``FixedCombinations.fit`` method.

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
            'c': ['g', 'h', 'i']
        })
        instance.fit(table_data)

        # Asserts
        expected_combinations = pd.DataFrame({
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        assert instance._separator == '#'
        assert instance._joint_column == 'b#c'
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

        If some of the columns needed for the transform are missing, and
        ``fit_columns_model`` is False, it will raise a ``MissingConstraintColumnError``.

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
        instance = FixedCombinations(column_names=columns, fit_columns_model=False)
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


class TestGreaterThan():

    def test__validate_scalar(self):
        """Test the ``_validate_scalar`` method.

        This method validates the inputs if and transforms them into
        the correct format.

        Input:
        - scalar_column = 0
        - column_names = 'b'
        Output:
        - column_names == ['b']
        """
        # Setup
        scalar_column = 0
        column_names = 'b'
        scalar = 'high'

        # Run
        out = GreaterThan._validate_scalar(scalar_column, column_names, scalar)

        # Assert
        out == ['b']

    def test__validate_scalar_list(self):
        """Test the ``_validate_scalar`` method.

        This method validates the inputs if and transforms them into
        the correct format.

        Input:
        - scalar_column = 0
        - column_names = ['b']
        Output:
        - column_names == ['b']
        """
        # Setup
        scalar_column = 0
        column_names = ['b']
        scalar = 'low'

        # Run
        out = GreaterThan._validate_scalar(scalar_column, column_names, scalar)

        # Assert
        out == ['b']

    def test__validate_scalar_error(self):
        """Test the ``_validate_scalar`` method.

        This method raises an error when the the scalar column is a list.

        Input:
        - scalar_column = 0
        - column_names = 'b'
        Side effect:
        - Raise error since the scalar is a list
        """
        # Setup
        scalar_column = [0]
        column_names = 'b'
        scalar = 'high'

        # Run / Assert
        with pytest.raises(TypeError):
            GreaterThan._validate_scalar(scalar_column, column_names, scalar)

    def test__validate_inputs_high_is_scalar(self):
        """Test the ``_validate_inputs`` method.

        This method checks ``scalar`` and formats the data based
        on what is expected to be a list or not. In addition, it
        returns the ``constraint_columns``.

        Input:
        - low = 'a'
        - high = 3
        - scalar = 'high'
        Output:
        - low == ['a']
        - high == 3
        - constraint_columns = ('a')
        """
        # Setup / Run
        low, high, constraint_columns = GreaterThan._validate_inputs(
            low='a', high=3, scalar='high', drop=None)

        # Assert
        low == ['a']
        high == 3
        constraint_columns == ('a',)

    def test__validate_inputs_low_is_scalar(self):
        """Test the ``_validate_inputs`` method.

        This method checks ``scalar`` and formats the data based
        on what is expected to be a list or not. In addition, it
        returns the ``constraint_columns``.

        Input:
        - low = 3
        - high = 'b'
        - scalar = 'low'
        - drop = None
        Output:
        - low == 3
        - high == ['b']
        - constraint_columns = ('b')
        """
        # Setup / Run
        low, high, constraint_columns = GreaterThan._validate_inputs(
            low=3, high='b', scalar='low', drop=None)

        # Assert
        low == 3
        high == ['b']
        constraint_columns == ('b',)

    def test__validate_inputs_scalar_none(self):
        """Test the ``_validate_inputs`` method.

        This method checks ``scalar`` and formats the data based
        on what is expected to be a list or not. In addition, it
        returns the ``constraint_columns``.

        Input:
        - low = 'a'
        - high = 3 # where 3 is a column name
        - scalar = None
        - drop = None
        Output:
        - low == ['a']
        - high == [3]
        - constraint_columns = ('a', 3)
        """
        # Setup / Run
        low, high, constraint_columns = GreaterThan._validate_inputs(
            low='a', high=3, scalar=None, drop=None)

        # Assert
        low == ['a']
        high == [3]
        constraint_columns == ('a', 3)

    def test__validate_inputs_scalar_none_lists(self):
        """Test the ``_validate_inputs`` method.

        This method checks ``scalar`` and formats the data based
        on what is expected to be a list or not. In addition, it
        returns the ``constraint_columns``.

        Input:
        - low = ['a']
        - high = ['b', 'c']
        - scalar = None
        - drop = None
        Output:
        - low == ['a']
        - high == ['b', 'c']
        - constraint_columns = ('a', 'b', 'c')
        """
        # Setup / Run
        low, high, constraint_columns = GreaterThan._validate_inputs(
            low=['a'], high=['b', 'c'], scalar=None, drop=None)

        # Assert
        low == ['a']
        high == ['b', 'c']
        constraint_columns == ('a', 'b', 'c')

    def test__validate_inputs_scalar_none_two_lists(self):
        """Test the ``_validate_inputs`` method.

        This method checks ``scalar`` and formats the data based
        on what is expected to be a list or not. In addition, it
        returns the ``constraint_columns``.

        Input:
        - low = ['a', 0]
        - high = ['b', 'c']
        - scalar = None
        - drop = None
        Side effect:
        - Raise error because both high and low are more than one column
        """
        # Run / Assert
        with pytest.raises(ValueError):
            GreaterThan._validate_inputs(low=['a', 0], high=['b', 'c'], scalar=None, drop=None)

    def test__validate_inputs_scalar_unknown(self):
        """Test the ``_validate_inputs`` method.

        This method checks ``scalar`` and formats the data based
        on what is expected to be a list or not. In addition, it
        returns the ``constraint_columns``.

        Input:
        - low = 'a'
        - high = 'b'
        - scalar = 'unknown'
        - drop = None
        Side effect:
        - Raise error because scalar is unknown
        """
        # Run / Assert
        with pytest.raises(ValueError):
            GreaterThan._validate_inputs(low='a', high='b', scalar='unknown', drop=None)

    def test__validate_inputs_drop_error_low(self):
        """Test the ``_validate_inputs`` method.

        Make sure the method raises an error if ``drop``==``scalar``
        when ``scalar`` is not ``None``.

        Input:
        - low = 2
        - high = 'b'
        - scalar = 'low'
        - drop = 'low'
        Side effect:
        - Raise error because scalar is unknown
        """
        # Run / Assert
        with pytest.raises(ValueError):
            GreaterThan._validate_inputs(low=2, high='b', scalar='low', drop='low')

    def test__validate_inputs_drop_error_high(self):
        """Test the ``_validate_inputs`` method.

        Make sure the method raises an error if ``drop``==``scalar``
        when ``scalar`` is not ``None``.

        Input:
        - low = 'a'
        - high = 3
        - scalar = 'high'
        - drop = 'high'
        Side effect:
        - Raise error because scalar is unknown
        """
        # Run / Assert
        with pytest.raises(ValueError):
            GreaterThan._validate_inputs(low='a', high=3, scalar='high', drop='high')

    def test__validate_inputs_drop_success(self):
        """Test the ``_validate_inputs`` method.

        Make sure the method raises an error if ``drop``==``scalar``
        when ``scalar`` is not ``None``.

        Input:
        - low = 'a'
        - high = 'b'
        - scalar = 'high'
        - drop = 'low'
        Output:
        - low = ['a']
        - high = 0
        - constraint_columns == ('a')
        """
        # Run / Assert
        low, high, constraint_columns = GreaterThan._validate_inputs(
            low='a', high=0, scalar='high', drop='low')

        assert low == ['a']
        assert high == 0
        assert constraint_columns == ('a',)

    def test___init___(self):
        """Test the ``GreaterThan.__init__`` method.

        The passed arguments should be stored as attributes.

        Input:
        - low = 'a'
        - high = 'b'
        Side effects:
        - instance._low == 'a'
        - instance._high == 'b'
        - instance._strict == False
        """
        # Run
        instance = GreaterThan(low='a', high='b')

        # Asserts
        assert instance._low == ['a']
        assert instance._high == ['b']
        assert instance._strict is False
        assert instance._scalar is None
        assert instance._drop is None
        assert instance.constraint_columns == ('a', 'b')

    def test___init__sets_rebuild_columns_if_not_reject_sampling(self):
        """Test the ``GreaterThan.__init__`` method.

        The rebuild columns should only be set if the ``handling_strategy``
        is not ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are set
        """
        # Run
        instance = GreaterThan(low='a', high='b', handling_strategy='transform')

        # Assert
        assert instance.rebuild_columns == ['b']

    def test___init__does_not_set_rebuild_columns_reject_sampling(self):
        """Test the ``GreaterThan.__init__`` method.

        The rebuild columns should not be set if the ``handling_strategy``
        is ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are empty
        """
        # Run
        instance = GreaterThan(low='a', high='b', handling_strategy='reject_sampling')

        # Assert
        assert instance.rebuild_columns == ()

    def test___init___high_is_scalar(self):
        """Test the ``GreaterThan.__init__`` method.

        The passed arguments should be stored as attributes. Make sure ``scalar``
        is set to ``'high'``.

        Input:
        - low = 'a'
        - high = 0
        - strict = True
        - drop = 'low'
        - scalar = 'high'
        Side effects:
        - instance._low == 'a'
        - instance._high == 0
        - instance._strict == True
        - instance._drop = 'low'
        - instance._scalar == 'high'
        """
        # Run
        instance = GreaterThan(low='a', high=0, strict=True, drop='low', scalar='high')

        # Asserts
        assert instance._low == ['a']
        assert instance._high == 0
        assert instance._strict is True
        assert instance._scalar == 'high'
        assert instance._drop == 'low'
        assert instance.constraint_columns == ('a',)

    def test___init___low_is_scalar(self):
        """Test the ``GreaterThan.__init__`` method.

        The passed arguments should be stored as attributes. Make sure ``scalar``
        is set to ``'high'``.

        Input:
        - low = 0
        - high = 'a'
        - strict = True
        - drop = 'high'
        - scalar = 'low'
        Side effects:
        - instance._low == 0
        - instance._high == 'a'
        - instance._stric == True
        - instance._drop = 'high'
        - instance._scalar == 'low'
        """
        # Run
        instance = GreaterThan(low=0, high='a', strict=True, drop='high', scalar='low')

        # Asserts
        assert instance._low == 0
        assert instance._high == ['a']
        assert instance._strict is True
        assert instance._scalar == 'low'
        assert instance._drop == 'high'
        assert instance.constraint_columns == ('a',)

    def test___init___strict_is_false(self):
        """Test the ``GreaterThan.__init__`` method.

        Ensure that ``operator`` is set to ``np.greater_equal``
        when ``strict`` is set to ``False``.

        Input:
        - low = 'a'
        - high = 'b'
        - strict = False
        """
        # Run
        instance = GreaterThan(low='a', high='b', strict=False)

        # Assert
        assert instance.operator == np.greater_equal

    def test___init___strict_is_true(self):
        """Test the ``GreaterThan.__init__`` method.

        Ensure that ``operator`` is set to ``np.greater``
        when ``strict`` is set to ``True``.

        Input:
        - low = 'a'
        - high = 'b'
        - strict = True
        """
        # Run
        instance = GreaterThan(low='a', high='b', strict=True)

        # Assert
        assert instance.operator == np.greater

    def test__init__get_columns_to_reconstruct_default(self):
        """Test the ``GreaterThan._get_columns_to_reconstruct`` method.

        This method returns:
            - ``_high`` if drop is "high"
            - ``_low`` if drop is "low"
            - ``_low`` if scalar is "high"
            - ``_high`` otherwise

        Setup:
        - low = 'a'
        - high = 'b'
        Side effects:
        - self._columns_to_reconstruct == ['b']
        """
        # Setup
        instance = GreaterThan(low='a', high='b')
        instance._columns_to_reconstruct == ['b']

    def test__init__get_columns_to_reconstruct_drop_high(self):
        """Test the ``GreaterThan._get_columns_to_reconstruct`` method.

        This method returns:
            - ``_high`` if drop is "high"
            - ``_low`` if drop is "low"
            - ``_low`` if scalar is "high"
            - ``_high`` otherwise

        Setup:
        - low = 'a'
        - high = 'b'
        - drop = 'high'
        Side effects:
        - self._columns_to_reconstruct == ['b']
        """
        # Setup
        instance = GreaterThan(low='a', high='b', drop='high')
        instance._columns_to_reconstruct == ['b']

    def test__init__get_columns_to_reconstruct_drop_low(self):
        """Test the ``GreaterThan._get_columns_to_reconstruct`` method.

        This method returns:
            - ``_high`` if drop is "high"
            - ``_low`` if drop is "low"
            - ``_low`` if scalar is "high"
            - ``_high`` otherwise

        Setup:
        - low = 'a'
        - high = 'b'
        - drop = 'low'
        Side effects:
        - self._columns_to_reconstruct == ['a']
        """
        # Setup
        instance = GreaterThan(low='a', high='b', drop='low')
        instance._columns_to_reconstruct == ['a']

    def test__init__get_columns_to_reconstruct_scalar_high(self):
        """Test the ``GreaterThan._get_columns_to_reconstruct`` method.

        This method returns:
            - ``_high`` if drop is "high"
            - ``_low`` if drop is "low"
            - ``_low`` if scalar is "high"
            - ``_high`` otherwise

        Setup:
        - low = 'a'
        - high = 0
        - scalar = 'high'
        Side effects:
        - self._columns_to_reconstruct == ['a']
        """
        # Setup
        instance = GreaterThan(low='a', high=0, scalar='high')
        instance._columns_to_reconstruct == ['a']

    def test__get_value_column_list(self):
        """Test the ``GreaterThan._get_value`` method.

        This method returns a scalar or a ndarray of values
        depending on the type of the ``field``.

        Input:
        - Table with given data.
        - field = 'low'
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        out = instance._get_value(table_data, 'low')

        # Assert
        expected = table_data[['a']].values
        np.testing.assert_array_equal(out, expected)

    def test__get_value_scalar(self):
        """Test the ``GreaterThan._get_value`` method.

        This method returns a scalar or a ndarray of values
        depending on the type of the ``field``.

        Input:
        - Table with given data.
        - field = 'low'
        - scalar = 'low'
        """
        # Setup
        instance = GreaterThan(low=3, high='b', scalar='low')

        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        out = instance._get_value(table_data, 'low')

        # Assert
        expected = 3
        assert out == expected

    def test__get_diff_columns_name_low_is_scalar(self):
        """Test the ``GreaterThan._get_diff_columns_name`` method.

        The returned names should be equal to the given columns plus
        tokenized with '#'.

        Input:
        - Table with given data.
        """
        # Setup
        instance = GreaterThan(low=0, high=['a', 'b#'], scalar='low')

        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b#': [4, 5, 6]
        })
        out = instance._get_diff_columns_name(table_data)

        # Assert
        expected = ['a#', 'b##']
        assert out == expected

    def test__get_diff_columns_name_high_is_scalar(self):
        """Test the ``GreaterThan._get_diff_columns_name`` method.

        The returned names should be equal to the given columns plus
        tokenized with '#'.

        Input:
        - Table with given data.
        """
        # Setup
        instance = GreaterThan(low=['a', 'b'], high=0, scalar='high')

        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6]
        })
        out = instance._get_diff_columns_name(table_data)

        # Assert
        expected = ['a#', 'b#']
        assert out == expected

    def test__get_diff_columns_name_scalar_is_none(self):
        """Test the ``GreaterThan._get_diff_columns_name`` method.

        The returned names should be equal one name of the two columns
        with a token between them.

        Input:
        - Table with given data.
        """
        # Setup
        instance = GreaterThan(low='a', high='b#', scalar=None)

        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b#': [4, 5, 6]
        })
        out = instance._get_diff_columns_name(table_data)

        # Assert
        expected = ['b##a']
        assert out == expected

    def test__get_diff_columns_name_scalar_is_none_multi_column_low(self):
        """Test the ``GreaterThan._get_diff_columns_name`` method.

        The returned names should be equal one name of the two columns
        with a token between them.

        Input:
        - Table with given data.
        """
        # Setup
        instance = GreaterThan(low=['a#', 'c'], high='b', scalar=None)

        table_data = pd.DataFrame({
            'a#': [1, 2, 4],
            'b': [4, 5, 6],
            'c#': [7, 8, 9]
        })
        out = instance._get_diff_columns_name(table_data)

        # Assert
        expected = ['a##b', 'c#b']
        assert out == expected

    def test__get_diff_columns_name_scalar_is_none_multi_column_high(self):
        """Test the ``GreaterThan._get_diff_columns_name`` method.

        The returned names should be equal one name of the two columns
        with a token between them.

        Input:
        - Table with given data.
        """
        # Setup
        instance = GreaterThan(low=0, high=['b', 'c'], scalar=None)

        table_data = pd.DataFrame({
            0: [1, 2, 4],
            'b': [4, 5, 6],
            'c#': [7, 8, 9]
        })
        out = instance._get_diff_columns_name(table_data)

        # Assert
        expected = ['b#0', 'c#0']
        assert out == expected

    def test__check_columns_exist_success(self):
        """Test the ``GreaterThan._check_columns_exist`` method.

        This method raises an error if the specified columns in
        ``low`` or ``high`` do not exist.

        Input:
        - Table with given data.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run / Assert
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6]
        })
        instance._check_columns_exist(table_data, 'low')
        instance._check_columns_exist(table_data, 'high')

    def test__check_columns_exist_error(self):
        """Test the ``GreaterThan._check_columns_exist`` method.

        This method raises an error if the specified columns in
        ``low`` or ``high`` do not exist.

        Input:
        - Table with given data.
        """
        # Setup
        instance = GreaterThan(low='a', high='c')

        # Run / Assert
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6]
        })
        instance._check_columns_exist(table_data, 'low')
        with pytest.raises(KeyError):
            instance._check_columns_exist(table_data, 'high')

    def test__fit_only_one_datetime_arg(self):
        """Test the ``Between._fit`` method by passing in only one arg as datetime.

        If only one of the high / low args is a datetime type, expect a ValueError.

        Input:
        - low is an int column
        - high is a datetime
        Output:
        - n/a
        Side Effects:
        - ValueError
        """
        # Setup
        instance = GreaterThan(low='a', high=pd.to_datetime('2021-01-01'), scalar='high')

        # Run and assert
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        with pytest.raises(ValueError):
            instance._fit(table_data)

    def test__fit__low_is_not_found_and_scalar_is_none(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should raise an error if
        the ``low`` is set to a value not seen in ``table_data``.

        Input:
        - Table without ``low`` in columns.
        Side Effect:
        - KeyError.
        """
        # Setup
        instance = GreaterThan(low=3, high='b')

        # Run / Assert
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        with pytest.raises(KeyError):
            instance._fit(table_data)

    def test__fit__high_is_not_found_and_scalar_is_none(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should raise an error if
        the ``high`` is set to a value not seen in ``table_data``.

        Input:
        - Table without ``high`` in columns.
        Side Effect:
        - KeyError.
        """
        # Setup
        instance = GreaterThan(low='a', high=3)

        # Run / Assert
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        with pytest.raises(KeyError):
            instance._fit(table_data)

    def test__fit__low_is_not_found_scalar_is_high(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should raise an error if
        the ``low`` is set to a value not seen in ``table_data``.

        Input:
        - Table without ``low`` in columns.
        Side Effect:
        - KeyError.
        """
        # Setup
        instance = GreaterThan(low='c', high=3, scalar='high')

        # Run / Assert
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        with pytest.raises(KeyError):
            instance._fit(table_data)

    def test__fit__high_is_not_found_scalar_is_high(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should raise an error if
        the ``high`` is set to a value not seen in ``table_data``.

        Input:
        - Table without ``high`` in columns.
        Side Effect:
        - KeyError.
        """
        # Setup
        instance = GreaterThan(low=3, high='c', scalar='low')

        # Run / Assert
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6]
        })
        with pytest.raises(KeyError):
            instance._fit(table_data)

    def test__fit__columns_to_reconstruct_drop_high(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should set ``_columns_to_reconstruct``
        to ``instance._high`` if ``instance_drop`` is `high`.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_columns_to_reconstruct`` is ``instance._high``
        """
        # Setup
        instance = GreaterThan(low='a', high='b', drop='high')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance._fit(table_data)

        # Asserts
        assert instance._columns_to_reconstruct == ['b']

    def test__fit__columns_to_reconstruct_drop_low(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should set ``_columns_to_reconstruct``
        to ``instance._low`` if ``instance_drop`` is `low`.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_columns_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high='b', drop='low')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance._fit(table_data)

        # Asserts
        assert instance._columns_to_reconstruct == ['a']

    def test__fit__columns_to_reconstruct_default(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should set ``_columns_to_reconstruct``
        to `high` by default.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_columns_to_reconstruct`` is ``instance._high``
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance._fit(table_data)

        # Asserts
        assert instance._columns_to_reconstruct == ['b']

    def test__fit__columns_to_reconstruct_high_is_scalar(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should set ``_columns_to_reconstruct``
        to `low` if ``instance._scalar`` is ``'high'``.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_columns_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high='b', scalar='high')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance._fit(table_data)

        # Asserts
        assert instance._columns_to_reconstruct == ['a']

    def test__fit__columns_to_reconstruct_low_is_scalar(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should set ``_columns_to_reconstruct``
        to `high` if ``instance._scalar`` is ``'low'``.

        Input:
        - Table with two columns.
        Side Effect:
        - ``_columns_to_reconstruct`` is ``instance._high``
        """
        # Setup
        instance = GreaterThan(low='a', high='b', scalar='low')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance._fit(table_data)

        # Asserts
        assert instance._columns_to_reconstruct == ['b']

    def test__fit__diff_columns_one_column(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should set ``_diff_columns``
        to the one column in ``instance.constraint_columns`` plus a
        token if there is only one column in that set.

        Input:
        - Table with one column.
        Side Effect:
        - ``_columns_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high=3, scalar='high')

        # Run
        table_data = pd.DataFrame({'a': [1, 2, 3]})
        instance._fit(table_data)

        # Asserts
        assert instance._diff_columns == ['a#']

    def test__fit__diff_columns_multiple_columns(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should set ``_diff_columns``
        to the two columns in ``instance.constraint_columns`` separated
        by a token if there both columns are in that set.

        Input:
        - Table with two column.
        Side Effect:
        - ``_columns_to_reconstruct`` is ``instance._low``
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance._fit(table_data)

        # Asserts
        assert instance._diff_columns == ['b#a']

    def test__fit_int(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_low_is_scalar`` and ``high_is_scalar`` are ``False``.

        Input:
        - Table that contains two constrained columns with the high one
          being made of integers.
        Side Effect:
        - The _dtype attribute gets `int` as the value even if the low
          column has a different dtype.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        instance._fit(table_data)

        # Asserts
        assert all([dtype.kind == 'i' for dtype in instance._dtype])

    def test__fit_float(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_low_is_scalar`` and ``high_is_scalar`` are ``False``.

        Input:
        - Table that contains two constrained columns with the high one
          being made of float values.
        Side Effect:
        - The _dtype attribute gets `float` as the value even if the low
          column has a different dtype.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9]
        })
        instance._fit(table_data)

        # Asserts
        assert all([dtype.kind == 'f' for dtype in instance._dtype])

    def test__fit_datetime(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_low_is_scalar`` and ``high_is_scalar`` are ``False``.

        Input:
        - Table that contains two constrained columns of datetimes.
        Side Effect:
        - The _dtype attribute gets `datetime` as the value.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01']),
            'b': pd.to_datetime(['2020-01-02'])
        })
        instance._fit(table_data)

        # Asserts
        assert all([dtype.kind == 'M' for dtype in instance._dtype])

    def test__fit_type__high_is_scalar(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should learn and store the
        ``dtype`` of the ``low`` column as the ``_dtype`` attribute
        if ``_scalar`` is ``'high'``.

        Input:
        - Table that contains two constrained columns with the low one
          being made of floats.
        Side Effect:
        - The _dtype attribute gets `float` as the value.
        """
        # Setup
        instance = GreaterThan(low='a', high=3, scalar='high')

        # Run
        table_data = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        instance._fit(table_data)

        # Asserts
        assert all([dtype.kind == 'f' for dtype in instance._dtype])

    def test__fit_type__low_is_scalar(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute
        if ``_scalar`` is ``'low'``.

        Input:
        - Table that contains two constrained columns with the high one
          being made of floats.
        Side Effect:
        - The _dtype attribute gets `float` as the value.
        """
        # Setup
        instance = GreaterThan(low=3, high='b', scalar='low')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9]
        })
        instance._fit(table_data)

        # Asserts
        assert all([dtype.kind == 'f' for dtype in instance._dtype])

    def test__fit_high_is_scalar_multi_column(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute.

        Input:
        - Table that contains two constrained columns with different dtype.
        """
        # Setup
        instance = GreaterThan(low=['a', 'b'], high=0, scalar='high')
        dtype_int = pd.Series([1]).dtype
        dtype_float = np.dtype('float')
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4., 5., 6.]
        })
        instance._fit(table_data)

        # Assert
        expected_diff_columns = ['a#', 'b#']
        expected_dtype = pd.Series([dtype_int, dtype_float], index=table_data.columns)
        assert instance._diff_columns == expected_diff_columns
        pd.testing.assert_series_equal(instance._dtype, expected_dtype)

    def test__fit_low_is_scalar_multi_column(self):
        """Test the ``GreaterThan._fit`` method.

        The ``GreaterThan._fit`` method should learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute.

        Input:
        - Table that contains two constrained columns with different dtype.
        """
        # Setup
        instance = GreaterThan(low=0, high=['a', 'b'], scalar='low')
        dtype_int = pd.Series([1]).dtype
        dtype_float = np.dtype('float')
        table_data = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4., 5., 6.]
        })
        instance._fit(table_data)

        # Assert
        expected_diff_columns = ['a#', 'b#']
        expected_dtype = pd.Series([dtype_int, dtype_float], index=table_data.columns)
        assert instance._diff_columns == expected_diff_columns
        pd.testing.assert_series_equal(instance._dtype, expected_dtype)

    def test_is_valid_strict_false(self):
        """Test the ``GreaterThan.is_valid`` method with strict False.

        If strict is False, equal values should count as valid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - False should be returned for the strictly invalid row and True
          for the other two.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=False)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, True, False]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_strict_true(self):
        """Test the ``GreaterThan.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - True should be returned for the strictly valid row and False
          for the other two.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, False, False]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_low_is_scalar_high_is_column(self):
        """Test the ``GreaterThan.is_valid`` method.

        If low is a scalar, and high is a column name, then
        the values in that column should all be higher than
        ``instance._low``.

        Input:
        - Table with values above and below low.
        Output:
        - True should be returned for the rows where the high
        column is above low.
        """
        # Setup
        instance = GreaterThan(low=3, high='b', strict=False, scalar='low')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, False, False]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_high_is_scalar_low_is_column(self):
        """Test the ``GreaterThan.is_valid`` method.

        If high is a scalar, and low is a column name, then
        the values in that column should all be lower than
        ``instance._high``.

        Input:
        - Table with values above and below high.
        Output:
        - True should be returned for the rows where the low
        column is below high.
        """
        # Setup
        instance = GreaterThan(low='a', high=2, strict=False, scalar='high')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, True, False]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_high_is_scalar_multi_column(self):
        """Test the ``GreaterThan.is_valid`` method.

        If high is a scalar, and low is multi column, then
        the values in that column should all be lower than
        ``instance._high``.

        Input:
        - Table with values above and below high.
        Output:
        - True should be returned for the rows where the low
        column is below high.
        """
        # Setup
        instance = GreaterThan(low=['a', 'b'], high=2, strict=False, scalar='high')
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [False, True, False]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_low_is_scalar_multi_column(self):
        """Test the ``GreaterThan.is_valid`` method.

        If low is a scalar, and high is multi column, then
        the values in that column should all be higher than
        ``instance._low``.

        Input:
        - Table with values above and below low.
        Output:
        - True should be returned for the rows where the high
        column is above low.
        """
        # Setup
        instance = GreaterThan(low=2, high=['a', 'b'], strict=False, scalar='low')
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_scalar_is_none_multi_column(self):
        """Test the ``GreaterThan.is_valid`` method.

        If scalar is none, and high is multi column, then
        the values in that column should all be higher than
        in the low column.

        Input:
        - Table with values above and below low.
        Output:
        - True should be returned for the rows where the high
        column is above low.
        """
        # Setup
        instance = GreaterThan(low='b', high=['a', 'c'], strict=False)
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 2],
            'c': [7, 8, 9]
        })

        # Run
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_high_is_datetime(self):
        """Test the ``GreaterThan.is_valid`` method.

        If high is a datetime and low is a column,
        the values in that column should all be lower than
        ``instance._high``.

        Input:
        - Table with values above and below `high`.
        Output:
        - True should be returned for the rows where the low
        column is below `high`.
        """
        # Setup
        high_dt = pd.to_datetime('8/31/2021')
        instance = GreaterThan(low='a', high=high_dt, strict=False, scalar='high')
        table_data = pd.DataFrame({
            'a': [datetime(2020, 5, 17), datetime(2020, 2, 1), datetime(2021, 9, 1)],
            'b': [4, 2, 2],
        })

        # Run
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, True, False]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_low_is_datetime(self):
        """Test the ``GreaterThan.is_valid`` method.

        If low is a datetime and high is a column,
        the values in that column should all be higher than
        ``instance._low``.

        Input:
        - Table with values above and below `low`.
        Output:
        - True should be returned for the rows where the high
        column is above `low`.
        """
        # Setup
        low_dt = pd.to_datetime('8/31/2021')
        instance = GreaterThan(low=low_dt, high='a', strict=False, scalar='low')
        table_data = pd.DataFrame({
            'a': [datetime(2021, 9, 17), datetime(2021, 7, 1), datetime(2021, 9, 1)],
            'b': [4, 2, 2],
        })

        # Run
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, False, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_two_cols_with_nans(self):
        """Test the ``GreaterThan.is_valid`` method with nan values.

        If there is a NaN row, expect that `is_valid` returns True.

        Input:
        - Table with a NaN row
        Output:
        - True should be returned for the NaN row.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, None, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, True, False]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_two_cols_with_one_nan(self):
        """Test the ``GreaterThan.is_valid`` method with nan values.

        If there is a row in which we compare one NaN value with one
        non-NaN value, expect that `is_valid` returns True.

        Input:
        - Table with a row that contains only one NaN value.
        Output:
        - True should be returned for the row with the NaN value.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, 5, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = [True, True, False]
        np.testing.assert_array_equal(expected_out, out)

    def test__transform_int_drop_none(self):
        """Test the ``GreaterThan._transform`` method passing a high column of type int.

        The ``GreaterThan._transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1.

        Setup:
        - ``_drop`` is set to ``None``, so all original columns will be in output.
        Input:
        - Table with two columns two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._diff_columns = ['a#b']

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
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_int_drop_high(self):
        """Test the ``GreaterThan._transform`` method passing a high column of type int.

        The ``GreaterThan._transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1. It should also drop the high column.

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with two columns two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4) and the high column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._diff_columns = ['a#b']

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

    def test__transform_int_drop_low(self):
        """Test the ``GreaterThan._transform`` method passing a high column of type int.

        The ``GreaterThan._transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1. It should also drop the low column.

        Setup:
        - ``_drop`` is set to ``low``.
        Input:
        - Table with two columns two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4) and the low column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._diff_columns = ['a#b']

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_float_drop_none(self):
        """Test the ``GreaterThan._transform`` method passing a high column of type float.

        The ``GreaterThan._transform`` method is expected to compute the distance
        between the high and low columns and create a diff column with the
        logarithm of the distance + 1.

        Setup:
        - ``_drop`` is set to ``None``, so all original columns will be in output.
        Input:
        - Table with two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._diff_columns = ['a#b']

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9],
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_datetime_drop_none(self):
        """Test the ``GreaterThan._transform`` method passing a high column of type datetime.

        If the columns are of type datetime, ``_transform`` is expected
        to convert the timedelta distance into numeric before applying
        the +1 and logarithm.

        Setup:
        - ``_drop`` is set to ``None``, so all original columns will be in output.
        Input:
        - Table with values at a distance of exactly 1 second.
        Output:
        - Same table with a diff column of the logarithms
          of the dinstance in nanoseconds + 1, which is np.log(1_000_000_001).
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._diff_columns = ['a#b']
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
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_not_all_columns_provided(self):
        """Test the ``GreaterThan.transform`` method.

        If some of the columns needed for the transform are missing, it will raise
        a ``MissingConstraintColumnError``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Raises ``MissingConstraintColumnError``.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, fit_columns_model=False)

        # Run/Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame({'a': ['a', 'b', 'c']}))

    def test__transform_high_is_scalar(self):
        """Test the ``GreaterThan._transform`` method with high as scalar.

        The ``GreaterThan._transform`` method is expected to compute the distance
        between the high scalar value and the low column and create a diff column
        with the logarithm of the distance + 1.

        Setup:
        - ``_high`` is set to 5 and ``_scalar`` is ``'high'``.
        Input:
        - Table with one low column and two dummy columns.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low='a', high=5, strict=True, scalar='high')
        instance._diff_columns = ['a#b']
        instance.constraint_columns = ['a']

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
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(5), np.log(4), np.log(3)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_low_is_scalar(self):
        """Test the ``GreaterThan._transform`` method with high as scalar.

        The ``GreaterThan._transform`` method is expected to compute the distance
        between the high scalar value and the low column and create a diff column
        with the logarithm of the distance + 1.

        Setup:
        - ``_high`` is set to 5 and ``_scalar`` is ``'low'``.
        Input:
        - Table with one low column and two dummy columns.
        Output:
        - Same table with a diff column of the logarithms of the distances + 1,
        which is np.log(4).
        """
        # Setup
        instance = GreaterThan(low=2, high='b', strict=True, scalar='low')
        instance._diff_columns = ['a#b']
        instance.constraint_columns = ['b']

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
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(3), np.log(4), np.log(5)],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test__transform_high_is_scalar_multi_column(self):
        """Test the ``GreaterThan._transform`` method.

        The ``GreaterThan._transform`` method is expected to compute the logarithm
        of given columns + 1.

        Input:
        - Table with given data.
        Output:
        - Same table with additional columns of the logarithms + 1.
        """
        # Setup
        instance = GreaterThan(low=['a', 'b'], high=3, strict=True, scalar='high')
        instance._diff_columns = ['a#', 'b#']
        instance.constraint_columns = ['a', 'b']

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance._transform(table_data)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#': [np.log(3), np.log(2), np.log(1)],
            'b#': [np.log(0), np.log(-1), np.log(-2)],
        })
        pd.testing.assert_frame_equal(out, expected)

    def test__transform_low_is_scalar_multi_column(self):
        """Test the ``GreaterThan._transform`` method.

        The ``GreaterThan._transform`` method is expected to compute the logarithm
        of given columns + 1.

        Input:
        - Table with given data.
        Output:
        - Same table with additional columns of the logarithms + 1.
        """
        # Setup
        instance = GreaterThan(low=3, high=['a', 'b'], strict=True, scalar='low')
        instance._diff_columns = ['a#', 'b#']
        instance.constraint_columns = ['a', 'b']

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance._transform(table_data)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#': [np.log(-1), np.log(0), np.log(1)],
            'b#': [np.log(2), np.log(3), np.log(4)],
        })
        pd.testing.assert_frame_equal(out, expected)

    def test__transform_scalar_is_none_multi_column(self):
        """Test the ``GreaterThan._transform`` method.

        The ``GreaterThan._transform`` method is expected to compute the logarithm
        of given columns + 1.

        Input:
        - Table with given data.
        Output:
        - Same table with additional columns of the logarithms + 1.
        """
        # Setup
        instance = GreaterThan(low=['a', 'c'], high='b', strict=True)
        instance._diff_columns = ['a#', 'c#']
        instance.constraint_columns = ['a', 'c']

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance._transform(table_data)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#': [np.log(4)] * 3,
            'c#': [np.log(-2)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected)

    def test_reverse_transform_int_drop_high(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype int.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to integers
            - add back the dropped column

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as int
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._dtype = [pd.Series([1]).dtype]    # exact dtype (32 or 64) depends on OS
        instance._diff_columns = ['a#b']
        instance._columns_to_reconstruct = ['b']

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

    def test_reverse_transform_float_drop_high(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype float.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to float values
            - add back the dropped column

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as float values
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._dtype = [np.dtype('float')]
        instance._diff_columns = ['a#b']
        instance._columns_to_reconstruct = ['b']

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

    def test_reverse_transform_datetime_drop_high(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype datetime.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - convert the distance to a timedelta
            - add the low column
            - convert the output to datetimes

        Setup:
        - ``_drop`` is set to ``high``.
        Input:
        - Table with a diff column that contains the constant np.log(1_000_000_001).
        Output:
        - Same table with the high column replaced by the low one + one second
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='high')
        instance._dtype = [np.dtype('<M8[ns]')]
        instance._diff_columns = ['a#b']
        instance._is_datetime = True
        instance._columns_to_reconstruct = ['b']

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

    def test_reverse_transform_int_drop_low(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype int.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - subtract from the high column
            - convert the output to integers
            - add back the dropped column

        Setup:
        - ``_drop`` is set to ``low``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        Output:
        - Same table with the low column replaced by the high one - 3, as int
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._dtype = [pd.Series([1]).dtype]    # exact dtype (32 or 64) depends on OS
        instance._diff_columns = ['a#b']
        instance._columns_to_reconstruct = ['a']

        # Run
        transformed = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a': [1, 2, 3],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime_drop_low(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype datetime.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - convert the distance to a timedelta
            - subtract from the high column
            - convert the output to datetimes

        Setup:
        - ``_drop`` is set to ``low``.
        Input:
        - Table with a diff column that contains the constant np.log(1_000_000_001).
        Output:
        - Same table with the low column replaced by the high one - one second
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._dtype = [np.dtype('<M8[ns]')]
        instance._diff_columns = ['a#b']
        instance._is_datetime = True
        instance._columns_to_reconstruct = ['a']

        # Run
        transformed = pd.DataFrame({
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00'])
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_int_drop_none(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype int.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        The table should have one invalid row where the low column is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low one + 3 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = [pd.Series([1]).dtype]    # exact dtype (32 or 64) depends on OS
        instance._diff_columns = ['a#b']
        instance._columns_to_reconstruct = ['b']

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 1, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime_drop_none(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype datetime.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - convert the distance to a timedelta
            - add the low column when the row is invalid
            - convert the output to datetimes

        Setup:
        - ``_drop`` is set to ``None``.
        Input:
        - Table with a diff column that contains the constant np.log(1_000_000_001).
        The table should have one invalid row where the low column is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low one + one second
        for all invalid rows, and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = [np.dtype('<M8[ns]')]
        instance._diff_columns = ['a#b']
        instance._is_datetime = True
        instance._columns_to_reconstruct = ['b']

        # Run
        transformed = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-01T00:00:01']),
            'c': [1, 2],
            'a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2]
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_low_is_scalar(self):
        """Test the ``GreaterThan.reverse_transform`` method with low as a scalar.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low value when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        - ``_low`` is set to an int and ``_scalar`` is ``'low'``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        The table should have one invalid row where the low value is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low value + 3 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low=3, high='b', strict=True, scalar='low')
        instance._dtype = [pd.Series([1]).dtype]    # exact dtype (32 or 64) depends on OS
        instance._diff_columns = ['a#b']
        instance._columns_to_reconstruct = ['b']

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 1, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 6, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_high_is_scalar(self):
        """Test the ``GreaterThan.reverse_transform`` method with high as a scalar.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - subtract from the high value when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        - ``_high`` is set to an int and ``_scalar`` is ``'high'``.
        Input:
        - Table with a diff column that contains the constant np.log(4).
        The table should have one invalid row where the low column is
        higher than the high value.
        Output:
        - Same table with the low column replaced by the high one - 3 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high=3, strict=True, scalar='high')
        instance._dtype = [pd.Series([1]).dtype]    # exact dtype (32 or 64) depends on OS
        instance._diff_columns = ['a#b']
        instance._columns_to_reconstruct = ['a']

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b': [np.log(4)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 0],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_high_is_scalar_multi_column(self):
        """Test the ``GreaterThan.reverse_transform`` method with high as a scalar.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - subtract from the high value when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        - ``_high`` is set to an int and ``_scalar`` is ``'high'``.
        - ``_low`` is set to multiple columns.
        Input:
        - Table with a diff column that contains the constant np.log(4)/np.log(5).
        The table should have one invalid row where the low column is
        higher than the high value.
        Output:
        - Same table with the low column replaced by the high one - 3/-4 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low=['a', 'b'], high=3, strict=True, scalar='high')
        dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._dtype = [dtype, dtype]
        instance._diff_columns = ['a#', 'b#']
        instance._columns_to_reconstruct = ['a', 'b']

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [0, 5, 6],
            'c': [7, 8, 9],
            'a#': [np.log(4)] * 3,
            'b#': [np.log(5)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 0, 0],
            'b': [0, -1, -1],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_low_is_scalar_multi_column(self):
        """Test the ``GreaterThan.reverse_transform`` method with low as a scalar.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low value when the row is invalid
            - convert the output to integers

        Setup:
        - ``_drop`` is set to ``None``.
        - ``_low`` is set to an int and ``_scalar`` is ``'low'``.
        - ``_high`` is set to multiple columns.
        Input:
        - Table with a diff column that contains the constant np.log(4)/np.log(5).
        The table should have one invalid row where the low value is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low value +3/+4 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low=3, high=['a', 'b'], strict=True, scalar='low')
        dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._dtype = [dtype, dtype]
        instance._diff_columns = ['a#', 'b#']
        instance._columns_to_reconstruct = ['a', 'b']

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#': [np.log(4)] * 3,
            'b#': [np.log(5)] * 3,
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [6, 6, 4],
            'b': [7, 7, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_scalar_is_none_multi_column(self):
        """Test the ``GreaterThan.reverse_transform`` method with low as a scalar.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low value when the row is invalid
            - convert the output to integers

        Setup:
        - ``_low`` = ['a', 'c'].
        - ``_high`` = ['b'].
        Input:
        - Table with a diff column that contains the constant np.log(4)/np.log(-2).
        The table should have one invalid row where the low value is
        higher than the high column.
        Output:
        - Same table with the high column replaced by the low value +3/-4 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low=['a', 'c'], high=['b'], strict=True)
        dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._dtype = [dtype, dtype]
        instance._diff_columns = ['a#', 'c#']
        instance._columns_to_reconstruct = ['a', 'c']

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#': [np.log(1)] * 3,
            'c#': [np.log(1)] * 3,
        })
        out = instance.reverse_transform(transformed)
        print(out)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 4],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_multi_column_positive(self):
        """Test the ``GreaterThan.reverse_transform`` method for positive constraint.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - subtract from the high value when the row is invalid
            - convert the output to integers

        Input:
        - Table with given data.
        Output:
        - Same table with with replaced rows and dropped columns.
        """
        # Setup
        instance = GreaterThan(low=0, high=['a', 'b'], strict=True, scalar='low')
        dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._dtype = [dtype, dtype]
        instance._diff_columns = ['a#', 'b#']
        instance._columns_to_reconstruct = ['a', 'b']

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, -1],
            'c': [7, 8, 9],
            'a#': [np.log(2), np.log(3), np.log(4)],
            'b#': [np.log(5), np.log(6), np.log(0)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 0],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_multi_column_negative(self):
        """Test the ``GreaterThan.reverse_transform`` method for negative constraint.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - subtract from the high value when the row is invalid
            - convert the output to integers

        Input:
        - Table with given data.
        Output:
        - Same table with with replaced rows and dropped columns.
        """
        # Setup
        instance = GreaterThan(low=['a', 'b'], high=0, strict=True, scalar='high')
        dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS
        instance._dtype = [dtype, dtype]
        instance._diff_columns = ['a#', 'b#']
        instance._columns_to_reconstruct = ['a', 'b']

        # Run
        transformed = pd.DataFrame({
            'a': [-1, -2, 1],
            'b': [-4, -5, -1],
            'c': [7, 8, 9],
            'a#': [np.log(2), np.log(3), np.log(0)],
            'b#': [np.log(5), np.log(6), np.log(2)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [-1, -2, 0],
            'b': [-4, -5, -1],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)


class TestPositive():

    def test__init__(self):
        """
        Test the ``Positive.__init__`` method.

        The method is expected to set the ``_low`` instance variable
        to 0, the ``_scalar`` variable to ``'low'``. The rest of the
        parameters should be passed. Check that ``_drop`` is set to
        ``None`` when ``drop`` is ``False``.

        Input:
        - strict = True
        - low = 'a'
        - drop = False
        Side effects:
        - instance._low == 'a'
        - instance._high == 0
        - instance._strict == True
        - instance._scalar == 'low'
        - instance._drop = None
        """
        # Run
        instance = Positive(columns='a', strict=True, drop=False)

        # Asserts
        assert instance._low == 0
        assert instance._high == ['a']
        assert instance._strict is True
        assert instance._scalar == 'low'
        assert instance._drop is None

    def test__init__drop_true(self):
        """
        Test the ``Positive.__init__`` method with drop is ``True``.

        Check that ``_drop`` is set to 'high' when ``drop`` is ``True``.

        Input:
        - strict = True
        - low = 'a'
        - drop = True
        Side effects:
        - instance._low == 'a'
        - instance._high == 0
        - instance._strict == True
        - instance._scalar == 'low'
        - instance._drop = 'high'
        """
        # Run
        instance = Positive(columns='a', strict=True, drop=True)

        # Asserts
        assert instance._low == 0
        assert instance._high == ['a']
        assert instance._strict is True
        assert instance._scalar == 'low'
        assert instance._drop == 'high'


class TestNegative():

    def test__init__(self):
        """
        Test the ``Negative.__init__`` method.

        The method is expected to set the ``_high`` instance variable
        to 0, the ``_scalar`` variable to ``'high'``. The rest of the
        parameters should be passed. Check that ``_drop`` is set to
        ``None`` when ``drop`` is ``False``.

        Input:
        - strict = True
        - low = 'a'
        - drop = False
        Side effects:
        - instance._low == 'a'
        - instance._high == 0
        - instance._strict == True
        - instance._scalar = 'high'
        - instance._drop = None
        """
        # Run
        instance = Negative(columns='a', strict=True, drop=False)

        # Asserts
        assert instance._low == ['a']
        assert instance._high == 0
        assert instance._strict is True
        assert instance._scalar == 'high'
        assert instance._drop is None

    def test__init__drop_true(self):
        """
        Test the ``Negative.__init__`` method with drop is ``True``.

        Check that ``_drop`` is set to 'low' when ``drop`` is ``True``.

        Input:
        - strict = True
        - low = 'a'
        - drop = True
        Side effects:
        - instance._low == 'a'
        - instance._high == 0
        - instance._strict == True
        - instance._scalar = 'high'
        - instance._drop = 'low'
        """
        # Run
        instance = Negative(columns='a', strict=True, drop=True)

        # Asserts
        assert instance._low == ['a']
        assert instance._high == 0
        assert instance._strict is True
        assert instance._scalar == 'high'
        assert instance._drop == 'low'


def new_column(data):
    """Formula to be used for the ``TestColumnFormula`` class."""
    if data['a'] is None or data['b'] is None:
        return None

    return data['a'] + data['b']


class TestColumnFormula():

    def test___init__(self):
        """Test the ``ColumnFormula.__init__`` method.

        It is expected to create a new Constraint instance,
        import the formula to use for the computation, and
        set the specified constraint column.

        Input:
        - column = 'col'
        - formula = new_column
        """
        # Setup
        column = 'col'

        # Run
        instance = ColumnFormula(column=column, formula=new_column)

        # Assert
        assert instance._column == column
        assert instance._formula == new_column
        assert instance.constraint_columns == ('col', )

    def test___init__sets_rebuild_columns_if_not_reject_sampling(self):
        """Test the ``ColumnFormula.__init__`` method.

        The rebuild columns should only be set if the ``handling_strategy``
        is not ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are set
        """
        # Setup
        column = 'col'

        # Run
        instance = ColumnFormula(column=column, formula=new_column, handling_strategy='transform')

        # Assert
        assert instance.rebuild_columns == (column,)

    def test___init__does_not_set_rebuild_columns_reject_sampling(self):
        """Test the ``ColumnFormula.__init__`` method.

        The rebuild columns should not be set if the ``handling_strategy``
        is ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are empty
        """
        # Setup
        column = 'col'

        # Run
        instance = ColumnFormula(column=column, formula=new_column,
                                 handling_strategy='reject_sampling')

        # Assert
        assert instance.rebuild_columns == ()

    def test_is_valid_valid(self):
        """Test the ``ColumnFormula.is_valid`` method for a valid data.

        If the data fulfills the formula, result is a series of ``True`` values.

        Input:
        - Table data fulfilling the formula (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_non_valid(self):
        """Test the ``ColumnFormula.is_valid`` method for a non-valid data.

        If the data does not fulfill the formula, result is a series of ``False`` values.

        Input:
        - Table data not fulfilling the formula (pandas.DataFrame)
        Output:
        - Series of ``False`` values (pandas.Series)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        })
        instance = ColumnFormula(column=column, formula=new_column)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_with_nans(self):
        """Test the ``ColumnFormula.is_valid`` method for with a formula that produces nans.

        If the data fulfills the formula, result is a series of ``True`` values.

        Input:
        - Table data fulfilling the formula (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, None],
            'c': [5, 7, None]
        })
        instance = ColumnFormula(column=column, formula=new_column)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test__transform(self):
        """Test the ``ColumnFormula._transform`` method.

        It is expected to drop the indicated column from the table.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data without the indicated column (pandas.DataFrame)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_without_dropping_column(self):
        """Test the ``ColumnFormula._transform`` method without dropping the column.

        If `drop_column` is false, expect to not drop the constraint column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with the indicated column (pandas.DataFrame)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column, drop_column=False)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_missing_column(self):
        """Test the ``ColumnFormula._transform`` method when the constraint column is missing.

        When ``_transform`` is called with data that does not contain the constraint column,
        expect to return the data as-is.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data, unchanged (pandas.DataFrame)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'd': [5, 7, 9]
        })
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'd': [5, 7, 9]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform(self):
        """Test the ``ColumnFormula.reverse_transform`` method.

        It is expected to compute the indicated column by applying the given formula.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        column = 'c'
        instance = ColumnFormula(column=column, formula=new_column)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 1, 1]
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })
        pd.testing.assert_frame_equal(expected_out, out)


class TestRounding():

    def test___init__(self):
        """Test the ``Rounding.__init__`` method.

        It is expected to create a new Constraint instance
        and set the rounding args.

        Input:
        - columns = ['b', 'c']
        - digits = 2
        """
        # Setup
        columns = ['b', 'c']
        digits = 2

        # Run
        instance = Rounding(columns=columns, digits=digits)

        # Assert
        assert instance._columns == columns
        assert instance._digits == digits

    def test___init__invalid_digits(self):
        """Test the ``Rounding.__init__`` method with an invalid argument.

        Pass in an invalid ``digits`` argument, and expect a ValueError.

        Input:
        - columns = ['b', 'c']
        - digits = 20
        """
        # Setup
        columns = ['b', 'c']
        digits = 20

        # Run
        with pytest.raises(ValueError):
            Rounding(columns=columns, digits=digits)

    def test___init__invalid_tolerance(self):
        """Test the ``Rounding.__init__`` method with an invalid argument.

        Pass in an invalid ``tolerance`` argument, and expect a ValueError.

        Input:
        - columns = ['b', 'c']
        - digits = 2
        - tolerance = 0.1
        """
        # Setup
        columns = ['b', 'c']
        digits = 2
        tolerance = 0.1

        # Run
        with pytest.raises(ValueError):
            Rounding(columns=columns, digits=digits, tolerance=tolerance)

    def test_is_valid_positive_digits(self):
        """Test the ``Rounding.is_valid`` method for a positive digits argument.

        Input:
        - Table data with desired decimal places (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        columns = ['b', 'c']
        digits = 2
        tolerance = 1e-3
        instance = Rounding(columns=columns, digits=digits, tolerance=tolerance)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [4.12, 5.51, None, 6.941, 1.129],
            'c': [5.315, 7.12, 1.12, 9.131, 12.329],
            'd': ['a', 'b', 'd', 'e', None],
            'e': [123.31598, -1.12001, 1.12453, 8.12129, 1.32923]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, True, False, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_negative_digits(self):
        """Test the ``Rounding.is_valid`` method for a negative digits argument.

        Input:
        - Table data with desired decimal places (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        columns = ['b']
        digits = -2
        tolerance = 1
        instance = Rounding(columns=columns, digits=digits, tolerance=tolerance)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [401, 500, 6921, 799, None],
            'c': [5.3134, 7.1212, 9.1209, 101.1234, None],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_zero_digits(self):
        """Test the ``Rounding.is_valid`` method for a zero digits argument.

        Input:
        - Table data not with the desired decimal places (pandas.DataFrame)
        Output:
        - Series of ``False`` values (pandas.Series)
        """
        # Setup
        columns = ['b', 'c']
        digits = 0
        tolerance = 1e-4
        instance = Rounding(columns=columns, digits=digits, tolerance=tolerance)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, None, 3, 4],
            'b': [4, 5.5, 1.2, 6.0001, 5.99999],
            'c': [5, 7.12, 1.31, 9.00001, 4.9999],
            'd': ['a', 'b', None, 'd', 'e'],
            'e': [2.1254, 17.12123, 124.12, 123.0112, -9.129434]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_reverse_transform_positive_digits(self):
        """Test the ``Rounding.reverse_transform`` method with positive digits.

        Expect that the columns are rounded to the specified integer digit.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        columns = ['b', 'c']
        digits = 3
        instance = Rounding(columns=columns, digits=digits)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, None, 4],
            'b': [4.12345, None, 5.100, 6.0001, 1.7999],
            'c': [1.1, 1.234, 9.13459, 4.3248, 6.1312],
            'd': ['a', 'b', 'd', 'e', None]
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3, None, 4],
            'b': [4.123, None, 5.100, 6.000, 1.800],
            'c': [1.100, 1.234, 9.135, 4.325, 6.131],
            'd': ['a', 'b', 'd', 'e', None]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_negative_digits(self):
        """Test the ``Rounding.reverse_transform`` method with negative digits.

        Expect that the columns are rounded to the specified integer digit.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        columns = ['b']
        digits = -3
        instance = Rounding(columns=columns, digits=digits)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [41234.5, None, 5000, 6001, 5928],
            'c': [1.1, 1.23423, 9.13459, 12.12125, 18.12152],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [41000.0, None, 5000.0, 6000.0, 6000.0],
            'c': [1.1, 1.23423, 9.13459, 12.12125, 18.12152],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_zero_digits(self):
        """Test the ``Rounding.reverse_transform`` method with zero digits.

        Expect that the columns are rounded to the specified integer digit.

        Input:
        - Table data with the column with incorrect values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        columns = ['b', 'c']
        digits = 0
        instance = Rounding(columns=columns, digits=digits)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [4.12345, None, 5.0, 6.01, 7.9],
            'c': [1.1, 1.0, 9.13459, None, 8.89],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [4.0, None, 5.0, 6.0, 8.0],
            'c': [1.0, 1.0, 9.0, None, 9.0],
            'd': ['a', 'b', 'd', 'e', 'f']
        })
        pd.testing.assert_frame_equal(expected_out, out)


def transform(data, low, high):
    """Transform to be used for the TestBetween class."""
    data = (data - low) / (high - low) * 0.95 + 0.025
    return np.log(data / (1.0 - data))


class TestBetween():

    def test___init__sets_rebuild_columns_if_not_reject_sampling(self):
        """Test the ``Between.__init__`` method.

        The rebuild columns should only be set if the ``handling_strategy``
        is not ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are set
        """
        # Setup
        column = 'col'

        # Run
        instance = Between(column=column, low=10, high=20, handling_strategy='transform')

        # Assert
        assert instance.rebuild_columns == (column,)

    def test___init__does_not_set_rebuild_columns_reject_sampling(self):
        """Test the ``Between.__init__`` method.

        The rebuild columns should not be set if the ``handling_strategy``
        is ``reject_sampling``.

        Side effects:
        - instance.rebuild_columns are empty
        """
        # Setup
        column = 'col'

        # Run
        instance = Between(column=column, low=10, high=20, handling_strategy='reject_sampling')

        # Assert
        assert instance.rebuild_columns == ()

    def test_fit_only_one_datetime_arg(self):
        """Test the ``Between.fit`` method by passing in only one arg as datetime.

        If only one of the bound parameters is a datetime type, expect a ValueError.

        Input:
        - low is an int scalar
        - high is a datetime
        Output:
        - n/a
        Side Effects:
        - ValueError
        """
        # Setup
        column = 'a'
        low = 0.0
        high = pd.to_datetime('2021-01-01')
        instance = Between(column=column, low=low, high=high)

        # Run and assert
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [4, 5, 6],
        })
        with pytest.raises(ValueError):
            instance.fit(table_data)

    def test_transform_scalar_scalar(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as scalars.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True,
                           low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [4, 5, 6],
        })
        instance.fit(table_data)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'a#0.0#1.0': transform(table_data[column], low, high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_scalar_column(self):
        """Test the ``Between._transform`` method with ``low`` as scalar and ``high`` as a column.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0.5, 1, 6],
        })
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0.5, 1, 6],
            'a#0.0#b': transform(table_data[column], low, table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_column_scalar(self):
        """Test the ``Between._transform`` method with ``low`` as a column and ``high`` as scalar.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
        })
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0, -1, 0.5],
            'a#b#1.0': transform(table_data[column], table_data[low], high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_column_column(self):
        """Test the ``Between._transform`` method by passing ``low`` and ``high`` as columns.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6]
        })
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            'a#b#c': transform(table_data[column], table_data[low], table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_datetime_datetime(self):
        """Test the ``Between._transform`` method by passing ``low`` and ``high`` as datetimes.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        - High and Low as datetimes
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = pd.to_datetime('1900-01-01')
        high = pd.to_datetime('2021-01-01')
        instance = Between(column=column, low=low, high=high, high_is_scalar=True,
                           low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-01'),
                pd.to_datetime('2020-08-03'),
            ],
            'b': [4, 5, 6],
        })
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'a#1900-01-01T00:00:00.000000000#2021-01-01T00:00:00.000000000': transform(
                table_data[column], low, high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_datetime_column(self):
        """Test the ``Between._transform`` method with ``low`` as datetime and ``high`` as a column.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = pd.to_datetime('1900-01-01')
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-01'),
                pd.to_datetime('2020-08-03'),
            ],
            'b': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ],
        })
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ],
            'a#1900-01-01T00:00:00.000000000#b': transform(
                table_data[column], low, table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_column_datetime(self):
        """Test the ``Between._transform`` method with ``low`` as a column and ``high`` as datetime.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = pd.to_datetime('2021-01-01')
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-01'),
                pd.to_datetime('2020-08-03'),
            ],
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
        })
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
            'a#b#2021-01-01T00:00:00.000000000': transform(
                table_data[column], table_data[low], high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__transform_column_column_datetime(self):
        """Test the ``Between._transform`` method with ``low`` and ``high`` as datetime columns.

        It is expected to create a new column similar to the constraint ``column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column containing the transformed ``column`` (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-01'),
                pd.to_datetime('2020-08-03'),
            ],
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
            'c': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ]
        })
        instance.fit(table_data)
        out = instance._transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
            'c': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ],
            'a#b#c': transform(table_data[column], table_data[low], table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_scalar_scalar(self):
        """Test ``Between.reverse_transform`` with ``low`` and ``high`` as scalars.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True,
                           low_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [4, 5, 6],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [4, 5, 6],
            'a#0.0#1.0': transform(table_data[column], low, high)
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_scalar_column(self):
        """Test ``Between.reverse_transform`` with ``low`` as scalar and ``high`` as a column.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [0.5, 1, 6],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [0.5, 1, 6],
            'a#0.0#b': transform(table_data[column], low, table_data[high])
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_column_scalar(self):
        """Test ``Between.reverse_transform`` with ``low`` as a column and ``high`` as scalar.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [0, -1, 0.5],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [0, -1, 0.5],
            'a#b#1.0': transform(table_data[column], table_data[low], high)
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_column_column(self):
        """Test ``Between.reverse_transform`` with ``low`` and ``high`` as columns.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        table_data = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            'a': [0.1, 0.5, 0.9]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            'a#b#c': transform(table_data[column], table_data[low], table_data[high])
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_datetime_datetime(self):
        """Test ``Between.reverse_transform`` with ``low`` and ``high`` as datetime.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        - High and low as datetimes
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = pd.to_datetime('1900-01-01')
        high = pd.to_datetime('2021-01-01')
        instance = Between(column=column, low=low, high=high, high_is_scalar=True,
                           low_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [4, 5, 6],
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-01'),
                pd.to_datetime('2020-08-03'),
            ],
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [4, 5, 6],
            'a#1900-01-01T00:00:00.000000000#2021-01-01T00:00:00.000000000': transform(
                table_data[column], low, high)
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_series_equal(expected_out['b'], out['b'])
        pd.testing.assert_series_equal(expected_out['a'], out['a'].astype('datetime64[ms]'))

    def test_reverse_transform_datetime_column(self):
        """Test ``Between.reverse_transform`` with ``low`` as datetime and ``high`` as a column.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = pd.to_datetime('1900-01-01')
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ],
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-02'),
                pd.to_datetime('2020-08-03'),
            ]
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ],
            'a#1900-01-01T00:00:00.000000000#b': transform(
                table_data[column], low, table_data[high])
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_column_datetime(self):
        """Test ``Between.reverse_transform`` with ``low`` as a column and ``high`` as datetime.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = pd.to_datetime('2021-01-01')
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        table_data = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-03'),
                pd.to_datetime('2020-08-04'),
            ],
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
            'a#b#2021-01-01T00:00:00.000000000': transform(
                table_data[column], table_data[low], high)
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_series_equal(expected_out['b'], out['b'])
        pd.testing.assert_series_equal(expected_out['a'], out['a'].astype('datetime64[ms]'))

    def test_reverse_transform_column_column_datetime(self):
        """Test ``Between.reverse_transform`` with ``low`` and ``high`` as datetime columns.

        It is expected to recover the original table which was transformed, but with different
        column order. It does so by applying a sigmoid to the transformed column and then
        scaling it back to the original space. It also replaces the transformed column with
        an equal column but with the original name.

        Input:
        - Transformed table data (pandas.DataFrame)
        Output:
        - Original table data, without necessarily keepying the column order (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        table_data = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
            'c': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ],
            'a': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-01'),
                pd.to_datetime('2020-08-03'),
            ],
        })

        # Run
        instance.fit(table_data)
        transformed = pd.DataFrame({
            'b': [
                pd.to_datetime('2020-01-03'),
                pd.to_datetime('2020-02-01'),
                pd.to_datetime('2020-02-03'),
            ],
            'c': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-03'),
            ],
            'a#b#c': transform(table_data[column], table_data[low], table_data[high])
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_is_valid_strict_true(self):
        """Test the ``Between.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a valid row, a strictly invalid row and an
          invalid row. (pandas.DataFrame)
        Output:
        - True should be returned for the valid row and False
          for the other two. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, strict=True, high_is_scalar=True,
                           low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 1, 3],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out, check_names=False)

    def test_is_valid_strict_false(self):
        """Test the ``Between.is_valid`` method with strict False.

        If strict is False, equal values should count as valid.

        Input:
        - Table with a valid row, a strictly invalid row and an
          invalid row. (pandas.DataFrame)
        Output:
        - True should be returned for the first two rows, and False
          for the last one (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, strict=False, high_is_scalar=True,
                           low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 1, 3],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out, check_names=False)

    def test_is_valid_scalar_column(self):
        """Test the ``Between.is_valid`` method with ``low`` as scalar and ``high`` as a column.

        Is expected to return whether the constraint ``column`` is between the
        ``low`` and ``high`` values.

        Input:
        - Table data where the last value is greater than ``high``. (pandas.DataFrame)
        Output:
        - True should be returned for the two first rows, False
          for the last one. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high, low_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0.5, 1, 0.6],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_column_scalar(self):
        """Test the ``Between.is_valid`` method with ``low`` as a column and ``high`` as scalar.

        Is expected to return whether the constraint ``column`` is between the
        ``low`` and ``high`` values.

        Input:
        - Table data where the second value is smaller than ``low`` and
          last value is greater than ``high``. (pandas.DataFrame)
        Output:
        - True should be returned for the first row, False
          for the last two. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 1.9],
            'b': [-0.5, 1, 0.6],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_column_column(self):
        """Test the ``Between.is_valid`` method with ``low`` and ``high`` as columns.

        Is expected to return whether the constraint ``column`` is between the
        ``low`` and ``high`` values.

        Input:
        - Table data where the last value is greater than ``high``. (pandas.DataFrame)
        Output:
        - True should be returned for the two first rows, False
          for the last one. (pandas.Series)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 'c'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 0.6]
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_low_high_nans(self):
        """Test the ``Between.is_valid`` method with nan values in low and high columns.

        If one of `low` or `high` is NaN, expect it to be ignored in the comparison.
        If both are NaN or the constraint column is NaN, return True.

        Input:
        - Table with a NaN row
        Output:
        - True should be returned for the NaN row.
        """
        # Setup
        instance = Between(column='a', low='b', high='c')

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9, 1.0],
            'b': [0, None, None, 0.4],
            'c': [0.5, None, 0.6, None]
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_column_nans(self):
        """Test the ``Between.is_valid`` method with nan values in constraint column.

        If the constraint column is Nan, expect that `is_valid` returns True.

        Input:
        - Table with a NaN row
        Output:
        - True should be returned for the NaN row.
        """
        # Setup
        instance = Between(column='a', low='b', high='c')

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, None],
            'b': [0, 0.1, 0.5],
            'c': [0.5, 1.5, 0.6]
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_high_scalar_low_nans(self):
        """Test the ``Between.is_valid`` method with ``high`` as scalar and ``low`` containing NaNs.

        The NaNs in ``low`` should be ignored.

        Input:
        - Table with a NaN row
        Output:
        - The NaN values should be ignored when making comparisons.
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 1.9],
            'b': [-0.5, None, None],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_low_high_nans_datetime(self):
        """Test the ``Between.is_valid`` method with nan values in low and high datetime columns.

        If one of `low` or `high` is NaN, expect it to be ignored in the comparison.
        If both are NaN or the constraint column is NaN, return True.

        Input:
        - Table with row NaN containing NaNs.
        Output:
        - True should be returned for the NaN row.
        """
        # Setup
        instance = Between(column='a', low='b', high='c')

        # Run
        table_data = pd.DataFrame({
            'a': [
                pd.to_datetime('2020-09-13'),
                pd.to_datetime('2020-08-12'),
                pd.to_datetime('2020-08-13'),
                pd.to_datetime('2020-08-14'),
            ],
            'b': [
                pd.to_datetime('2020-09-03'),
                None,
                None,
                pd.to_datetime('2020-10-03'),
            ],
            'c': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                None,
                None,
            ]
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_column_nans_datetime(self):
        """Test the ``Between.is_valid`` method with nan values in the constraint column.

        If there is a row containing NaNs, expect that `is_valid` returns True.

        Input:
        - Table with row NaN containing NaNs.
        Output:
        - True should be returned for the NaN row.
        """
        # Setup
        instance = Between(column='a', low='b', high='c')

        # Run
        table_data = pd.DataFrame({
            'a': [
                None,
                pd.to_datetime('2020-08-12'),
                pd.to_datetime('2020-08-13'),
            ],
            'b': [
                pd.to_datetime('2020-09-03'),
                pd.to_datetime('2020-08-02'),
                pd.to_datetime('2020-08-03'),
            ],
            'c': [
                pd.to_datetime('2020-10-03'),
                pd.to_datetime('2020-11-01'),
                pd.to_datetime('2020-11-01'),
            ]
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_high_datetime_low_nans(self):
        """Test the ``Between.is_valid`` method with ``high`` as datetime and ``low`` with NaNs.

        The NaNs in ``low`` should be ignored.

        Input:
        - Table with a NaN row
        Output:
        - The NaN values should be ignored when making comparisons.
        """
        # Setup
        column = 'a'
        low = 'b'
        high = pd.to_datetime('2020-08-13')
        instance = Between(column=column, low=low, high=high, high_is_scalar=True)

        # Run
        table_data = pd.DataFrame({
            'a': [
                pd.to_datetime('2020-08-12'),
                pd.to_datetime('2020-08-12'),
                pd.to_datetime('2020-08-14'),
            ],
            'b': [
                pd.to_datetime('2020-06-03'),
                None,
                None,
            ],
        })
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)


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
        instance = OneHotEncoding(columns=['a', 'b'])

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
        instance = OneHotEncoding(columns=['a', 'b', 'c'])

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

    def test__sample_constraint_columns_proper(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to return a table with the appropriate complementary column ``b``,
        since column ``a`` is entirely defined by the ``condition`` table.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where ``a`` is the same as in ``condition``
          and ``b`` is complementary`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
        })
        instance = OneHotEncoding(columns=['a', 'b'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0, 0.0, 0.0] * 5,
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1.0, 0.0, 0.0] * 5,
            'b': [0.0, 1.0, 1.0] * 5,
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_one_one(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Since the condition column contains a one for all rows, expected to assign
        all other columns to zeros.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where the first column contains one's and others columns zero's (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [0.0] * 10,
            'c': [0.0] * 10
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_two_ones(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since the condition contains two ones
        in a single row.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [1.0] * 10,
            'c': [0.0] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test__sample_constraint_columns_non_binary(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since the condition contains a non binary value.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.5] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test__sample_constraint_columns_all_zeros(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since the condition contains only zeros.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 0] * 5,
            'b': [0, 1] * 5,
            'c': [0, 0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0] * 10,
            'b': [0.0] * 10,
            'c': [0.0] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test__sample_constraint_columns_valid_condition(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to generate a table where every column satisfies the ``condition``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table satifying the ``condition`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0] * 10,
            'b': [1.0] * 10,
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'a': [0.0] * 10,
            'b': [1.0] * 10,
            'c': [0.0] * 10
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_one_zero(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Since the condition column contains only one zero, expected to randomly sample
        from unset columns any valid possibility. Since the ``b`` column in ``data``
        contains all the ones, it's expected to return a table where only ``b`` has ones.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where ``b`` is all one`s and other columns are all zero`s (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [0.0, 0.0] * 5,
            'b': [1.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_out = pd.DataFrame({
            'c': [0.0] * 10,
            'a': [0.0] * 10,
            'b': [1.0] * 10
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test__sample_constraint_columns_one_zero_alt(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Since the condition column contains only one zero, expected to randomly sample
        from unset columns any valid possibility.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table where ``c`` is all zero`s and ``b`` xor ``a`` is always one (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        assert (out['c'] == 0.0).all()
        assert ((out['b'] == 1.0) ^ (out['a'] == 1.0)).all()

    def test_sample_constraint_columns_list_of_conditions(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to generate a table satisfying the ``condition``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table satisfying the ``condition`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_output = pd.DataFrame({
            'a': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5,
            'b': [1.0, 0.0] * 5
        })
        pd.testing.assert_frame_equal(out, expected_output)

    def test_sample_constraint_columns_negative_values(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to raise a ``ValueError``, since condition is not a one hot vector.
        This tests that even if the sum of a row is one it still crashes.

        Input:
        - Table data (pandas.DataFrame)
        Raise:
        - ``ValueError``
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [-1.0] * 10,
            'c': [1.0] * 10
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [1.0] * 10,
            'b': [-1.0] * 10,
            'c': [1.0] * 10
        })

        # Assert
        with pytest.raises(ValueError):
            instance._sample_constraint_columns(condition)

    def test_sample_constraint_columns_all_zeros_but_one(self):
        """Test the ``OneHotEncoding._sample_constraint_columns`` method.

        Expected to generate a table where column ``a`` is filled with ones,
        and ``b`` and ``c`` filled with zeros.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table satisfying the ``condition`` (pandas.DataFrame)
        """
        # Setup
        data = pd.DataFrame({
            'a': [1.0, 0.0] * 5,
            'b': [0.0, 1.0] * 5,
            'c': [0.0, 0.0] * 5
        })
        instance = OneHotEncoding(columns=['a', 'b', 'c'])
        instance.fit(data)

        # Run
        condition = pd.DataFrame({
            'a': [0.0] * 10,
            'c': [0.0] * 10
        })
        out = instance._sample_constraint_columns(condition)

        # Assert
        expected_output = pd.DataFrame({
            'a': [0.0] * 10,
            'c': [0.0] * 10,
            'b': [1.0] * 10
        })
        pd.testing.assert_frame_equal(out, expected_output)


class TestUnique():

    def test___init__(self):
        """Test the ``Unique.__init__`` method.

        The ``columns`` should be set to those provided and the
        ``handling_strategy`` should be set to ``'reject_sampling'``.

        Input:
        - column names to keep unique.
        Output:
        - Instance with ``columns`` set and ``transform``
        and ``reverse_transform`` methods set to ``instance._identity``.
        """
        # Run
        instance = Unique(columns=['a', 'b'])

        # Assert
        assert instance.columns == ['a', 'b']
        assert instance.fit_columns_model is False
        assert instance.transform == instance._identity
        assert instance.reverse_transform == instance._identity

    def test___init__one_column(self):
        """Test the ``Unique.__init__`` method.

        The ``columns`` should be set to a list even if a string is
        provided.

        Input:
        - string that is the name of a column.
        Output:
        - Instance with ``columns`` set to list of one element.
        """
        # Run
        instance = Unique(columns='a')

        # Assert
        assert instance.columns == ['a']

    def test_is_valid(self):
        """Test the ``Unique.is_valid`` method.

        This method should return a pd.Series where the index
        of the first occurence of a unique combination of ``instance.columns``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple of the same combinations of columns.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(columns=['a', 'b', 'c'])

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
        of the first occurence of a unique combination of ``instance.columns``
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
        instance = Unique(columns=['a', 'b', 'c'])

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
        of the first occurence of a unique combination of ``instance.columns``
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
        instance = Unique(columns=['a', 'b', 'c'])

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
        of the first occurence of a unique value of ``self.columns``
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
        instance = Unique(columns='a')

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
        of the first occurence of a unique value of ``self.columns``
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
        instance = Unique(columns='a')

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
        of the first occurence of a unique value of ``self.columns``
        is set to ``True``, and every other occurence is set to ``False``.

        Input:
        - DataFrame with multiple occurences of the same value of the
        one column in ``instance.columns``.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(columns='a')

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
        of the first occurence of a unique value of ``instance.columns``
        is set to ``True``, and every other occurence is set to ``False``.
        ``None``, ``np.nan`` and ``float('nan')`` should be treated as the same category.

        Input:
        - DataFrame with some repeated values, some of which are nan's.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(columns=['a'])

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
        of the first occurence of a unique combination of ``instance.columns``
        is set to ``True``, and every other occurence is set to ``False``.
        ``None``, ``np.nan`` and ``float('nan')`` should be treated as the same category.

        Input:
        - DataFrame with multiple of the same combinations of columns, some of which are nan's.
        Output:
        - Series with the index of the first occurences set to ``True``.
        """
        # Setup
        instance = Unique(columns=['a', 'b'])

        # Run
        data = pd.DataFrame({
            'a': [1, None, 1, np.nan, float('nan'), 1],
            'b': [np.nan, 1, None, float('nan'), float('nan'), 1],
        })
        valid = instance.is_valid(data)

        # Assert
        expected = pd.Series([True, True, False, True, False, True])
        pd.testing.assert_series_equal(valid, expected)
