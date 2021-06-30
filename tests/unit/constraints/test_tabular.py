"""Tests for the sdv.constraints.tabular module."""

import numpy as np
from numpy.core.defchararray import translate
import pandas as pd
import pytest
from random import random

from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.tabular import (
    Between, ColumnFormula, CustomConstraint, GreaterThan, UniqueCombinations)


def dummy_transform():
    pass


def dummy_reverse_transform():
    pass


def dummy_is_valid():
    pass


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
        is_valid_fqn = __name__ + '.dummy_is_valid'

        # Run
        instance = CustomConstraint(
            transform=dummy_transform,
            reverse_transform=dummy_reverse_transform,
            is_valid=is_valid_fqn
        )

        # Assert
        assert instance.transform == dummy_transform
        assert instance.reverse_transform == dummy_reverse_transform
        assert instance.is_valid == dummy_is_valid


class TestUniqueCombinations():

    def test___init__(self):
        """Test the ``UniqueCombinations.__init__`` method.

        It is expected to create a new Constraint instance and receiving the names of
        the columns that need to produce unique combinations.

        Side effects:
        - instance._colums == columns
        """
        # Setup
        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)

        # Assert
        assert instance._columns == columns

    def test__valid_separator_valid(self):
        """Test ``_valid_separator`` for a valid separator.

        If the separator and data are valid, result is ``True``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - True (bool).
        """
        # Setup
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance._separator = '#'

        # Run
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        is_valid = instance._valid_separator(table_data, instance._separator, columns)

        # Assert
        assert is_valid

    def test__valid_separator_non_valid_separator_contained(self):
        """Test ``_valid_separator`` passing a column that contains the separator.

        If any of the columns contains the separator string, result is ``False``.

        Input:
        - Table data (pandas.DataFrame) with a column that contains the separator string ('#')
        Output:
        - False (bool).
        """
        # Setup
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance._separator = '#'

        # Run
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', '#', 'f'],
            'c': ['g', 'h', 'i']
        })
        is_valid = instance._valid_separator(table_data, instance._separator, columns)

        # Assert
        assert not is_valid

    def test__valid_separator_non_valid_name_joined_exists(self):
        """Test ``_valid_separator`` passing a column whose name is obtained after joining
        the column names using the separator.

        If the column name obtained after joining the column names using the separator
        already exists, result is ``False``.

        Input:
        - Table data (pandas.DataFrame) with a column name that will be obtained by joining
        the column names and the separator.
        Output:
        - False (bool).
        """
        # Setup
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance._separator = '#'

        # Run
        table_data = pd.DataFrame({
            'b#c': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        is_valid = instance._valid_separator(table_data, instance._separator, columns)

        # Assert
        assert not is_valid

    def test_fit(self):
        """Test the ``UniqueCombinations.fit`` method.

        The ``UniqueCombinations.fit`` method is expected to:
        - Call ``UniqueCombinations._valid_separator``.
        - Find a valid separator for the data and generate the joint column name.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)

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
        """Test the ``UniqueCombinations.is_valid`` method.

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
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        out = instance.is_valid(table_data)

        expected_out = pd.Series([True, True, True], name='b#c')
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false(self):
        """Test the ``UniqueCombinations.is_valid`` method.

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
        instance = UniqueCombinations(columns=columns)
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

    def test_transform(self):
        """Test the ``UniqueCombinations.transform`` method.

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
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Run
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b#c': ['d#g', 'e#h', 'f#i']
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_transform_not_all_columns_provided(self):
        """Test the ``UniqueCombinations.transform`` method.

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
        instance = UniqueCombinations(columns=columns, fit_columns_model=False)
        instance.fit(table_data)

        # Run/Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame({'a': ['a', 'b', 'c']}))

    def reverse_transform(self):
        """Test the ``UniqueCombinations.reverse_transform`` method.

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
        transformed_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b#c': ['d#g', 'e#h', 'f#i']
        })
        columns = ['b', 'c']
        instance = UniqueCombinations(columns=columns)
        instance.fit(transformed_data)

        # Run
        out = instance.reverse_transform(transformed_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })
        pd.testing.assert_frame_equal(expected_out, out)


class TestGreaterThan():

    def test___init___strict_false(self):
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
        assert instance._low == 'a'
        assert instance._high == 'b'
        assert instance._strict is False

    def test___init___strict_true(self):
        """Test the ``GreaterThan.__init__`` method.

        The passed arguments should be stored as attributes.

        Input:
        - low = 'a'
        - high = 'b'
        - strict = True
        Side effects:
        - instance._low == 'a'
        - instance._high == 'b'
        - instance._stric == True
        """
        # Run
        instance = GreaterThan(low='a', high='b', strict=True)

        # Asserts
        assert instance._low == 'a'
        assert instance._high == 'b'
        assert instance._strict is True

    def test_fit_int(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute.

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
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'i'

    def test_fit_float(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute.

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
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'f'

    def test_fit_datetime(self):
        """Test the ``GreaterThan.fit`` method.

        The ``GreaterThan.fit`` method should only learn and store the
        ``dtype`` of the ``high`` column as the ``_dtype`` attribute.

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
        instance.fit(table_data)

        # Asserts
        assert instance._dtype.kind == 'M'

    def test_is_valid_strict_false(self):
        """Test the ``GreaterThan.is_valid`` method with strict False.

        If strict is False, equal values should count as valid

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
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)

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
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform_int_drop_none(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type int.

        The ``GreaterThan.transform`` method is expected to compute the distance
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

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_int_drop_high(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type int.

        The ``GreaterThan.transform`` method is expected to compute the distance
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

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_int_drop_low(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type int.

        The ``GreaterThan.transform`` method is expected to compute the distance
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

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_float_drop_none(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type float.

        The ``GreaterThan.transform`` method is expected to compute the distance
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

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4., 5., 6.],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_datetime_drop_none(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type datetime.

        If the columns are of type datetime, ``transform`` is expected
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

        # Run
        table_data = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            '#a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
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
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
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
        instance._dtype = np.dtype('float')

        # Run
        transformed = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
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
        instance._dtype = np.dtype('<M8[ns]')

        # Run
        transformed = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'c': [1, 2],
            '#a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
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
        - Same table with the low column replaced by the low one + 3, as int
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS

        # Run
        transformed = pd.DataFrame({
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
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
        - Same table with the low column replaced by the low one + one second
        and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True, drop='low')
        instance._dtype = np.dtype('<M8[ns]')

        # Run
        transformed = pd.DataFrame({
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2],
            '#a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
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
        - Same table with the low column replaced by the low one + 3 for all
        invalid rows, as int and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 1, 6],
            'c': [7, 8, 9],
            '#a#b': [np.log(4)] * 3,
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
        - Same table with the low column replaced by the low one + one second
        for all invalid rows, and the diff column dropped.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = np.dtype('<M8[ns]')

        # Run
        transformed = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-01T00:00:01']),
            'c': [1, 2],
            '#a#b': [np.log(1_000_000_001), np.log(1_000_000_001)],
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': pd.to_datetime(['2020-01-01T00:00:01', '2020-01-02T00:00:01']),
            'c': [1, 2]
        })
        pd.testing.assert_frame_equal(out, expected_out)


def new_column(data):
    """Formula to be used for the ``TestColumnFormula`` class."""
    return data['a'] + data['b']


class TestColumnFormula():

    def test___init__(self):
        """Test the ``ColumnFormula.__init__`` method.

        It is expected to create a new Constraint instance
        and import the formula to use for the computation.

        Input:
        - column = 'c'
        - formula = new_column
        """
        # Setup
        column = 'c'

        # Run
        instance = ColumnFormula(column=column, formula=new_column)

        # Assert
        assert instance._column == column
        assert instance._formula == new_column

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

    def test_transform(self):
        """Test the ``ColumnFormula.transform`` method.

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
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
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


class TestBetween():

    def transform(self, data, low, high):
        data = (data - low)/(high - low) * 0.95 + 0.025
        return np.log(data/(1.0 - data))

    def test_transform_scalar_scalar(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as scalars.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [4, 5, 6],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [4, 5, 6],
            '#a#0.0#1.0': self.transform(table_data[column], low, high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_transform_scalar_column(self):
        """Test the ``Between.transform`` method by passing ``low`` as scalar and ``high`` as column.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0.5, 1, 6],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0.5, 1, 6],
            '#a#0.0#b': self.transform(table_data[column], low, table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)


    def test_transform_column_scalar(self):
        """Test the ``Between.transform`` method by passing ``low`` as column and ``high`` as scalar.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0, -1, 0.5],
            '#a#b#1.0': self.transform(table_data[column], table_data[low], high)
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_transform_column_column(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as columns.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
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
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            '#a#b#c': self.transform(table_data[column], table_data[low], table_data[high])
        })
        pd.testing.assert_frame_equal(expected_out, out)


    def test_transform(self):
        data = pd.DataFrame({
            'a': [1.0,1.0,1.0],
            'b': [1.0, 2.0, 3.0],
        })

        constraint = Between('b', 0.0, 4.0)
        constraint.fit(data)

        out = constraint.reverse_transform(constraint.transform(data))
        pd.testing.assert_frame_equal(data, out)


    def test_transform_2(self):
        data = pd.DataFrame({
            'low': [1.0,4.0,0.0],
            'high': [3.0, 6.0, 4.0],
            'med': [2.0, 5.0, 1.0],
        })

        constraint = Between('med', 'low', 'high')
        constraint.fit(data)

        out = constraint.reverse_transform(constraint.transform(data))
        pd.testing.assert_frame_equal(data, out)

    #TODO: test that strict is passed correctly?

    def test_reverse_transform_scalar_scalar(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as scalars.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [4, 5, 6],
        })

        transformed = pd.DataFrame({
            'b': [4, 5, 6],
            '#a#0.0#1.0': self.transform(table_data[column], low, high)
        })

        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_scalar_column(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as scalars.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0.5, 1, 6],
        })

        transformed = pd.DataFrame({
            'b': [0.5, 1, 6],
            '#a#0.0#b': self.transform(table_data[column], low, table_data[high])
        })

        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_column_scalar(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as scalars.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0, -1, 0.5],
        })

        transformed = pd.DataFrame({
            'b': [0, -1, 0.5],
            '#a#b#1.0': self.transform(table_data[column], table_data[low], high)
        })

        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform_column_column(self):
        """Test the ``Between.transform`` method by passing ``low`` and ``high`` as scalars.

        It is expected to create a new column similar to the ``constraint_column``, and then
        scale and apply a logit function to that column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data with an extra column (pandas.DataFrame)
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

        transformed = pd.DataFrame({
            'b': [0, -1, 0.5],
            'c': [0.5, 1, 6],
            '#a#b#c': self.transform(table_data[column], table_data[low], table_data[high])
        })

        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = table_data
        pd.testing.assert_frame_equal(expected_out, out)
    
    def test_reverse_transform_valid(self):
        """Test that regardless of the values, it always returns in the range."""
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high)

        # Run
        transformed = pd.DataFrame({
            '#a#0.0#1.0': [random() for i in range(1000)]
        })
        out = instance.reverse_transform(transformed)

        # Assert
        assert out['a'].between(0.0, 1.0).all()
    
    def test_is_valid_strict_true(self):
        """Test the ``Between.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - True should be returned for the strictly valid row and False
          for the other two.
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, strict=True)


        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 1, 3],
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out, check_names=False)

    def test_is_valid_strict_false(self):
        """Test the ``Between.is_valid`` method with strict False.

        If strict is False, equal values should count as valid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - True should be returned for the strictly valid row and False
          for the other two.
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 1.0
        instance = Between(column=column, low=low, high=high, strict=False)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 1, 3],
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        print(expected_out)
        print(out)
        pd.testing.assert_series_equal(expected_out, out, check_names=False)
    
    def test_is_valid_scalar_column(self):
        """Test the ``Between.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - True should be returned for the strictly valid row and False
          for the other two.
        """
        # Setup
        column = 'a'
        low = 0.0
        high = 'b'
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 0.9],
            'b': [0.5, 1, 0.6],
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])
        pd.testing.assert_series_equal(expected_out, out)
    
    def test_is_valid_column_scalar(self):
        """Test the ``Between.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - True should be returned for the strictly valid row and False
          for the other two.
        """
        # Setup
        column = 'a'
        low = 'b'
        high = 1.0
        instance = Between(column=column, low=low, high=high)

        # Run
        table_data = pd.DataFrame({
            'a': [0.1, 0.5, 1.9],
            'b': [-0.5, 1, 0.6],
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, False, False])
        pd.testing.assert_series_equal(expected_out, out)
    
    def test_is_valid_column_column(self):
        """Test the ``Between.is_valid`` method with strict True.

        If strict is True, equal values should count as invalid.

        Input:
        - Table with a strictly valid row, a strictly invalid row and
          a row that has the same value for both high and low.
        Output:
        - True should be returned for the strictly valid row and False
          for the other two.
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
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, False])

        pd.testing.assert_series_equal(expected_out, out)


# Do we care about the order of the columns?
# Add drop back