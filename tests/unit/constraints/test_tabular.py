"""Tests for the sdv.constraints.tabular module."""

import numpy as np
import pandas as pd

from sdv.constraints.tabular import (
    ColumnFormula, CustomConstraint, GreaterThan, UniqueCombinations)


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
        is_valid = instance._valid_separator(table_data)

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
        is_valid = instance._valid_separator(table_data)

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
        is_valid = instance._valid_separator(table_data)

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

    def test_transform_int(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type int.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high and low columns and replace the high column with the
        logarithm of the distance + 1.

        Input:
        - Table with two columns two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with the high column transformed into the logarithms
          of the distances + 1, which is np.log(4).
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
            'b': [np.log(4)] * 3,
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_float(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type float.

        The ``GreaterThan.transform`` method is expected to compute the distance
        between the high and low columns and replace the high column with the
        logarithm of the distance + 1.

        Input:
        - Table with two constrained columns at a constant distance of
          exactly 3 and one additional dummy column.
        Output:
        - Same table with the high column transformed into the logarithms
          of the distances + 1, which is np.log(4).
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
            'b': [np.log(4)] * 3,
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_transform_datetime(self):
        """Test the ``GreaterThan.transform`` method passing a high column of type datetime.

        If the columns are of type datetime, ``transform`` is expected
        to convert the timedelta distance into numeric before applying
        the +1 and logarithm.

        Input:
        - Table with values at a distance of exactly 1 second.
        Output:
        - Same table with the high column transformed into the logarithms
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
            'b': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'c': [1, 2],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_int(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype int.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to integers

        Input:
        - Table with a high column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as int.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = pd.Series([1]).dtype    # exact dtype (32 or 64) depends on OS

        # Run
        transformed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [np.log(4)] * 3,
            'c': [7, 8, 9]
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_float(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype float.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - add the low column
            - convert the output to float values

        Input:
        - Table with a high column that contains the constant np.log(4).
        Output:
        - Same table with the high column replaced by the low one + 3, as float values.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = np.dtype('float')

        # Run
        transformed = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'b': [np.log(4)] * 3,
            'c': [7, 8, 9]
        })
        out = instance.reverse_transform(transformed)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1.1, 2.2, 3.3],
            'b': [4.1, 5.2, 6.3],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform_datetime(self):
        """Test the ``GreaterThan.reverse_transform`` method for dtype datetime.

        The ``GreaterThan.reverse_transform`` method is expected to:
            - apply an exponential to the input
            - subtract 1
            - convert the distance to a timedelta
            - add the low column
            - convert the output to datetimes

        Input:
        - Table with a high column that contains the constant np.log(1_000_000_001).
        Output:
        - Same table with the high column replaced by the low one + one second.
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)
        instance._dtype = np.dtype('<M8[ns]')

        # Run
        transformed = pd.DataFrame({
            'a': pd.to_datetime(['2020-01-01T00:00:00', '2020-01-02T00:00:00']),
            'b': [np.log(1_000_000_001), np.log(1_000_000_001)],
            'c': [1, 2]
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
