"""Tests for the sdv.constraints.tabular module."""
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
        expected_combinations = set(table_data[columns].itertuples(index=False))
        assert instance._separator == '#'
        assert instance._joint_column == 'b#c'
        assert instance._combinations == expected_combinations

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

        # Assert
        expected_out = pd.Series([True, True, True])
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
        expected_out = pd.Series([False, False, False])
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

        It is expected to create a new Constraint instance and receiving ``low`` and ``high``,
        names of the columns that contain the low and high value.

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

        It is expected to create a new Constraint instance and receiving ``low`` and ``high``,
        names of the columns that contain the low and high value. It also receives ``strict``,
        a bool that indicates the comparison of the values should be strict.

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

    def test_fit(self):
        """Test the ``GreaterThan.fit`` method.

        It is expected to return the dtype of the ``high`` column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - dtype of the ``high`` column.
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        instance.fit(table_data)

        # Asserts
        expected = table_data['b'].dtype
        assert instance._dtype == expected

    def test_is_valid_true_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are valid
        and the comparison is strict.

        If the columns satisfy the constraint, result is a series of ``True`` values.

        Input:
        - Table data, where the values of the ``low`` column are lower
        than the values of the ``high`` column (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are not valid
        and the comparison is strict.

        If the columns do not satisfy the costraint, result is a series of ``False`` values.

        Input:
        - Table data, where the values of the ``low`` column are higher or equal
        than the values of the ``high`` column (pandas.DataFrame)
        Output:
        - Series of ``False`` values (pandas.Series)
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 1, 1],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_true_not_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are valid
        and the comparison is not strict.

        If the columns satisfy the constraint, result is a series of ``True`` values.

        Input:
        - Table data, where the values of the ``low`` column are lower or equal
        than the values of the ``high`` column (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 3],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false_not_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are not valid
        and the comparison is not strict.

        If the columns do not satisfy the costraint, result is a series of ``False`` values.

        Input:
        - Table data, where the values of the ``low`` column are higher
        than the values of the ``high`` column (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        instance = GreaterThan(low='a', high='b')

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [0, 1, 2],
            'c': [7, 8, 9]
        })
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform(self):
        """Test the ``GreaterThan.transform`` method.

        The ``GreaterThan.transform`` method is expected to:
        - Transform the original table data.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data transformed (pandas.DataFrame)
        """
        # Setup
        instance = GreaterThan(low='a', high='b', strict=True)

        # Run
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.3862944, 1.3862944, 1.3862944]
        })
        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform(self):
        """Test the ``GreaterThan.reverse_transform`` method.

        The ``GreaterThan.reverse_transform`` method is expected to:
        - Return the original table data.

        Input:
        - Table data transformed (pandas.DataFrame)
        Output:
        - Table data (pandas.DataFrame)
        Side effects:
        - Since ``reverse_transform`` uses the class variable ``_dtype``, the ``fit`` method
        must be called as well.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        instance = GreaterThan(low='a', high='b', strict=True)
        instance.fit(table_data)

        # Run
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [55, 149, 405],
            'c': [7, 8, 9],
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
