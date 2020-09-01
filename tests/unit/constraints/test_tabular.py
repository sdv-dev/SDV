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

        The ``UniqueCombinations.__init__`` method is expected to:
        - Receive the names of the columns that need to produce unique combinations.
        - Create a Constraint instance.

        Side effects:
        - instance._colums == columns
        """
        # Setup
        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)

        # Assert
        assert instance._columns == columns

    def test__valid_separator_true(self):
        """Test the ``UniqueCombinations._valid_separator`` method for a valid separator.

        The ``UniqueCombinations._valid_separator`` method is expected to:
        - Return ``True`` since, the separator is valid for the data.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - True (bool).
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })

        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)
        instance._separator = '#'

        # Assert
        assert instance._valid_separator(table_data) is True

    def test__valid_separator_false_separator_contained(self):
        """Test the ``UniqueCombinations._valid_separator`` method for a non-valid separator.
        The separator is contained within any of the columns.

        The ``UniqueCombinations._valid_separator`` method is expected to:
        - Return ``False``, since the separator is non-valid for the data.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - False (bool).
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })

        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)
        instance._separator = 'e'

        # Assert
        assert instance._valid_separator(table_data) is False

    def test__valid_separator_false_name_joined_exists(self):
        """Test the ``UniqueCombinations._valid_separator`` method for a non-valid separator.
        The column name obtained after joining the column names using the separator
        already exists.

        The ``UniqueCombinations._valid_separator`` method is expected to:
        - Return ``False``, since the separator is non-valid for the data.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - False (bool).
        """
        # Setup
        table_data = pd.DataFrame({
            'bec': ['a', 'b', 'c'],
            'b': ['d', 'ed', 'f'],
            'c': ['g', 'h', 'i']
        })

        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)
        instance._separator = 'e'

        # Assert
        assert instance._valid_separator(table_data) is False

    def test_fit(self):
        """Test the ``UniqueCombinations.fit`` method.

        The ``UniqueCombinations.fit`` method is expected to:
        - Call ``UniqueCombinations._valid_separator``.
        - Find a valid separator for the data and generate de joint column name.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i']
        })

        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)

        # Asserts
        expected_combinations = set(table_data[columns].itertuples(index=False))
        assert instance._separator == '#'
        assert instance._joint_column == 'b#c'
        assert instance._combinations == expected_combinations

    def test_is_valid_true(self):
        """Test the ``UniqueCombinations.is_valid`` method.

        The ``UniqueCombinations.is_valid`` method is expected to:
        - Return a pandas.Series, to say whether each row is valid.

        Input:
        - Table data (pandas.DataFrame)
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

        # Run
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false(self):
        """Test the ``UniqueCombinations.is_valid`` method.

        The ``UniqueCombinations.is_valid`` method is expected to:
        - Return a pandas.Series, to say whether each row is valid.

        Input:
        - Table data (pandas.DataFrame)
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

        incorrect_table = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['D', 'E', 'F'],
            'c': ['g', 'h', 'i']
        })

        columns = ['b', 'c']

        # Run
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)
        out = instance.is_valid(incorrect_table)

        # Assert
        expected_out = pd.Series([False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform(self):
        """Test the ``UniqueCombinations.transform`` method.

        The ``UniqueCombinations.transform`` method is expected to:
        - Return a Table data with the columns concatenated by the separator.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data transformed (pandas.DataFrame)
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

        # Run
        instance = UniqueCombinations(columns=columns)
        instance.fit(table_data)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b#c': ['d#g', 'e#h', 'f#i']
        })

        pd.testing.assert_frame_equal(expected_out, out)

    def reverse_transform(self):
        """Test the ``UniqueCombinations.reverse_transform`` method.

        The ``UniqueCombinations.reverse_transform`` method is expected to:
        - Return the original data separating the concatenated columns.

        Input:
        - Table data transformed (pandas.DataFrame)
        Output:
        - Table data (pandas.DataFrame)
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

        # Run
        instance = UniqueCombinations(columns=columns)
        instance.fit(transformed_data)
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

        The ``GreaterThan.__init__`` method is expected to:
        - Receive ``low`` and ``high``, names of the columns that containt the
        low and high value.
        - Create a Constraint instance.

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

        The ``GreaterThan.__init__`` method is expected to:
        - Receive ``low`` and ``high``, names of the columns that containt the
        low and high value. It also receives ``strict``, a bool that says that
        the comparison of the values should be strict.
        - Create a Constraint instance.

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

        The ``GreaterThan.fit`` method is expected to:
        - Learn the dtype of the ``high`` column.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - dtype of the ``high`` column.
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        # Run
        instance = GreaterThan(low='a', high='b')
        instance.fit(table_data)

        # Asserts
        expected = table_data['b'].dtype
        assert instance._dtype == expected

    def test_is_valid_true_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are valid
        and the comparison is strict.

        The ``GreaterThan.is_valid`` method is expected to:
        - Say whether ``high`` is greater than ``low`` in each row.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        # Run
        instance = GreaterThan(low='a', high='b', strict=True)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are not valid
        and the comparison is strict.

        The ``GreaterThan.is_valid`` method is expected to:
        - Say whether ``high`` is greater than ``low`` in each row.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``False`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 1, 1],
            'c': [7, 8, 9]
        })

        # Run
        instance = GreaterThan(low='a', high='b', strict=True)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_true_not_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are valid
        and the comparison is not strict.

        The ``GreaterThan.is_valid`` method is expected to:
        - Say whether ``high`` is equal or greater than ``low`` in each row.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 2, 3],
            'c': [7, 8, 9]
        })

        # Run
        instance = GreaterThan(low='a', high='b')
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false_not_strict(self):
        """Test the ``GreaterThan.is_valid`` method when the column values are valid
        and the comparison is not strict.

        The ``GreaterThan.is_valid`` method is expected to:
        - Say whether ``high`` is equal or greater than ``low`` in each row.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [0, 1, 2],
            'c': [7, 8, 9]
        })

        # Run
        instance = GreaterThan(low='a', high='b')
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
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })

        # Run
        instance = GreaterThan(low='a', high='b', strict=True)
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

        # Run
        instance = GreaterThan(low='a', high='b', strict=True)
        instance.fit(table_data)
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [55, 149, 405],
            'c': [7, 8, 9],
        })
        pd.testing.assert_frame_equal(out, expected_out)


def new_column(data):
    return data['a'] + data['b']


class TestColumnFormula():

    def test___init__(self):
        """Test the ``ColumnFormula.__init__`` method.

        The ``ColumnFormula.__init__`` method is expected to:
        - Import the formula to use for the computation.
        - Create a Constraint instance.

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

    def test_is_valid_true(self):
        """Test the ``ColumnFormula.is_valid`` method.

        The ``ColumnFormula.is_valid`` method is expected to:
        - Say whether each row fulfills the formula.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })

        column = 'c'

        # Run
        instance = ColumnFormula(column=column, formula=new_column)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_is_valid_false(self):
        """Test the ``ColumnFormula.is_valid`` method.

        The ``ColumnFormula.is_valid`` method is expected to:
        - Say whether each row fulfills the formula.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``False`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        })

        column = 'c'

        def new_column(data):
            return data['a'] + data['b']

        # Run
        instance = ColumnFormula(column=column, formula=new_column)
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([False, False, False])
        pd.testing.assert_series_equal(expected_out, out)

    def test_transform(self):
        """Test the ``ColumnFormula.transform`` method.

        The ``ColumnFormula.transform`` method is expected to:
        - Drop the indicated column from the table.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data without the indicated column (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })

        column = 'c'

        def new_column(data):
            return data['a'] + data['b']

        # Run
        instance = ColumnFormula(column=column, formula=new_column)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })

        pd.testing.assert_frame_equal(expected_out, out)

    def test_reverse_transform(self):
        """Test the ``ColumnFormula.reverse_transform`` method.

        The ``ColumnFormula.reverse_transform`` method is expected to:
        - Compute the indicated column by applying the given formula.

        Input:
        - Table data without the column with the correct values (pandas.DataFrame)
        Output:
        - Table data with the computed column (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 1, 1]
        })

        column = 'c'

        def new_column(data):
            return data['a'] + data['b']

        # Run
        instance = ColumnFormula(column=column, formula=new_column)
        out = instance.reverse_transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [5, 7, 9]
        })

        pd.testing.assert_frame_equal(expected_out, out)
