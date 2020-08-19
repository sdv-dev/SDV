"""Tests for the sdv.constraints.tabular module."""
import pandas as pd

from sdv.constraints.base import import_object
from sdv.constraints.tabular import (
    ColumnFormula, CustomConstraint, GreaterThan, UniqueCombinations)


class TestCustomConstraint():

    def test___init___transform(self):
        """Test the ``CustomConstraint.__init__`` method if a function to replace
        ``transform`` method is passed.

        The ``CustomConstraint.__init__`` method is expected to:
        - Import a function to replace ``transform`` method.
        - Create a Constraint instance.

        Setup:
        - Create a new function ``replace_transform`` to replace ``transform`` method.

        Input:
        - replace_transform.
        """
        # Setup
        def replace_transform(table_data):
            pass

        # Run
        instance = CustomConstraint(transform=replace_transform)

        # Assert
        assert instance.transform == import_object(replace_transform)
        assert instance.reverse_transform != import_object(replace_transform)
        assert instance.is_valid != import_object(replace_transform)

    def test___init___reverse_transform(self):
        """Test the ``CustomConstraint.__init__`` if a function to replace
        ``reverse_transform`` method is passed.

        The ``CustomConstraint.__init__`` method is expected to:
        - Import a function to replace ``reverse_transform`` method.
        - Create a Constraint instance.


        Setup:
        - Create a new function ``replace_reverse_transform`` to replace
        ``reverse_transform`` method.

        Input:
        - replace_reverse_transform.
        """
        # Setup
        def replace_reverse_transform(table_data):
            pass

        # Run
        instance = CustomConstraint(reverse_transform=replace_reverse_transform)

        # Assert
        assert instance.transform != import_object(replace_reverse_transform)
        assert instance.reverse_transform == import_object(replace_reverse_transform)
        assert instance.is_valid != import_object(replace_reverse_transform)

    def test___init___is_valid(self):
        """Test the ``CustomConstraint.__init__`` if a function to replace
        ``is_valid`` method is passed.

        The ``CustomConstraint.__init__`` method is expected to:
        - Import a function to replace ``is_valid`` method.
        - Create a Constraint instance.

        Setup:
        - Create a new function ``replace_is_valid`` to replace ``is_valid`` method.

        Input:
        - replace_is_valid.
        """
        # Setup
        def replace_is_valid(table_data):
            pass

        # Run
        instance = CustomConstraint(is_valid=replace_is_valid)

        # Assert
        assert instance.transform != import_object(replace_is_valid)
        assert instance.reverse_transform != import_object(replace_is_valid)
        assert instance.is_valid == import_object(replace_is_valid)

    def test___init___no_function(self):
        """Test the ``CustomConstraint.__init__`` if no function is passed.

        The ``CustomConstraint.__init__`` method is expected to:
        - Create a Constraint instance.

        Setup:
        - Create a new function ``do_nothing`` in order to run the asserts.
        """
        # Setup
        def do_nothing():
            pass

        # Run
        instance = CustomConstraint()

        # Assert
        assert instance.transform != import_object(do_nothing)
        assert instance.reverse_transform != import_object(do_nothing)
        assert instance.is_valid != import_object(do_nothing)


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
        - Return ``True`` if the separator is valid for the data.

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
        - Return ``False`` if the separator is non-valid for the data.

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
        instance._separator = 'd'

        # Assert
        assert instance._valid_separator(table_data) is False

    def test__valid_separator_false_joined(self):
        """Test the ``UniqueCombinations._valid_separator`` method for a non-valid separator.
        The column name obtained after joining the column names using the separator
        already exists.

        The ``UniqueCombinations._valid_separator`` method is expected to:
        - Return ``False`` if the separator is non-valid for the data.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - False (bool).
        """
        # Setup
        table_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
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
        - Find a separtor that works for the current data by iteratively
        adding `#` to it.
        - Generate the joint column name by concatenating
         the names of ``self._columns`` with the separator.

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
        - Transform the data by removing all the ``self._columns`` from
        the dataframe, concatenating them using the found separator, and
        setting them back to the data as a single name with the previously
        computed name.

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
        - Return the original data by popping the joint column from
        the table, splitting it by the previously found separator and
        then setting all the columns back to the table with the original
        names.

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
        out = instance.transform(transformed_data)

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
        - Transform the input data replacing the ``high`` value with difference
        between it and the ``low`` value. Then, a logarithm is applied to the difference + 1
        to be able to ensure that the value stays positive when reverted afterwards
        using an exponential.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data transformed (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        # Run
        instance = GreaterThan(low='a', high='b', strict=True)
        out = instance.transform(table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2, 3],
            'c': [7, 8, 9],
            'b': [1.386294, 1.386294, 1.386294]
        })

        pd.testing.assert_frame_equal(out, expected_out)

    def test_reverse_transform(self):
        """Test the ``GreaterThan.reverse_transform`` method.

        The ``GreaterThan.reverse_transform`` method is expected to:
        - Transform the input data replacing the ``high`` value with the original one.
        The transformation is reversed by computing an exponential of the given
        value, converting it to the original dtype, subtracting 1 and finally
        clipping the value to 0 on the low end to ensure the value is positive.
        Finally, the obtained value is added to the ``low`` column to get the final
        ``high`` value.

        Input:
        - Table data transformed (pandas.DataFrame)
        - Table data (pandas.DataFrame)
        Side effects:
        - Since ``reverse_transform`` uses the class variable ``_dtype``, so the ``fit`` method
        must be called.
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


class TestColumnFormula():

    def test___init__(self):
        """Test the ``ColumnFormula.__init__`` method.

        The ``ColumnFormula.__init__`` method is expected to:
        - Receive the name of the column to compute applying the formula,
        the function to use for the computation and the ``handling_strategy``.
        - Import a function to use for the computation.
        - Create a Constraint instance.

        Setup:
        - Create a simple function to use for the computation.

        Input:
        - column = 'c'
        - formula = new_column
        """
        # Setup
        column = 'c'

        def new_column(data):
            return data['a'] + data['b']

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

        def new_column(data):
            return data['a'] + data['b']

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
