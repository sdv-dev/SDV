"""Tests for the sdv.sampling.tabular module."""

import re
from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.sampling.tabular import Condition, DataFrameCondition, InterTableCondition


class TestCondition:
    def test___init__(self):
        """Test ```Condition.__init__`` method.

        Expect that `column_values` and `num_rows` are defined correctly.

        Input:
            - column_values
            - num_rows
        """
        # Setup
        column_values = {'a': 1, 'b': 2}
        num_rows = 5

        # Run
        condition = Condition(column_values=column_values, num_rows=num_rows)

        # Assert
        assert condition.column_values == column_values
        assert condition.num_rows == num_rows

    def test_get_column_values(self):
        """Test ```Condition.get_column_values`` method.

        Expect that the correct `column_values` value is returned.

        Input:
            - column_values
        """
        # Setup
        column_values = {'a': 1, 'b': 2}
        condition = Condition(column_values=column_values)

        # Run
        condition_column_values = condition.get_column_values()

        # Assert
        assert condition_column_values == column_values

    def test_get_num_rows_default(self):
        """Test ```Condition.get_num_rows`` method.

        Expect that the default `num_rows` value is returned, and
        that the default value is 1.

        Input:
            - column_values
        """
        # Setup
        column_values = {'a': 1, 'b': 2}
        condition = Condition(column_values=column_values)

        # Run
        default_num_rows = condition.get_num_rows()

        # Assert
        assert default_num_rows == 1

    def test_get_num_rows(self):
        """Test ```Condition.get_num_rows`` method.

        Expect that the correct `num_rows` value is returned.

        Input:
            - column_values
            - num_rows
        """
        # Setup
        column_values = {'a': 1, 'b': 2}
        num_rows = 100
        condition = Condition(column_values=column_values, num_rows=num_rows)

        # Run
        condition_num_rows = condition.get_num_rows()

        # Assert
        assert condition_num_rows == num_rows


class TestDataFrameCondition:
    def test___init__(self):
        """Test ```DataFrameCondition.__init__`` method."""
        # Setup
        dataframe = pd.DataFrame({'a': [1], 'b': [2]})

        # Run
        condition = DataFrameCondition(dataframe)

        # Assert
        pd.testing.assert_frame_equal(condition.dataframe, dataframe)
        assert not condition.table_name

    def test___init__raises_dataframe(self):
        """Test ```DataFrameCondition.__init__`` method raises error."""
        # Setup
        msg = '`dataframe` must be a pandas DataFrame object.'

        # Run
        with pytest.raises(ValueError, match=msg):
            DataFrameCondition({'a': 1})

    def test___init__raises_table_name(self):
        """Test ```DataFrameCondition.__init__`` method raises error."""
        # Setup
        msg = '`table_name` must be a string or None.'

        with pytest.raises(ValueError, match=msg):
            DataFrameCondition(pd.DataFrame({'a': [1]}), table_name=1)

    def test_get_table_name(self):
        """Test ```DataFrameCondition.get_table_name`` method."""
        # Setup
        dataframe = pd.DataFrame({'a': [1], 'b': [2]})
        table_name = 'users'

        # Run
        condition = DataFrameCondition(dataframe, table_name)

        # Assert
        assert condition.get_table_name() == table_name

    def test_get_dataframe(self):
        """Test ```DataFrameCondition.get_dataframe`` method."""
        # Setup
        dataframe = pd.DataFrame({'a': [1], 'b': [2]})

        # Run
        condition = DataFrameCondition(dataframe)

        # Assert
        pd.testing.assert_frame_equal(condition.get_dataframe(), dataframe)

    def test_get_num_rows(self):
        # Setup
        dataframe = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})
        condition = DataFrameCondition(dataframe)

        # Run
        num_rows = condition.get_num_rows()

        # Assert
        assert num_rows == 2


class TestInterTableCondition:
    def test__validate_conditions_invalid_conditions(self):
        """Test the ``_validate_conditions`` method errors with invalid conditions."""
        # Setup
        instance = Mock()
        condition_no_table = Condition({'a': 0})
        bad_condition = {'a': 0}

        # Run and Assert
        expected_msg = re.escape('Invalid Condition. Condition does not have a table name set.')
        with pytest.raises(ValueError, match=expected_msg):
            InterTableCondition._validate_conditions(instance, [condition_no_table])

        expected_msg = re.escape(
            "Invalid condition ({'a': 0}). Conditions must be a `Condition`"
            'or `DataFrameCondition` object.'
        )
        with pytest.raises(ValueError, match=expected_msg):
            InterTableCondition._validate_conditions(instance, [bad_condition])

    def test__validate_conditions_num_rows_not_set(self):
        """Test the ``_validate_condtions`` errors if num_rows not set."""
        # Setup
        instance = Mock()
        condition_1 = Condition({'a': 0}, num_rows=None, table_name='table')
        condition_2 = Condition({'b': 1}, num_rows=None, table_name='parent')
        condition_3 = Condition({'a': 2}, num_rows=None, table_name='table')

        # Run and Assert
        expected_msg = re.escape('At least one condition must have `num_rows` set.')
        with pytest.raises(ValueError, match=expected_msg):
            InterTableCondition._validate_conditions(instance, [condition_1, condition_2])

        expected_msg = re.escape(
            "Multiple conditions found for table 'table'. If multiple conditions "
            'are supplied for a table, all conditions for that table must have `num_rows` set.'
        )
        with pytest.raises(ValueError, match=expected_msg):
            InterTableCondition._validate_conditions(instance, [condition_1, condition_3])

    def test__validate_conditions(self):
        """Test ``_validate_conditions`` returns a map of conditions to table names."""
        instance = Mock()
        condition_1 = Condition({'a': 0}, num_rows=5, table_name='table')
        condition_2 = Condition({'b': 1}, num_rows=3, table_name='parent')
        condition_3 = Condition({'a': 2}, num_rows=10, table_name='table')
        condition_4 = DataFrameCondition(pd.DataFrame({'x': ['a', 'b', 'c']}), table_name='child')

        conditions = [condition_1, condition_2, condition_3, condition_4]

        # Run
        condition_map = InterTableCondition._validate_conditions(instance, conditions)

        # Assert
        expected_condition_map = {
            'table': [condition_1, condition_3],
            'parent': [condition_2],
            'child': [condition_4],
        }
        assert condition_map == expected_condition_map

    def test___init__(self):
        """Test the ``__init__`` method."""
        instance = Mock()
        condition_1 = Condition({'a': 0}, num_rows=5, table_name='table')
        condition_2 = Condition({'b': 1}, num_rows=3, table_name='parent')
        condition_3 = Condition({'a': 2}, num_rows=10, table_name='table')
        condition_4 = DataFrameCondition(pd.DataFrame({'x': ['a', 'b', 'c']}), table_name='child')
        conditions = [condition_1, condition_2, condition_3, condition_4]

        # Run
        InterTableCondition.__init__(instance, conditions)

        # Assert
        instance._validate_conditions.assert_called_once_with(conditions)
        assert instance.conditions == conditions
        assert instance.table_conditions == instance._validate_conditions.return_value

    def test_get_conditions(self):
        """Test the ``get_table_conditions`` returns the conditions for each table."""
        instance = Mock()
        condition_1 = Condition({'a': 0}, num_rows=5, table_name='table')
        condition_2 = Condition({'b': 1}, num_rows=3, table_name='parent')
        condition_3 = Condition({'a': 2}, num_rows=10, table_name='table')
        condition_4 = DataFrameCondition(pd.DataFrame({'x': ['a', 'b', 'c']}), table_name='child')
        instance.conditions = [condition_1, condition_2, condition_3, condition_4]

        # Run
        conditions = InterTableCondition.get_conditions(instance)

        # Assert
        assert conditions == [condition_1, condition_2, condition_3, condition_4]

    def test_get_table_conditions(self):
        """Test the ``get_table_conditions`` returns the conditions for each table."""
        instance = Mock()
        condition_1 = Condition({'a': 0}, num_rows=5, table_name='table')
        condition_2 = Condition({'b': 1}, num_rows=3, table_name='parent')
        condition_3 = Condition({'a': 2}, num_rows=10, table_name='table')
        condition_4 = DataFrameCondition(pd.DataFrame({'x': ['a', 'b', 'c']}), table_name='child')
        instance.table_conditions = {
            'table': [condition_1, condition_3],
            'parent': [condition_2],
            'child': [condition_4],
        }

        # Run
        table_conditions = InterTableCondition.get_table_conditions(instance)

        # Assert
        assert table_conditions == {
            'table': [condition_1, condition_3],
            'parent': [condition_2],
            'child': [condition_4],
        }
