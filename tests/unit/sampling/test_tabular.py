"""Tests for the sdv.sampling.tabular module."""
from sdv.sampling.tabular import Condition


class TestCondition():

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
