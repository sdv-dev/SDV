from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.constraints.base import Constraint


def test__get_qualified_name_class():
    """Checks if the returned Fully Qualified Name is the expected,
    if object has the given named attribute."""


def test__get_qualified_name_no_function():
    """Checks if the returned Fully Qualified Name is the expected,
    if object has not the given named attribute."""


def test_import_object():
    """Checks if the object imported from its qualified name is the expected."""


class TestConstraint():

    @pytest.mark.xfail
    def test__init___transform(self):
        """Create default instance."""
        instance = Constraint(handling_strategy='transform')
        assert (instance.transform is instance._identity)

    @pytest.mark.xfail
    def test__init___reject_sampling(self):
        """Create default instance."""
        instance = Constraint(handling_strategy='reject_sampling')
        assert (instance.transform is instance._identity)

    def test__init___not_kown(self):
        """Create default instance."""

    def test_fit(self):
        """It checks if method `fit` is correctly called."""

        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        instance.fit(table_data)

    def test_transform(self):
        """It calls `transform` and checks if the return is identycal
        to the table_data passed."""

        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        new_table = instance.transform(table_data)

        # Assert
        assert (table_data == new_table).all

    def test_fit_transform(self):
        """With Mock, it makes sure methods `fit` and `transform`
        are correctly called."""

        # Setup
        constraint_mock = Mock()
        constraint_mock.transform.return_value = 'the_transformed_data'

        # Run
        data = 'my_data'
        output = Constraint.fit_transform(constraint_mock, data)

        # Assert
        assert output == 'the_transformed_data'

        constraint_mock.fit.assert_called_once_with('my_data')
        constraint_mock.transform.assert_called_once_with('my_data')

    def test_reverse_transform(self):
        """It calls `reverse_transform` and checks if the return is identycal
        to the table_data passed."""

        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        new_table = instance.reverse_transform(table_data)

        # Assert
        assert (table_data == new_table).all

    def test_is_valid(self):
        """Given a table data, it checks if the resulting series of
        "True" values is corect."""

        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        result = instance.is_valid(table_data)

        # Assert
        expected = pd.Series([True, True, True])
        assert (result == expected).all

    def test_filter_valid(self):
        """Given a table data, it checks if the resulting
        valid rows are corect. Mith Mock, it makes sure method `is valid` is
        called correctly."""

        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        constraint_mock = Mock()
        constraint_mock.is_valid.return_value = pd.Series([True, True, False])

        # Run
        result = Constraint.filter_valid(constraint_mock, table_data)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2]
        })
        assert (result == expected).all

    def test_filter_valid_zero_rows(self):
        """Given a table data and a constraint, it checks if the resulting
        valid rows are corect if no column satisfy the constraint.
        Mith Mock, it makes sure method `is valid` is called correctly."""

        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        constraint_mock = Mock()
        constraint_mock.is_valid.return_value = pd.Series([False, False, False])

        # Run
        result = Constraint.filter_valid(constraint_mock, table_data)

        # Assert
        expected = pd.DataFrame({
            'a': []
        })

        assert (result == expected).all

    def test_to_dict(self):
        """Checks if the returning dict is correct."""

    def test_from_dict(self):
        """ """

    def test__get_subclasses(self):
        """ """
