from unittest.mock import Mock

import pandas as pd
import pytest

import sdv
from sdv.constraints.base import Constraint, _get_qualified_name, import_object


def test__get_qualified_name_class():
    """Checks the returned Fully Qualified Name, if object is a class."""
    # Run
    FullyQualifiedName = _get_qualified_name(Constraint)

    # Assert
    expectedName = 'sdv.constraints.base.Constraint'
    assert FullyQualifiedName == expectedName


def test__get_qualified_name_function():
    """Checks the returned Fully Qualified Name, if object is a function."""
    # Run
    fully_qualified_name = _get_qualified_name(Constraint.reverse_transform)

    # Assert
    expected_name = 'sdv.constraints.base.reverse_transform'
    assert fully_qualified_name == expected_name


def test_import_object():
    """Checks the object imported from its qualified name."""
    # Run
    object = import_object('sdv.constraints.base.Constraint')

    # Assert
    assert object is Constraint


class TestConstraint():

    def test__init___transform(self):
        """Create default instance."""
        # Run
        instance = Constraint(handling_strategy='transform')

        # Assert
        assert instance.filter_valid == instance._identity

    def test__init___reject_sampling(self):
        """Create default instance."""
        # Run
        instance = Constraint(handling_strategy='reject_sampling')

        # Asserts
        assert instance.transform == instance._identity
        assert instance.reverse_transform == instance._identity

    def test__init___not_kown(self):
        """Create default instance."""
        # Run
        with pytest.raises(ValueError):
            Constraint(handling_strategy='not_known')

    def test_fit(self):
        """Checks if method `fit` is correctly called."""
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        instance.fit(table_data)

    def test_transform(self):
        """Checks if method `transform` performs correctly."""
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
        """Checks if methods `fit` and `transform` are correctly called."""
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
        """Checks if method `transform` performs correctly."""
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
        """Checks if the resulting series of "True" values is corect."""
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
        """Checks if the resulting valid rows are correct.
        It also checks if method `is_valid` is correctly called."""
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
        """Checks if the resulting valid rows are correct if no column satisfy the constraint.
        It also checks if method `is valid` is correctly called."""
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

    def test__get_subclasses(self):
        """Checks the resulting dict with the subclasses for the current class object."""
        # Run
        subclasses = Constraint._get_subclasses()

        # Assert
        expected_subclasses = {
            'CustomConstraint': sdv.constraints.tabular.CustomConstraint,
            'UniqueCombinations': sdv.constraints.tabular.UniqueCombinations,
            'GreaterThan': sdv.constraints.tabular.GreaterThan,
            'ColumnFormula': sdv.constraints.tabular.ColumnFormula
        }

        assert subclasses == expected_subclasses

    def test_from_dict(self):
        """Checks the Constraint object built from a dict."""
        # Setup
        constraint_dict = {
            'constraint': 'sdv.constraints.base.Constraint',
            'handling_strategy': 'transform'
        }

        # Run
        instance_from_dict = Constraint.from_dict(constraint_dict)

        # Assert
        assert instance_from_dict.filter_valid == instance_from_dict._identity

    def test_to_dict(self):
        """Checks the returning dict representation of the Constraint."""
        # Run
        instance = Constraint(handling_strategy='transform')
        dict = instance.to_dict()

        # Assert
        expected_dict = {
            'constraint': 'sdv.constraints.base.Constraint',
            'handling_strategy': 'transform'
        }

        assert dict == expected_dict
