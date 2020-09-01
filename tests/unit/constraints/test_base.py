"""Tests for the sdv.constraints.base module."""
from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.constraints.base import Constraint, _get_qualified_name, get_subclasses, import_object
from sdv.constraints.tabular import UniqueCombinations


def test__get_qualified_name_class():
    """Test the ``_get_qualified_name`` function, if a class is passed.

    The ``_get_qualified_name`` function is expected to:
    - Return the Fully Qualified Name from a class.

    Input:
    - A class.
    Output:
    - The class qualified name.
    """
    # Run
    fully_qualified_name = _get_qualified_name(Constraint)

    # Assert
    expected_name = 'sdv.constraints.base.Constraint'
    assert fully_qualified_name == expected_name


def test__get_qualified_name_function():
    """Test the ``_get_qualified_name`` function, if a function is passed.

    The ``_get_qualified_name`` function is expected to:
    - Return the Fully Qualified Name from a function.

    Input:
    - A function.
    Output:
    - The function qualified name.
    """
    # Run
    fully_qualified_name = _get_qualified_name(_get_qualified_name)

    # Assert
    expected_name = 'sdv.constraints.base._get_qualified_name'
    assert fully_qualified_name == expected_name


def test_get_subclasses():
    """Test the ``get_subclasses`` function.

    The ``get_subclasses`` function is expected to:
    - Recursively find subclasses for the class object passed.

    Setup:
    - Create three classes, Parent, Child and GrandChild,
      which inherit of each other hierarchically.

    Input:
    - The Parent class.
    Output:
    - Dict of the subclasses of the class: ``Child`` and ``GrandChild`` classes.
    """
    # Setup
    class Parent:
        pass

    class Child(Parent):
        pass

    class GrandChild(Child):
        pass

    # Run
    subclasses = get_subclasses(Parent)

    # Assert
    expected_subclasses = {
        'Child': Child,
        'GrandChild': GrandChild
    }

    assert subclasses == expected_subclasses


def test_import_object_class():
    """Test the ``import_object`` function, when importing a class.

    The ``import_object`` function is expected to:
    - Import a class from its qualifed name.

    Input:
    - Qualified name of the class.
    Output:
    - The imported class.
    """
    # Run
    obj = import_object('sdv.constraints.base.Constraint')

    # Assert
    assert obj is Constraint


def test_import_object_function():
    """Test the ``import_object`` function, when importing a function.

    The ``import_object`` function is expected to:
    - Import a function from its qualifed name.

    Input:
    - Qualified name of the function.
    Output:
    - The imported function.
    """
    # Run
    imported = import_object('sdv.constraints.base.import_object')

    # Assert
    assert imported is import_object


class TestConstraint():

    def test__identity(self):
        """Test ```Constraint._identity`` method.

        ``_identity`` method should return whatever it is passed.

        Input:
            - anything
        Output:
            - Input
        """
        # Run
        instance = Constraint('all')
        output = instance._identity('input')

        # Asserts
        assert output == 'input'

    def test___init___transform(self):
        """Test ```Constraint.__init__`` method when 'transform' is passed.

        If 'transform' is given, the ``__init__`` method should replace the ``is_valid`` method
        with an identity and leave ``transform`` and ``reverse_transform`` untouched.

        Input:
            - transform
        Side effects:
            - is_valid == identity
            - transform != identity
            - reverse_transform != identity
        """
        # Run
        instance = Constraint(handling_strategy='transform')

        # Asserts
        assert instance.filter_valid == instance._identity
        assert instance.transform != instance._identity
        assert instance.reverse_transform != instance._identity

    def test___init___reject_sampling(self):
        """Test ``Constraint.__init__`` method when 'reject_sampling' is passed.

        If 'reject_sampling' is given, the ``__init__`` method should replace the ``transform``
        and ``reverse_transform`` methods with an identity and leave ``is_valid`` untouched.

        Input:
            - reject_sampling
        Side effects:
            - is_valid != identity
            - transform == identity
            - reverse_transform == identity
        """
        # Run
        instance = Constraint(handling_strategy='reject_sampling')

        # Asserts
        assert instance.filter_valid != instance._identity
        assert instance.transform == instance._identity
        assert instance.reverse_transform == instance._identity

    def test___init___all(self):
        """Test ``Constraint.__init__`` method when 'all' is passed.

        If 'all' is given, the ``__init__`` method should leave ``transform``,
        ``reverse_transform`` and ``is_valid`` untouched.

        Input:
            - all
        Side effects:
            - is_valid != identity
            - transform != identity
            - reverse_transform != identity
        """
        # Run
        instance = Constraint(handling_strategy='all')

        # Asserts
        assert instance.filter_valid != instance._identity
        assert instance.transform != instance._identity
        assert instance.reverse_transform != instance._identity

    def test___init___not_kown(self):
        """Test ``Constraint.__init__`` method when a not known ``handling_strategy`` is passed.

        If a not known ``handling_strategy`` is given, a ValueError is raised.

        Input:
            - not_known
        Side effects:
            - ValueError
        """
        # Run
        with pytest.raises(ValueError):
            Constraint(handling_strategy='not_known')

    def test_fit(self):
        """Test the ``Constraint.fit`` method.

        The ``Constraint.fit`` method is a no-op method, so nothing needs to happen. We just call
        the method to certify that the interface is right.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        instance.fit(table_data)

    def test_transform(self):
        """Test the ``Constraint.transform`` method. It is an identity method for completion,
        to be optionally overwritten by subclasses.

        The ``Constraint.transform`` method is expected to:
        - Return the input data unmodified.

        Input:
        - Anything
        Output:
        - Input
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        output = instance.transform('input')

        # Assert
        assert output == 'input'

    def test_fit_transform(self):
        """Test the ``Constraint.fit_transform`` method.

        The ``Constraint.fit_transform`` method is expected to:
        - Call the ``fit`` method.
        - Call the ``transform`` method.
        - Return the input data unmodified.

        Input:
        - Anything
        Output:
        - self.transform output
        Side Effects:
        - self.fit is called with input
        - self.transform is called with input
        """
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
        """Test the ``Constraint.reverse_transform`` method. It is an identity method
        for completion, to be optionally overwritten by subclasses.

        The ``Constraint.reverse_transform`` method is expected to:
        - Return the input data unmodified.

        Input:
        - Anything
        Output:
        - Input
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        output = instance.reverse_transform('input')

        # Assert
        assert output == 'input'

    def test_is_valid(self):
        """Test the ``Constraint.is_valid` method. This should be overwritten by all the
        subclasses that have a way to decide which rows are valid and which are not.

        The ``Constraint.is_valid`` method is expected to:
        - Say whether the given table rows are valid.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Series of ``True`` values (pandas.Series)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        out = instance.is_valid(table_data)

        # Assert
        expected_out = pd.Series([True, True, True])
        pd.testing.assert_series_equal(expected_out, out)

    def test_filter_valid(self):
        """Test the ``Constraint.filter_valid`` method.

        The ``Constraint.filter_valid`` method is expected to:
        - Filter the input data by calling the method ``is_valid``.

        Input:
        - Table data (pandas.DataFrame)
        Output:
        - Table data, with only the valid rows (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        constraint_mock = Mock()
        constraint_mock.is_valid.return_value = pd.Series([True, True, False])

        # Run
        out = Constraint.filter_valid(constraint_mock, table_data)

        # Assert
        expected_out = pd.DataFrame({
            'a': [1, 2]
        })
        pd.testing.assert_frame_equal(expected_out, out)

    def test_from_dict_fqn(self):
        """Test the ``Constraint.from_dict`` method passing a FQN.

        If the ``constraint`` string is a FQN, import the class
        before creating an instance of it.

        Input:
        - constraint dict with a FQN and args
        Output:
        - Instance of the subclass with the right args.
        """
        # Setup
        constraint_dict = {
            'constraint': 'sdv.constraints.tabular.UniqueCombinations',
            'columns': ['a', 'b'],
        }

        # Run
        instance = Constraint.from_dict(constraint_dict)

        # Assert
        assert isinstance(instance, UniqueCombinations)
        assert instance._columns == ['a', 'b']

    def test_from_dict_subclass(self):
        """Test the ``Constraint.from_dict`` method passing a subclass name.

        If the ``constraint`` string is a subclass name, take it from the
        Subclasses dict.

        Input:
        - constraint dict with a subclass name and args
        Output:
        - Instance of the subclass with the right args.
        """
        # Setup
        constraint_dict = {
            'constraint': 'UniqueCombinations',
            'columns': ['a', 'b'],
        }

        # Run
        instance = Constraint.from_dict(constraint_dict)

        # Assert
        assert isinstance(instance, UniqueCombinations)
        assert instance._columns == ['a', 'b']

    def test_to_dict(self):
        """Test the ``Constraint.to_dict`` method.

        The ``Constraint.to_dict`` method is expected to return a dict
        containting the FQN of the constraint instance and all the
        required arguments rebuild it.

        Output:
        - Dict with the right values.
        """
        # Run
        instance = UniqueCombinations(columns=['a', 'b'], handling_strategy='transform')
        constraint_dict = instance.to_dict()

        # Assert
        expected_dict = {
            'constraint': 'sdv.constraints.tabular.UniqueCombinations',
            'handling_strategy': 'transform',
            'columns': ['a', 'b'],
        }
        assert constraint_dict == expected_dict
