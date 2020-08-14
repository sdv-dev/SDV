from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.constraints.base import Constraint, _get_qualified_name, get_subclasses, import_object


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
    - Qualifed name of the function.
    """
    # Run
    fully_qualified_name = _get_qualified_name(get_subclasses)

    # Assert
    expected_name = 'sdv.constraints.base.get_subclasses'
    assert fully_qualified_name == expected_name


def test_import_object_class():
    """Test the ``import_object`` function, when importing a class.

    The ``import_object`` function is expected to:
    - Import a class from its qualifed name.
    """
    # Run
    obj = import_object('sdv.constraints.base.Constraint')

    # Assert
    assert obj is Constraint


def test_import_object_function():
    """Test the ``import_object`` function, when importing a function.

    The ``import_object`` function is expected to:
    - Import a function from its qualifed name.
    """
    # Run
    obj = import_object('sdv.constraints.base.get_subclasses')

    # Assert
    assert obj is get_subclasses


def test_get_subclasses():
    """Test the ``get_subclasses`` function.

    The ``get_subclasses`` function is expected to:
    - Find subclasses for the class object passed.

    Input:
    - A class.
    Output:
    - Dict of the subclasses of the class.
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


class TestConstraint():

    def test___init___transform(self):
        """Test ```Constraint.__init__`` method when 'transform' is passed.

        If 'transform' is given, the __init__ method should
        replace the ``is_valid`` method with an identity
        and leave ``transform`` and ``reverse_transform``
        untouched.

        Input:
            - transform
        Side effects:
            - is_valid == identity
            - transform != identity
            - reverse_transform != identity
        """
        # Run
        instance = Constraint(handling_strategy='transform')

        # Assert
        assert instance.filter_valid == instance._identity
        assert instance.transform != instance._identity
        assert instance.reverse_transform != instance._identity

    def test___init___reject_sampling(self):
        """Test ``Constraint.__init__`` method when 'reject_sampling' is passed.

        If 'reject_sampling' is given, the ``__init__`` method should
        replace the ``transform`` and ``reverse_transform`` methods with an identity
        and leave ``is_valid`` untouched.

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
        - table_data (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        # Run
        instance = Constraint(handling_strategy='transform')
        instance.fit(table_data)

    def test_transform(self):
        """Test the ``Constraint.transform`` method.

        The ``Constraint.transform`` method is expected to:
        - Return the input data unmodified.

        Input:
        - table_data (pandas.DataFrame)
        Output:
        - Input dataframe, unmodified
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        expected_out = table_data

        # Run
        instance = Constraint(handling_strategy='transform')
        out = instance.transform(table_data)

        # Assert
        pd.testing.assert_frame_equal(expected_out, out)

    def test_fit_transform(self):
        """Test the ``Constraint.fit_transform`` method.

        The ``Constraint.fit_transform`` method is expected to:
        - Call the ``fit`` method.
        - Call the ``transform`` method.
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
        """Test the ``Constraint.reverse_transform`` method.

        The ``Constraint.reverse_transform`` method is expected to:
        - Return the input data unmodified.

        Input:
        - table_data (pandas.DataFrame)
        Output:
        - Input dataframe, unmodified
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })

        expected_out = table_data

        # Run
        instance = Constraint(handling_strategy='transform')
        out = instance.reverse_transform(table_data)

        # Assert
        pd.testing.assert_frame_equal(expected_out, out)

    def test_is_valid(self):
        """Test the ``Constraint.is_valid` method.

        The ``Constraint.is_valid`` method is expected to:
        - Say whether the given table rows are valid.

        Input:
        - table_data (pandas.DataFrame)
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
        - Call the ``is_valid`` method.
        - Return only the rows that are valid.

        Input:
        - table_data (pandas.DataFrame)
        Output:
        - table_data (pandas.DataFrame) with only the valid rows.
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

    def test_from_dict(self):
        """Test the ``Constraint.from_dict`` method.

        The ``Constraint.from_dict`` method is expected to:
        - Build a Constraint object from a dict.

        Input:
        - Dict representation of this Constraint.
        Side effects:
        - New Constraint instance.
        """
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
        """Test the ``Constraint.to_dict`` method.

        The ``Constraint.to_dict`` method is expected to:
        - Return a dict representation of this Constraint.

        Output:
        - Dict with the right values.
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        dict = instance.to_dict()

        # Assert
        expected_dict = {
            'constraint': 'sdv.constraints.base.Constraint',
            'handling_strategy': 'transform'
        }

        assert dict == expected_dict
