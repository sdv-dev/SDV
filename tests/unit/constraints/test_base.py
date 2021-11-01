"""Tests for the sdv.constraints.base module."""
import warnings
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from copulas.multivariate.gaussian import GaussianMultivariate
from copulas.univariate import GaussianUnivariate
from rdt.hyper_transformer import HyperTransformer

from sdv.constraints.base import Constraint, _get_qualified_name, get_subclasses, import_object
from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.tabular import ColumnFormula, UniqueCombinations


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

        The base ``Constraint.fit`` method is expected to:
        - Call ``_fit`` method.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        instance = Constraint(handling_strategy='transform', fit_columns_model=False)
        instance._fit = Mock()

        # Run
        instance.fit(table_data)

        # Assert
        instance._fit.assert_called_once_with(table_data)

    @patch('sdv.constraints.base.GaussianMultivariate', spec_set=GaussianMultivariate)
    def test_fit_gaussian_multivariate_correct_distribution(self, gm_mock):
        """Test the ``GaussianMultivariate`` from the ``Constraint.fit`` method.

        The ``GaussianMultivariate`` is expected to be called with default distribution
        set as ``GaussianUnivariate``.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 2, 3]
        })
        instance = Constraint(handling_strategy='transform', fit_columns_model=True)
        instance.constraint_columns = ('a', 'b')

        # Run
        instance.fit(table_data)

        # Assert
        gm_mock.assert_called_once_with(distribution=GaussianUnivariate)

    @patch('sdv.constraints.base.GaussianMultivariate', spec_set=GaussianMultivariate)
    @patch('sdv.constraints.base.HyperTransformer', spec_set=HyperTransformer)
    def test_fit_trains_column_model(self, ht_mock, gm_mock):
        """Test the ``Constraint.fit`` method trains the column model.

        When ``fit_columns_model`` is True and there are multiple ``constraint_columns``,
        the ``Constraint.fit`` method is expected to:
        - Call ``_fit`` method.
        - Create ``_hyper_transformer``.
        - Create ``_column_model`` and train it.

        Input:
        - Table data (pandas.DataFrame)
        """
        # Setup
        table_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        instance = Constraint(handling_strategy='transform', fit_columns_model=True)
        instance.constraint_columns = ('a', 'b')

        # Run
        instance.fit(table_data)

        # Assert
        gm_mock.return_value.fit.assert_called_once()
        calls = ht_mock.return_value.fit_transform.mock_calls
        args = calls[0][1]
        assert len(calls) == 1
        pd.testing.assert_frame_equal(args[0], table_data)

    def test_transform(self):
        """Test the ``Constraint.transform`` method.

        It is an identity method for completion, to be optionally
        overwritten by subclasses.

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

    def test_transform_calls__transform(self):
        """Test that the ``Constraint.transform`` method calls ``_transform``.

        The ``Constraint.transform`` method is expected to:
        - Return value returned by ``_transform``.

        Input:
        - Anything
        Output:
        - Result of ``_transform(input)``
        """
        # Setup
        constraint_mock = Mock()
        constraint_mock.fit_columns_model = False
        constraint_mock._transform.return_value = 'the_transformed_data'
        constraint_mock._validate_columns.return_value = pd.DataFrame()

        # Run
        output = Constraint.transform(constraint_mock, 'input')

        # Assert
        assert output == 'the_transformed_data'

    def test_transform_model_disabled_any_columns_missing(self):
        """Test the ``Constraint.transform`` method with invalid data.

        If ``table_data`` is missing any columns and ``fit_columns_model``
        is False, it should raise a ``MissingConstraintColumnError``.

        The ``Constraint.transform`` method is expected to:
        - Raise ``MissingConstraintColumnError``.
        """
        # Run
        instance = Constraint(handling_strategy='transform', fit_columns_model=False)
        instance._transform = lambda x: x
        instance.constraint_columns = ('a',)

        # Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame([[1, 2], [3, 4]], columns=['b', 'c']))

    def test_transform_model_enabled_all_columns_missing(self):
        """Test the ``Constraint.transform`` method with missing columns.

        If ``table_data`` is missing all of the ``constraint_columns`` and
        ``fit_columns_model`` is True, it should raise a
        ``MissingConstraintColumnError``.

        The ``Constraint.transform`` method is expected to:
        - Raise ``MissingConstraintColumnError``.
        """
        # Run
        instance = Constraint(handling_strategy='transform')
        instance._transform = lambda x: x
        instance.constraint_columns = ('a',)

        # Assert
        with pytest.raises(MissingConstraintColumnError):
            instance.transform(pd.DataFrame())

    def test_transform_model_enabled_some_columns_missing(self):
        """Test that the ``Constraint.transform`` method uses column model.

        If ``table_data`` is missing some of the ``constraint_columns``,
        the ``_column_model`` should be used to sample the rest and the
        data should be transformed.

        Input:
        - Table with some missing columns.
        Output:
        - Transformed data with all columns.
        """
        # Setup
        instance = Constraint(handling_strategy='transform')
        instance._transform = lambda x: x
        instance.constraint_columns = ('a', 'b')
        instance._hyper_transformer = Mock()
        instance._columns_model = Mock()
        conditions = [
            pd.DataFrame([[5, 1, 2]], columns=['a', 'b', 'c']),
            pd.DataFrame([[6, 3, 4]], columns=['a', 'b', 'c'])
        ]
        transformed_conditions = [
            pd.DataFrame([[1]], columns=['b']),
            pd.DataFrame([[3]], columns=['b'])
        ]
        instance._columns_model.sample.return_value = pd.DataFrame([
            [1, 2, 3]
        ], columns=['b', 'c', 'a'])
        instance._hyper_transformer.transform.side_effect = transformed_conditions
        instance._hyper_transformer.reverse_transform.side_effect = conditions

        # Run
        data = pd.DataFrame([[1, 2], [3, 4]], columns=['b', 'c'])
        transformed_data = instance.transform(data)

        # Assert
        expected_tranformed_data = pd.DataFrame([[1, 2, 3]], columns=['b', 'c', 'a'])
        expected_result = pd.DataFrame([
            [5, 1, 2],
            [6, 3, 4]
        ], columns=['a', 'b', 'c'])
        model_calls = instance._columns_model.sample.mock_calls
        assert len(model_calls) == 2
        instance._columns_model.sample.assert_any_call(num_rows=1, conditions={'b': 1})
        instance._columns_model.sample.assert_any_call(num_rows=1, conditions={'b': 3})
        reverse_transform_calls = instance._hyper_transformer.reverse_transform.mock_calls
        pd.testing.assert_frame_equal(reverse_transform_calls[0][1][0], expected_tranformed_data)
        pd.testing.assert_frame_equal(reverse_transform_calls[1][1][0], expected_tranformed_data)
        pd.testing.assert_frame_equal(transformed_data, expected_result)

    def test_transform_model_enabled_reject_sampling(self):
        """Test the ``Constraint.transform`` method's reject sampling.

        If the column model is used but doesn't return valid rows,
        reject sampling should be used to get the valid rows.

        Setup:
        - The ``_columns_model`` returns some valid_rows the first time,
        and then the rest with the next call.
        Input:
        - Table with some missing columns.
        Output:
        - Transformed data with all columns.
        """
        # Setup
        instance = Constraint(handling_strategy='transform')
        instance._transform = lambda x: x
        instance.constraint_columns = ('a', 'b')
        instance._hyper_transformer = Mock()
        instance._columns_model = Mock()
        transformed_conditions = [pd.DataFrame([[1], [1], [1], [1], [1]], columns=['b'])]
        instance._columns_model.sample.side_effect = [
            pd.DataFrame([
                [1, 2],
                [1, 3]
            ], columns=['a', 'b']),
            pd.DataFrame([
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7]
            ], columns=['a', 'b']),
        ]
        instance._hyper_transformer.transform.side_effect = transformed_conditions
        instance._hyper_transformer.reverse_transform = lambda x: x

        # Run
        data = pd.DataFrame([[1], [1], [1], [1], [1]], columns=['b'])
        transformed_data = instance.transform(data)

        # Assert
        expected_result = pd.DataFrame([
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6]
        ], columns=['a', 'b'])
        model_calls = instance._columns_model.sample.mock_calls
        assert len(model_calls) == 2
        instance._columns_model.sample.assert_any_call(num_rows=5, conditions={'b': 1})
        assert model_calls[1][2]['num_rows'] > 3
        pd.testing.assert_frame_equal(transformed_data, expected_result)

    def test_transform_model_enabled_reject_sampling_error(self):
        """Test that the ``Constraint.transform`` method raises an error appropriately.

        If the column model is used but doesn't return valid rows,
        reject sampling should be used to get the valid rows. If it doesn't
        get any valid rows in 100 tries, a ``ValueError`` is raised.

        Setup:
        - The ``_columns_model`` is fixed to always return an empty ``DataFrame``.
        Input:
        - Table with some missing columns.
        Side Effect:
        - ``ValueError`` raised.
        """
        # Setup
        instance = Constraint(handling_strategy='transform')
        instance.constraint_columns = ('a', 'b')
        instance._hyper_transformer = Mock()
        instance._columns_model = Mock()
        transformed_conditions = pd.DataFrame([[1]], columns=['b'])
        instance._columns_model.sample.return_value = pd.DataFrame()
        instance._hyper_transformer.transform.return_value = transformed_conditions
        instance._hyper_transformer.reverse_transform.return_value = pd.DataFrame()

        # Run / Assert
        data = pd.DataFrame([[1, 2], [3, 4]], columns=['b', 'c'])
        with pytest.raises(ValueError):
            instance.transform(data)

    def test_transform_model_enabled_reject_sampling_duplicates_valid_rows(self):
        """Test the ``Constraint.transform`` method's reject sampling fall back.

        If the column model is used but doesn't return valid rows,
        reject sampling should be used to get the valid rows. If after 100
        tries, some valid rows are created but not enough, then the valid rows
        are duplicated to meet the ``num_rows`` requirement.

        Setup:
        - The ``_columns_model`` returns some valid rows the first time, and then
        an empy ``DataFrame`` for every other call.
        Input:
        - Table with some missing columns.
        Output:
        - Transformed data with all columns.
        """
        # Setup
        instance = Constraint(handling_strategy='transform')
        instance._transform = lambda x: x
        instance.constraint_columns = ('a', 'b')
        instance._hyper_transformer = Mock()
        instance._columns_model = Mock()
        transformed_conditions = [pd.DataFrame([[1], [1], [1], [1], [1]], columns=['b'])]
        instance._columns_model.sample.side_effect = [
            pd.DataFrame([
                [1, 2],
                [1, 3]
            ], columns=['a', 'b'])
        ] + [pd.DataFrame()] * 100
        instance._hyper_transformer.transform.side_effect = transformed_conditions
        instance._hyper_transformer.reverse_transform = lambda x: x

        # Run
        data = pd.DataFrame([[1], [1], [1], [1], [1]], columns=['b'])
        transformed_data = instance.transform(data)

        # Assert
        expected_result = pd.DataFrame([
            [1, 2],
            [1, 3],
            [1, 2],
            [1, 3],
            [1, 2]
        ], columns=['a', 'b'])
        model_calls = instance._columns_model.sample.mock_calls
        assert len(model_calls) == 101
        instance._columns_model.sample.assert_any_call(num_rows=5, conditions={'b': 1})
        pd.testing.assert_frame_equal(transformed_data, expected_result)

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

    def test_filter_valid_with_invalid_index(self):
        """Test the ``Constraint.filter_valid`` method.

        Tests when the is_valid method returns a Series with an invalid index.

        Note: `is_valid.index` can be [0, 1, 5] if, for example, the Series is a subset
        of an original table with 10 rows, but only rows 0/1/5 were selected.

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
        is_valid = pd.Series([True, True, False])
        is_valid.index = [0, 1, 5]
        constraint_mock.is_valid.return_value = is_valid

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

    def test_to_dict_column_formula_lambda(self):
        """Test the ``Constraint.to_dict`` when the constraint is
        a ColumnFormula type and is passed a lambda.

        If the ``Constraint`` type is ColumnFormula,
        and the formula argument is a lambda, the dictionary
        should contain the lambda object as the value.

        Output:
        - Dict with the right values.
        """
        # Run
        instance = ColumnFormula(
            column='a',
            formula=lambda x: x + 1,
            handling_strategy='transform'
        )
        constraint_dict = instance.to_dict()

        # Assert
        assert constraint_dict['formula'](1) == 2

    def test_to_dict_column_formula_returned_function(self):
        """Test the ``Constraint.to_dict`` when the constraint is
        a ColumnFormula type and is passed a function returned
        from another function.

        If the ``Constraint`` type is ColumnFormula,
        and the formula argument is a function returned from another
        function, the dictionary should contain the function as the value.

        Output:
        - Dict with the right values.
        """
        # Run
        def func_creator():
            def func(x):
                return x + 1
            return func
        instance = ColumnFormula(
            column='a',
            formula=func_creator(),
            handling_strategy='transform'
        )
        constraint_dict = instance.to_dict()

        # Assert
        assert constraint_dict['formula'](1) == 2

    def test__validate_constraint_columns_warning(self):
        """Test the ``Constraint._validate_constraint_columns`` method.

        Expect that ``_validate_constraint_columns`` throws a warning
        when missing columns and not using columns model.

        Setup:
        - Mock the constraint columns to have one more column than the table_data: ('a', 'b').
        - Mock the ``_columns_model`` to be False.
        - Mock the ``_sample_constraint_columns`` to return a dataframe.
        Input:
        - table_data with one column ('a').
        Output:
        - table_data
        Side Effects:
        - A UserWarning is thrown.
        """
        # Setup
        constraint = Mock()
        constraint.constraint_columns = ['a', 'b']
        constraint._columns_model = False
        constraint._sample_constraint_columns.return_value = pd.DataFrame({'a': [0, 1, 2]})

        table_data = pd.DataFrame({'a': [0, 1, 2]})

        # Run and assert
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Constraint._validate_constraint_columns(constraint, table_data)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
