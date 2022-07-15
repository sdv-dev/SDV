import re
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest
from faker import Faker
from faker.config import DEFAULT_LOCALE
from rdt.transformers.numerical import NumericalTransformer

from sdv.constraints.errors import (
    FunctionError, MissingConstraintColumnError, MultipleConstraintsErrors)
from sdv.metadata import Table


class TestTable:

    def test__get_faker_default_locale(self):
        """Test that ``_get_faker`` without locales parameter has default locale.

        The ``_get_faker`` should return a Faker object localized to the default locale.
        When no locales are specified explicitly.

        Input:
        - Field metadata from metadata dict.
        Output:
        - Faker object with default localization.
        """
        # Setup
        metadata_dict = {
            'fields': {
                'foo': {
                    'type': 'categorical',
                    'pii': True,
                    'pii_category': 'company'
                }
            }
        }

        # Run
        faker = Table.from_dict(metadata_dict)._get_faker(metadata_dict['fields']['foo'])

        # Assert
        assert isinstance(faker, Faker)
        assert faker.locales == [DEFAULT_LOCALE]

    def test__get_faker_specified_locales_string(self):
        """Test that ``_get_faker`` with locales parameter sets localization correctly.

        The ``_get_faker`` should return a Faker object localized to the specified locale.

        Input:
        - Field metadata from metadata dict.
        Output:
        - Faker object with specified localization string.
        """
        # Setup
        metadata_dict = {
            'fields': {
                'foo': {
                    'type': 'categorical',
                    'pii': True,
                    'pii_category': 'company',
                    'pii_locales': 'sv_SE'
                }
            }
        }

        # Run
        faker = Table.from_dict(metadata_dict)._get_faker(metadata_dict['fields']['foo'])

        # Assert
        assert isinstance(faker, Faker)
        assert faker.locales == ['sv_SE']

    def test__get_faker_specified_locales_list(self):
        """Test that ``_get_faker`` with locales parameter sets localization correctly.

        The ``_get_faker`` should return a Faker object localized to the specified locales.

        Input:
        - Field metadata from metadata dict.
        Output:
        - Faker object with specified list of localizations.
        """
        # Setup
        metadata_dict = {
            'fields': {
                'foo': {
                    'type': 'categorical',
                    'pii': True,
                    'pii_category': 'company',
                    'pii_locales': ['en_US', 'sv_SE']
                }
            }
        }

        # Run
        faker = Table.from_dict(metadata_dict)._get_faker(metadata_dict['fields']['foo'])

        # Assert
        assert isinstance(faker, Faker)
        assert faker.locales == ['en_US', 'sv_SE']

    def test__get_faker_method_pass_args(self):
        """Test that ``_get_faker_method`` method utilizes parameters passed in category argument.

        The ``_get_faker_method`` method uses the parameters passed to it in the category argument.

        Input:
        - Faker object to create faked values with.
        - Category tuple of category name and parameters passed to the method creating fake values.
        Output:
        - Fake values created with the specified method from the Faker object.
        Utilizing the arguments given to it.
        """
        # Setup
        metadata_dict = {
            'fields': {
                'foo': {
                    'type': 'categorical',
                    'pii': True,
                    'pii_category': 'ean'
                }
            }
        }
        metadata = Table.from_dict(metadata_dict)

        # Run
        fake_8_ean = metadata._get_faker_method(Faker(), ('ean', 8))
        ean_8 = fake_8_ean()

        fake_13_ean = metadata._get_faker_method(Faker(), ('ean', 13))
        ean_13 = fake_13_ean()

        # Assert
        assert len(ean_8) == 8
        assert len(ean_13) == 13

    @patch('sdv.metadata.Table')
    def test__make_anonymization_mappings(self, mock_table):
        """Test that ``_make_anonymization_mappings`` creates the expected mappings.

        The ``_make_anonymization_mappings`` method should map values in the original
        data to fake values for non-id fields that are labeled pii.

        Setup:
        - Create a Table that has metadata about three fields (one pii field, one id field,
          and one non-pii field).
        Input:
        - Data that contains a pii field, an id field, and a non-pii field.
        Side Effects:
        - Expect ``_get_fake_values`` to be called with the number of unique values of the
          pii field.
        - Expect the resulting `_ANONYMIZATION_MAPPINGS` field to contain the pii field, with
          the correct number of mappings and keys.
        """
        # Setup
        metadata = Mock()
        metadata._ANONYMIZATION_MAPPINGS = {}
        foo_metadata = {
            'type': 'categorical',
            'pii': True,
            'pii_category': 'email',
        }
        metadata._fields_metadata = {
            'foo': foo_metadata,
            'bar': {
                'type': 'categorical',
            },
            'baz': {
                'type': 'id',
            }
        }
        foo_values = ['test1@example.com', 'test2@example.com', 'test3@example.com']
        data = pd.DataFrame({
            'foo': foo_values,
            'bar': ['a', 'b', 'c'],
            'baz': [1, 2, 3],
        })

        # Run
        Table._make_anonymization_mappings(metadata, data)

        # Assert
        assert mock_table._get_fake_values.called_once_with(foo_metadata, 3)

        mappings = metadata._ANONYMIZATION_MAPPINGS[id(metadata)]
        assert len(mappings) == 1

        foo_mappings = mappings['foo']
        assert len(foo_mappings) == 3
        assert list(foo_mappings.keys()) == foo_values

    @patch('sdv.metadata.Table')
    def test__make_anonymization_mappings_unique_faked_value_in_field(self, mock_table):
        """Test that ``_make_anonymization_mappings`` method creates mappings for anonymized values.

        The ``_make_anonymization_mappings`` method should map equal values in the original data
        to the same faked value.

        Input:
        - DataFrame with a field that should be anonymized based on the metadata description.
        Side Effect:
        - Mappings are created from the original values to faked values.
        """
        # Setup
        metadata = Mock()
        metadata._ANONYMIZATION_MAPPINGS = {}
        foo_metadata = {
            'type': 'categorical',
            'pii': True,
            'pii_category': 'email'
        }
        metadata._fields_metadata = {
            'foo': foo_metadata
        }
        data = pd.DataFrame({
            'foo': ['test1@example.com', 'test2@example.com', 'test1@example.com']
        })

        # Run
        Table._make_anonymization_mappings(metadata, data)

        # Assert
        assert mock_table._get_fake_values.called_once_with(foo_metadata, 2)

        mappings = metadata._ANONYMIZATION_MAPPINGS[id(metadata)]
        assert len(mappings) == 1

        foo_mappings = mappings['foo']
        assert len(foo_mappings) == 2
        assert list(foo_mappings.keys()) == ['test1@example.com', 'test2@example.com']

    @patch('sdv.metadata.table.rdt.transformers.NumericalTransformer',
           spec_set=NumericalTransformer)
    def test___init__(self, transformer_mock):
        """Test that ``__init__`` method passes parameters.

        The ``__init__`` method should pass the custom parameters
        to the ``NumericalTransformer``.

        Input:
        - rounding set to an int
        - max_value set to an int
        - min_value set to an int
        Side Effects:
        - ``NumericalTransformer`` should receive the correct parameters
        """
        # Run
        Table(rounding=-1, max_value=100, min_value=-50)

        # Asserts
        assert len(transformer_mock.mock_calls) == 2
        transformer_mock.assert_any_call(
            dtype=int, rounding=-1, max_value=100, min_value=-50)
        transformer_mock.assert_any_call(
            dtype=float, rounding=-1, max_value=100, min_value=-50)

    def test__make_ids(self):
        """Test whether regex is correctly generating expressions."""
        metadata = {'subtype': 'string', 'regex': '[a-d]'}
        keys = Table._make_ids(metadata, 3)
        assert (keys == pd.Series(['a', 'b', 'c'])).all()

    def test__make_ids_fail(self):
        """Test if regex fails with more requested ids than available unique values."""
        metadata = {'subtype': 'string', 'regex': '[a-d]'}
        with pytest.raises(ValueError):
            Table._make_ids(metadata, 20)

    def test__make_ids_unique_field_not_unique(self):
        """Test that id column is replaced with all unique values if not already unique."""
        metadata_dict = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'boolean'}
            },
            'primary_key': 'item 0'
        }
        metadata = Table.from_dict(metadata_dict)
        data = pd.DataFrame({
            'item 0': [0, 1, 1, 2, 3, 5, 5, 6],
            'item 1': [True, True, False, False, True, False, False, True]
        })

        new_data = metadata.make_ids_unique(data)

        assert new_data['item 1'].equals(data['item 1'])
        assert new_data['item 0'].is_unique

    def test__make_ids_unique_field_already_unique(self):
        """Test that id column is kept if already unique."""
        metadata_dict = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'boolean'}
            },
            'primary_key': 'item 0'
        }
        metadata = Table.from_dict(metadata_dict)
        data = pd.DataFrame({
            'item 0': [9, 1, 8, 2, 3, 7, 5, 6],
            'item 1': [True, True, False, False, True, False, False, True]
        })

        new_data = metadata.make_ids_unique(data)

        assert new_data['item 1'].equals(data['item 1'])
        assert new_data['item 0'].equals(data['item 0'])

    def test__make_ids_unique_field_index_out_of_order(self):
        """Test that updated id column is unique even if index is out of order."""
        metadata_dict = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'boolean'}
            },
            'primary_key': 'item 0'
        }
        metadata = Table.from_dict(metadata_dict)
        data = pd.DataFrame({
            'item 0': [0, 1, 1, 2, 3, 5, 5, 6],
            'item 1': [True, True, False, False, True, False, False, True]
        }, index=[0, 1, 1, 2, 3, 5, 5, 6])

        new_data = metadata.make_ids_unique(data)

        assert new_data['item 1'].equals(data['item 1'])
        assert new_data['item 0'].is_unique

    def test_fit_fits_and_transforms_constraints(self):
        """Test the ``fit`` method.

        The ``fit`` method should loop through all the constraints, fit them,
        and then call ``transform`` for all of them.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.

        Input:
            - A ``pandas.DataFrame``.

        Output:
            - Same ``pandas.DataFrame``.

        Side effect:
            - Each constraint should be fit and transform the data.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        transformed_data = pd.DataFrame({'a': [4, 5, 6]})
        instance = Table()
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1.transform.return_value = transformed_data
        constraint2.transform.return_value = data
        instance._constraints = [constraint1, constraint2]

        # Run
        instance.fit(data)

        # Assert
        constraint1.fit.assert_called_once_with(data)
        constraint2.fit.assert_called_once_with(data)
        constraint1.transform.assert_called_once_with(data)
        constraint2.transform.assert_called_once_with(transformed_data)

    def test_fit_constraint_fit_errors(self):
        """Test the ``fit`` method when constraints error on fit.

        The ``fit`` method should loop through all the constraints and try to fit them. If
        any errors are raised, they should be caught and surfaced together.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.
            - Set constraint mocks to raise Exceptions when calling fit.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - A ``MultipleConstraintsErrors`` error should be raised.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        instance = Table()
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1.fit.side_effect = Exception('error 1')
        constraint2.fit.side_effect = Exception('error 2')
        instance._constraints = [constraint1, constraint2]

        # Run / Assert
        error_message = re.escape('\nerror 1\n\nerror 2')
        with pytest.raises(MultipleConstraintsErrors, match=error_message):
            instance.fit(data)

    def test_fit_constraint_transform_errors(self):
        """Test the ``fit`` method when constraints error on transform.

        The ``fit`` method should loop through all the constraints and try to fit them. Then it
        should loop through again and try to transform. If any errors are raised, they should be
        caught and surfaced together.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.
            - Set constraint mocks to raise Exceptions when calling transform.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - A ``MultipleConstraintsErrors`` error should be raised.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        instance = Table()
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1.transform.side_effect = Exception('error 1')
        constraint2.transform.side_effect = Exception('error 2')
        instance._constraints = [constraint1, constraint2]

        # Run / Assert
        error_message = re.escape('\nerror 1\n\nerror 2')
        with pytest.raises(MultipleConstraintsErrors, match=error_message):
            instance.fit(data)

        constraint1.fit.assert_called_once_with(data)
        constraint2.fit.assert_called_once_with(data)

    @patch('sdv.metadata.table.warnings')
    def test_fit_constraint_transform_missing_columns_error(self, warnings_mock):
        """Test the ``fit`` method when transform raises a errors.

        The ``fit`` method should loop through all the constraints and try to fit them. Then it
        should loop through again and try to transform. If a ``MissingConstraintColumnError`` or
        ``FunctionError`` is raised, a warning should be raised and reject sampling should be used.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.
            - Set constraint mocks to raise ``MissingConstraintColumnError`` and ``FunctionError``
            when calling transform.
            - Mock warnings module.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - ``MissingConstraintColumnError`` and ``FunctionError`` warning messages.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        instance = Table()
        constraint1 = Mock()
        constraint2 = Mock()
        constraint3 = Mock()
        constraint1.transform.return_value = data
        constraint2.transform.side_effect = MissingConstraintColumnError(['column'])
        constraint3.transform.side_effect = FunctionError()
        instance._constraints = [constraint1, constraint2, constraint3]

        # Run
        instance.fit(data)

        # Assert
        constraint1.fit.assert_called_once_with(data)
        constraint2.fit.assert_called_once_with(data)
        constraint3.fit.assert_called_once_with(data)
        assert warnings_mock.warn.call_count == 2
        warning_message1 = (
            "Mock cannot be transformed because columns: ['column'] were not found. Using the "
            'reject sampling approach instead.'
        )
        warning_message2 = 'Error transforming Mock. Using the reject sampling approach instead.'
        warnings_mock.warn.assert_has_calls([call(warning_message1), call(warning_message2)])

    def test_transform_calls__transform_constraints(self):
        """Test that the `transform` method calls `_transform_constraints` with right parameters

        The ``transform`` method is expected to call the ``_transform_constraints`` method
        with the data and correct value for ``is_condition``.

        Input:
            - Table data
        Side Effects:
            - Calls _transform_constraints
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        dtypes = {'item 0': 'int', 'item 1': 'bool'}
        table_mock = Mock()
        table_mock.get_dtypes.return_value = dtypes
        table_mock._transform_constraints.return_value = data
        table_mock._anonymize.return_value = data
        table_mock._hyper_transformer.transform.return_value = data

        # Run
        Table.transform(table_mock, data, True)

        # Assert
        expected_data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        mock_calls = table_mock._transform_constraints.mock_calls
        args = mock_calls[0][1]
        assert len(mock_calls) == 1
        assert args[0].equals(expected_data)
        assert args[1] is True

    def test__transform_constraints(self):
        """Test that method correctly transforms data based on constraints

        The ``_transform_constraints`` method is expected to loop through constraints
        and call each constraint's ``transform`` method on the data.

        Input:
            - Table data
        Output:
            - Transformed data
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        transformed_data = pd.DataFrame({
            'item 0': [0, 0.5, 1],
            'item 1': [6, 8, 10]
        }, index=[0, 1, 2])
        first_constraint_mock = Mock()
        second_constraint_mock = Mock()
        first_constraint_mock.transform.return_value = transformed_data
        second_constraint_mock.return_value = transformed_data
        table_instance = Table()
        table_instance._constraints = [first_constraint_mock, second_constraint_mock]

        # Run
        result = table_instance._transform_constraints(data)

        # Assert
        assert result.equals(transformed_data)
        first_constraint_mock.transform.assert_called_once_with(data)
        second_constraint_mock.transform.assert_called_once_with(transformed_data)
        assert table_instance._constraints_to_reverse == [
            first_constraint_mock,
            second_constraint_mock
        ]

    def test__transform_constraints_is_condition_drops_columns(self):
        """Test that method drops columns when necessary.

        The ``_transform_constraints`` method is expected to drop columns associated with
        a constraint when its transform raises a ``MissingConstraintColumnError`` and the
        ``is_condition`` flag is True.

        Input:
            - Table data
            - ``is_condition`` set to True
        Output:
            - Table with dropped columns
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.transform.side_effect = MissingConstraintColumnError(missing_columns=[])
        constraint_mock.constraint_columns = ['item 0']
        table_instance = Table()
        table_instance._constraints = [constraint_mock]
        table_instance._constraints_to_reverse = [constraint_mock]

        # Run
        result = table_instance._transform_constraints(data, True)

        # Assert
        expected_result = pd.DataFrame({
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        assert result.equals(expected_result)
        assert table_instance._constraints_to_reverse == [constraint_mock]

    def test__transform_constraints_is_condition_false_returns_data(self):
        """Test that method returns data unchanged when necessary.

        The ``_transform_constraints`` method is expected to return data unchanged
        when the constraint transform raises a ``MissingConstraintColumnError`` and the
        ``is_condition`` flag is False.

        Input:
            - Table data
        Output:
            - Table with dropped columns
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.transform.side_effect = MissingConstraintColumnError(missing_columns=[])
        constraint_mock.constraint_columns = ['item 0']
        table_instance = Table()
        table_instance._constraints = [constraint_mock]
        table_instance._constraints_to_reverse = [constraint_mock]

        # Run
        result = table_instance._transform_constraints(data, False)

        # Assert
        expected_result = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        assert result.equals(expected_result)
        assert table_instance._constraints_to_reverse == []

    def test_reverse_transform_integer_rounding(self):
        """Test the ``Table.reverse_transform`` method correctly rounds.

        Expect the data to be rounded when the ``_fields_metadata`` specifies
        the ``'transformer'`` as ``'integer'``.

        Input:
        - A dictionary with float values.
        Output:
        - The input dictionary rounded.
        """
        # Setup
        data = pd.DataFrame({'bar': [0.2, 1.7, 2]})
        table = Table()
        table.fitted = True
        table._fields_metadata = {
            'bar': {
                'type': 'numerical',
                'subtype': 'integer'
            },
        }
        table._hyper_transformer = Mock()
        table._hyper_transformer.reverse_transform.return_value = data
        table._constraints_to_reverse = []
        table._dtypes = {'bar': 'int'}
        table._field_names = ['bar']

        # Run
        output = table.reverse_transform(data)

        # Assert
        expected_data = pd.DataFrame({'bar': [0, 2, 2]})
        pd.testing.assert_frame_equal(output, expected_data, check_dtype=False)

    def test_from_dict_min_max(self):
        """Test the ``Table.from_dict`` method.

        Expect that when min_value and max_value are not provided,
        they are set to 'auto'.

        Input:
        - A dictionary representing a table's metadata
        Output:
        - A Table object
        """
        # Setup
        metadata_dict = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'boolean'}
            },
            'primary_key': 'item 0'
        }

        # Run
        metadata = Table.from_dict(metadata_dict)

        # Assert
        assert metadata._transformer_templates['integer'].max_value == 'auto'
        assert metadata._transformer_templates['integer'].min_value == 'auto'
        assert metadata._transformer_templates['integer'].rounding == 'auto'
        assert metadata._transformer_templates['float'].max_value == 'auto'
        assert metadata._transformer_templates['float'].min_value == 'auto'
        assert metadata._transformer_templates['float'].rounding == 'auto'
