from unittest.mock import Mock, patch

import pandas as pd
import pytest
from faker import Faker
from faker.config import DEFAULT_LOCALE
from rdt.transformers.numerical import NumericalTransformer

from sdv.constraints.base import Constraint
from sdv.constraints.errors import MissingConstraintColumnError
from sdv.errors import ConstraintsNotMetError
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

    @patch.object(Constraint, 'from_dict')
    def test__prepare_constraints_sorts_constraints(self, from_dict_mock):
        """Test that ``_prepare_constraints`` method sorts constraints.

        The ``_prepare_constraints`` method should sort constraints by putting
        constraints with ``rebuild_columns`` before the ones without them.

        Input:
        - list of constraints with some having ``rebuild_columns``
        before constraints without them.
        Output:
        - List of constraints sorted properly.
        """
        # Setup
        constraint1 = Constraint(handling_strategy='transform')
        constraint2 = Constraint(handling_strategy='transform')
        constraint3 = Constraint(handling_strategy='reject_sampling')
        constraints = [constraint1, constraint2, constraint3]
        constraint1.rebuild_columns = ['a']
        constraint2.rebuild_columns = ['b']
        constraint3.rebuild_columns = []
        from_dict_mock.side_effect = [constraint1, constraint2, constraint3]

        # Run
        sorted_constraints = Table._prepare_constraints(constraints)

        # Asserts
        assert sorted_constraints == [constraint3, constraint1, constraint2]

    @patch.object(Constraint, 'from_dict')
    def test__prepare_constraints_sorts_constraints_none_rebuild_columns(self, from_dict_mock):
        """Test that ``_prepare_constraints`` method sorts constraints.

        The ``_prepare_constraints`` method should sort constraints with None as
        ``rebuild_columns`` before those that have them.

        Input:
        - list of constraints with some having None as ``rebuild_columns``
        listed after those with ``rebuild_columns``.
        Output:
        - List of constraints sorted properly.
        """
        # Setup
        constraint1 = Constraint(handling_strategy='transform')
        constraint2 = Constraint(handling_strategy='transform')
        constraint3 = Constraint(handling_strategy='reject_sampling')
        constraints = [constraint1, constraint2, constraint3]
        constraint1.rebuild_columns = ['a']
        constraint2.rebuild_columns = ['b']
        constraint3.rebuild_columns = None
        from_dict_mock.side_effect = [constraint1, constraint2, constraint3]

        # Run
        sorted_constraints = Table._prepare_constraints(constraints)

        # Asserts
        assert sorted_constraints == [constraint3, constraint1, constraint2]

    @patch.object(Constraint, 'from_dict')
    def test__prepare_constraints_validates_constraint_order(self, from_dict_mock):
        """Test the ``_prepare_constraints`` method validates the constraint order.

        If no constraint has ``rebuild_columns`` that are in a later
        constraint's ``constraint_columns``, no exception should be raised.

        Input:
        - List of constraints with none having ``rebuild_columns``
        that are in a later constraint's ``constraint_columns``.
        Output:
        - Sorted list of constraints.
        """
        # Setup
        constraint1 = Constraint(handling_strategy='reject_sampling')
        constraint2 = Constraint(handling_strategy='reject_sampling')
        constraint3 = Constraint(handling_strategy='transform')
        constraint4 = Constraint(handling_strategy='transform')
        constraints = [constraint1, constraint2, constraint3, constraint4]
        constraint3.rebuild_columns = ['e', 'd']
        constraint4.constraint_columns = ['a', 'b', 'c']
        constraint4.rebuild_columns = ['a']
        from_dict_mock.side_effect = [constraint1, constraint2, constraint3, constraint4]

        # Run
        sorted_constraints = Table._prepare_constraints(constraints)

        # Assert
        assert sorted_constraints == constraints

    @patch.object(Constraint, 'from_dict')
    def test__prepare_constraints_invalid_order_raises_exception(self, from_dict_mock):
        """Test the ``_prepare_constraints`` method validates the constraint order.

        If one constraint has ``rebuild_columns`` that are in a later
        constraint's ``constraint_columns``, an exception should be raised.

        Input:
        - List of constraints with some having ``rebuild_columns``
        that are in a later constraint's ``constraint_columns``.
        Side Effect:
        - Exception should be raised.
        """
        # Setup
        constraint1 = Constraint(handling_strategy='reject_sampling')
        constraint2 = Constraint(handling_strategy='reject_sampling')
        constraint3 = Constraint(handling_strategy='transform')
        constraint4 = Constraint(handling_strategy='transform')
        constraints = [constraint1, constraint2, constraint3, constraint4]
        constraint3.rebuild_columns = ['a', 'd']
        constraint4.constraint_columns = ['a', 'b', 'c']
        constraint4.rebuild_columns = ['a']
        from_dict_mock.side_effect = [constraint1, constraint2, constraint3, constraint4]

        # Run
        with pytest.raises(Exception):
            Table._prepare_constraints(constraints)

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

    @patch.object(Table, '_prepare_constraints')
    def test___init__calls_prepare_constraints(self, _prepare_constraints_mock):
        """Test that ``__init__`` method calls ``_prepare_constraints"""
        # Run
        Table(constraints=[])

        # Assert
        _prepare_constraints_mock.called_once_with([])

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

    def test_transform_calls__transform_constraints(self):
        """Test that the `transform` method calls `_transform_constraints` with right parameters

        The ``transform`` method is expected to call the ``_transform_constraints`` method
        with the data and correct value for ``on_missing_column``.

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
        Table.transform(table_mock, data, 'error')

        # Assert
        expected_data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        mock_calls = table_mock._transform_constraints.mock_calls
        args = mock_calls[0][1]
        assert len(mock_calls) == 1
        assert args[0].equals(expected_data)
        assert args[1] == 'error'

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
        table_mock = Mock()
        table_mock._constraints = [first_constraint_mock, second_constraint_mock]

        # Run
        result = Table._transform_constraints(table_mock, data)

        # Assert
        assert result.equals(transformed_data)
        first_constraint_mock.transform.assert_called_once_with(data)
        second_constraint_mock.transform.assert_called_once_with(transformed_data)

    def test__transform_constraints_raises_error(self):
        """Test that method raises error when specified.

        The ``_transform_constraints`` method is expected to raise ``MissingConstraintColumnError``
        if the constraint transform raises one and ``on_missing_column`` is set to error.

        Input:
        - Table data
        Side Effects:
        - MissingConstraintColumnError
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.transform.side_effect = MissingConstraintColumnError
        table_mock = Mock()
        table_mock._constraints = [constraint_mock]

        # Run/Assert
        with pytest.raises(MissingConstraintColumnError):
            Table._transform_constraints(table_mock, data, 'error')

    def test__transform_constraints_drops_columns(self):
        """Test that method drops columns when specified.

        The ``_transform_constraints`` method is expected to drop columns associated with
        a constraint its transform raises a MissingConstraintColumnError and ``on_missing_column``
        is set to drop.

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
        constraint_mock.transform.side_effect = MissingConstraintColumnError
        constraint_mock.constraint_columns = ['item 0']
        table_mock = Mock()
        table_mock._constraints = [constraint_mock]

        # Run
        result = Table._transform_constraints(table_mock, data, 'drop')

        # Assert
        expected_result = pd.DataFrame({
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        assert result.equals(expected_result)

    def test__validate_data_on_constraints(self):
        """Test the ``Table._validate_data_on_constraints`` method.

        Expect that the method returns True when the constraint columns are in the given data,
        and the constraint.is_valid method returns True.

        Input:
        - Table data
        Output:
        - None
        Side Effects:
        - No error
        """
        # Setup
        data = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.is_valid.return_value = pd.Series([True, True, True])
        constraint_mock.constraint_columns = ['a', 'b']
        table_mock = Mock()
        table_mock._constraints = [constraint_mock]

        # Run
        result = Table._validate_data_on_constraints(table_mock, data)

        # Assert
        assert result is None

    def test__validate_data_on_constraints_invalid_input(self):
        """Test the ``Table._validate_data_on_constraints`` method.

        Expect that the method returns False when the constraint columns are in the given data,
        and the constraint.is_valid method returns False for any row.

        Input:
        - Table data contains an invalid row
        Output:
        - None
        Side Effects:
        - A ConstraintsNotMetError is thrown
        """
        # Setup
        data = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.is_valid.return_value = pd.Series([True, False, True])
        constraint_mock.constraint_columns = ['a', 'b']
        table_mock = Mock()
        table_mock._constraints = [constraint_mock]

        # Run and assert
        with pytest.raises(ConstraintsNotMetError):
            Table._validate_data_on_constraints(table_mock, data)

    def test__validate_data_on_constraints_missing_cols(self):
        """Test the ``Table._validate_data_on_constraints`` method.

        Expect that the method returns True when the constraint columns are not
        in the given data.

        Input:
        - Table data that is missing a constraint column
        Output:
        - None
        Side Effects:
        - No error
        """
        # Setup
        data = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.constraint_columns = ['a', 'b', 'c']
        table_mock = Mock()
        table_mock._constraints = [constraint_mock]

        # Run
        result = Table._validate_data_on_constraints(table_mock, data)

        # Assert
        assert result is None

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
