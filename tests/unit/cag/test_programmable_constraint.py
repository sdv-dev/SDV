"""Test ProgrammableConstraint and ProgrammableConstraintHarness."""

import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdv.cag._errors import ConstraintNotMetError
from sdv.cag.programmable_constraint import (
    ProgrammableConstraint,
    ProgrammableConstraintHarness,
    SingleTableProgrammableConstraint,
)
from sdv.metadata import Metadata


class TestProgrammableConstraint:
    @pytest.mark.parametrize(
        ('method', 'kwargs'),
        [
            (ProgrammableConstraint.transform, {'data': {}}),
            (ProgrammableConstraint.get_updated_metadata, {'metadata': Metadata()}),
            (ProgrammableConstraint.reverse_transform, {'transformed_data': {}}),
            (ProgrammableConstraint.is_valid, {'synthetic_data': {}}),
        ],
    )
    def test_all_required_methods(self, method, kwargs):
        """Test all methods required to be implemented raise NotImplementedErrors."""
        # Setup
        instance = ProgrammableConstraint()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            method(instance, **kwargs)

    @pytest.mark.parametrize(
        ('method', 'kwargs'),
        [
            (ProgrammableConstraint.validate, {'metadata': Metadata()}),
            (ProgrammableConstraint.validate_input_data, {'data': {}}),
        ],
    )
    def test_validate_methods(self, method, kwargs):
        """Test the validation methods are not required and return None"""
        # Setup
        instance = ProgrammableConstraint()

        # Run and Assert
        assert method(instance, **kwargs) is None

    def test_fit_returns_none(self):
        """Test that the `fit` method returns `None` and doesn't crash."""
        # Setup
        constraint = ProgrammableConstraint()

        # Run
        result = constraint.fit(object(), object())

        # Assert
        assert result is None

    def test_fix_data(self):
        """Test the ``fix_data`` method returns original data by default"""
        # Setup
        constraint = ProgrammableConstraint()
        data = Mock()

        # Run
        fixed_data = constraint.fix_data(data)

        # Assert
        assert fixed_data == data

    def test_get_constraint_dict_errors_missing_attribute(self):
        """Test getting the constraint dict errors if attribute is missing."""

        # Setup
        class MockConstraint(ProgrammableConstraint):
            def __init__(self, param1, param2=None, param3=None):
                self.param1 = param1
                self.param2 = param2

        constraint = MockConstraint('value1')

        # Run and Assert
        expected_msg = re.escape(
            'Cannot convert constraint to dictionary because required parameters '
            "['param3'] are not saved as attributes on the constraint."
        )
        with pytest.raises(AttributeError, match=expected_msg):
            constraint.get_constraint_dict()

    def test_get_constraint_dict(self):
        """Test getting the constraint dict for a programmable constraint."""

        # Setup
        class MockConstraint(ProgrammableConstraint):
            def __init__(self, param1, param2=None):
                self.param1 = param1
                self.param2 = param2

        constraint = MockConstraint('value1')

        # Run
        constraint_dict = constraint.get_constraint_dict()

        # Assert
        assert constraint_dict == {
            'class_name': 'MockConstraint',
            'parameters': {
                'param1': 'value1',
                'param2': None,
            },
        }


class TestProgrammableConstraintHarness:
    def test___init__(self):
        # Setup
        programmable_constraint = ProgrammableConstraint()

        # Run
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Assert
        assert instance.programmable_constraint == programmable_constraint

    def test__validate_constraint(self):
        """Test the ``_validate_constraint`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.validate = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._validate_constraint_with_metadata(metadata)

        # Assert
        programmable_constraint.validate.assert_called_once_with(metadata)

    def test__validate_constraint_with_metadata(self):
        """Test that it errors if SingleTableProgrammableConstraint used with multi-table."""
        # Setup
        single_table_metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        multi_table_metadata = Metadata().load_from_dict({
            'tables': {
                'table1': {'columns': {'col_A': {'sdtype': 'numerical'}}},
                'table2': {
                    'columns': {'col_B': {'sdtype': 'categorical'}},
                },
            }
        })
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.validate = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)
        expected_err = re.escape(
            'SingleTableProgrammableConstraint cannot be used with multi-table metadata '
            'if the `table_name` attribute has not been set. Please set the table name '
            'attribute to the target table, or use the ProgrammableContraint '
            'base class instead.'
        )

        # Run and Assert
        instance._validate_constraint_with_metadata(single_table_metadata)
        with pytest.raises(ConstraintNotMetError, match=expected_err):
            instance._validate_constraint_with_metadata(multi_table_metadata)

    def test__validate_constraint_with_data(self):
        """Test the ``_validate_constraint_with_data`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {
                'table_1': {'columns': {'col_A': {'sdtype': 'numerical'}}},
            }
        })
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.validate_input_data = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._validate_constraint_with_data(data, metadata)

        # Assert
        programmable_constraint.validate_input_data.assert_called_once_with(data)

    def test__validate_constraint_with_data_backwards_compatible(self):
        """Test method is backwards compatible with SingleTableProgrammableConstraint."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {
                'table': {'columns': {'col_A': {'sdtype': 'numerical'}}},
            }
        })
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.validate_input_data = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._validate_constraint_with_data(data, metadata)

        # Assert
        programmable_constraint.validate_input_data.assert_called_once_with(data['table'])

    @patch('sdv.cag.programmable_constraint.deepcopy')
    def test__get_updated_metadata(self, mock_deepcopy):
        """Test the ``_get_updated_metadata`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        copied_metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        mock_deepcopy.return_value = copied_metadata
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.get_updated_metadata = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._get_updated_metadata(metadata)

        # Assert
        programmable_constraint.get_updated_metadata.assert_called_once_with(copied_metadata)

    def test__fit(self):
        """Test the ``_fit`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.fit = Mock()
        programmable_constraint.get_updated_metadata = Mock(return_value=metadata)
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._fit_constraint_column_formatters = Mock()

        # Run
        instance.fit(data, metadata)

        # Assert
        programmable_constraint.fit.assert_called_once_with(data, metadata)
        instance._fit_constraint_column_formatters.assert_called_once_with(data)

    def test__fit_backwards_compatible(self):
        """Test backwards compatiblity with SingleTableProgrammableConstraint."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.fit = Mock()
        programmable_constraint.get_updated_metadata = Mock(return_value=metadata)
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._fit_constraint_column_formatters = Mock()
        instance._table_name = 'table'

        # Run
        instance.fit(data, metadata)

        # Assert
        programmable_constraint.fit.assert_called_once_with(data['table'], metadata)
        instance._fit_constraint_column_formatters.assert_called_once_with(data)

    def test__transform(self):
        """Test the ``_transform`` method."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.transform = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._transform(data)

        # Assert
        programmable_constraint.transform.assert_called_once_with(data)

    def test__transform_backwards_compatible(self):
        """Test the method is backwards compatible with SingleTableProgrammableConstraint."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.transform = Mock(return_value=data['table'])
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._table_name = 'table'

        # Run
        transformed = instance._transform(data)

        # Assert
        programmable_constraint.transform.assert_called_once_with(data['table'])
        assert transformed == {'table': data['table']}

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.reverse_transform = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._reverse_transform(data)

        # Assert
        programmable_constraint.reverse_transform.assert_called_once_with(data)

    def test__reverse_transform_backwards_compatible(self):
        """Test the method is backwards compatible with SingleTableProgrammableConstraint."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.reverse_transform = Mock(return_value=data['table'])
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._table_name = 'table'

        # Run
        reverse_transformed = instance._reverse_transform(data)

        # Assert
        programmable_constraint.reverse_transform.assert_called_once_with(data['table'])
        assert reverse_transformed == {'table': data['table']}

    def test__is_valid(self):
        """Test the ``_is_valid`` method."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.is_valid = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)
        metadata = Mock()

        # Run
        instance._is_valid(data, metadata)

        # Assert
        programmable_constraint.is_valid.assert_called_once_with(data)

    def test__is_valid_backwards_compatability(self):
        """Test the method is backwards compatible with SingleTableProgrammableConstraint."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.is_valid = Mock(return_value=data['table'])
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._get_single_table_name = Mock(return_value='table')
        metadata = Mock()

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        programmable_constraint.is_valid.assert_called_once_with(data['table'])
        assert is_valid == {'table': data['table']}
