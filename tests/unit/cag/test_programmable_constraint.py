"""Test ProgrammableConstraint and ProgrammableConstraintHarness."""

import re
from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.cag._errors import ConstraintNotMetError
from sdv.cag.programmable_constraint import (
    ProgrammableConstraint,
    ProgrammableConstraintHarness,
    SingleTableProgrammableConstraint,
)
from sdv.metadata import Metadata
from tests.utils import DataFrameMatcher


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

    def test__validate_constraint_with_metadata_single_table_error(self):
        """Test the method errors if multi-table metadata supplied to single table constraint."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {
                'table_1': {'columns': {'col_A': {'sdtype': 'numerical'}}},
                'table_2': {'columns': {'col_B': {'sdtype': 'numerical'}}},
            }
        })
        programmable_constraint = SingleTableProgrammableConstraint()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run and Assert
        expected_msg = re.escape(
            'SingleTableProgrammableConstraint cannot be used with multi-table metadata. '
            'Please use the ProgrammableConstraint base class instead.'
        )
        with pytest.raises(ConstraintNotMetError, match=expected_msg):
            instance._validate_constraint_with_metadata(metadata)

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

    def test__validate_constraint_with_data_single_table(self):
        """Test the method converts the data dictionary to a single dataframe.."""
        metadata = Metadata().load_from_dict({
            'tables': {
                'table': {'columns': {'col_A': {'sdtype': 'numerical'}}},
            }
        })
        data_dict = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.validate_input_data = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._validate_constraint_with_data(data_dict, metadata)

        # Assert
        programmable_constraint.validate_input_data.assert_called_once_with(data_dict['table'])

    def test__get_updated_metadata(self):
        """Test the ``_get_updated_metadata`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.get_updated_metadata = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._get_updated_metadata(metadata)

        # Assert
        programmable_constraint.get_updated_metadata.assert_called_once_with(metadata)

    def test__fit(self):
        """Test the ``_fit`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.fit = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance.fit(data, metadata)

        # Assert
        programmable_constraint.fit.assert_called_once_with(data, metadata)

    def test__fit_single_table(self):
        """Test the method handles converting the data dictionary to a single dataframe."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        data = pd.DataFrame({'col_A': range(5)})
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.fit = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance.fit(data, metadata)

        # Assert
        programmable_constraint.fit.assert_called_once_with(data, metadata)

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

    def test__transform_single_table(self):
        """Test method handles converting the data dictionary to and from a single dataframe."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.transform = Mock()
        programmable_constraint.transform.return_value = data['table']
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._table_name = 'table'

        # Run
        transformed = instance._transform(data)

        # Assert
        programmable_constraint.transform.assert_called_once_with(DataFrameMatcher(data['table']))
        assert isinstance(transformed, dict)
        assert set(transformed.keys()) == {'table'}
        pd.testing.assert_frame_equal(transformed['table'], data['table'])

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

    def test__reverse_transform_single_table(self):
        """Test method handles converting the data dictionary to and from a single dataframe."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.reverse_transform = Mock()
        programmable_constraint.reverse_transform.return_value = data['table']
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._table_name = 'table'

        # Run
        reverse_transformed = instance._reverse_transform(data)

        # Assert
        programmable_constraint.reverse_transform.assert_called_once_with(
            DataFrameMatcher(data['table'])
        )
        assert isinstance(reverse_transformed, dict)
        assert set(reverse_transformed.keys()) == {'table'}
        pd.testing.assert_frame_equal(reverse_transformed['table'], data['table'])

    def test__is_valid(self):
        """Test the ``_is_valid`` method."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.is_valid = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._is_valid(data)

        # Assert
        programmable_constraint.is_valid.assert_called_once_with(data)

    def test__is_valid_single_table(self):
        """Test method handles converting the data dictionary to and from a single dataframe."""
        # Setup
        data = {'table': pd.DataFrame({'col_A': range(5)})}
        programmable_constraint = SingleTableProgrammableConstraint()
        programmable_constraint.is_valid = Mock()
        programmable_constraint.is_valid.return_value = pd.Series([True] * 5)
        instance = ProgrammableConstraintHarness(programmable_constraint)
        instance._table_name = 'table'

        # Run
        is_valid = instance._is_valid(data)

        # Assert
        programmable_constraint.is_valid.assert_called_once_with(DataFrameMatcher(data['table']))
        assert isinstance(is_valid, dict)
        assert set(is_valid.keys()) == {'table'}
        pd.testing.assert_series_equal(is_valid['table'], pd.Series([True] * 5))
