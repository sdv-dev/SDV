"""Test ProgrammableConstraint and ProgrammableConstraintHarness."""

import re
from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.cag._errors import PatternNotMetError
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
            (ProgrammableConstraint.fit, {'data': {}, 'metadata': Metadata()}),
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

    def test__convert_data(self):
        # Setup
        instance = ProgrammableConstraintHarness(Mock())
        data = pd.DataFrame()
        data.copy = Mock()

        data_dict = {'table': pd.DataFrame()}
        data_dict['table'].copy = Mock()

        # Run
        convert_data = instance._convert_data(data, Metadata())
        convert_data_dict = instance._convert_data(data_dict, Metadata())
        copied_data = instance._convert_data(data, Metadata(), copy=True)
        copied_data_dict = instance._convert_data(data_dict, Metadata(), copy=True)

        # Assert
        pd.testing.assert_frame_equal(convert_data, data)
        assert convert_data_dict == data_dict
        assert copied_data == data.copy.return_value
        assert copied_data_dict == {'table': data_dict['table'].copy.return_value}

    def test__validate_pattern(self):
        """Test the ``_validate_pattern`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.validate = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._validate_pattern_with_metadata(metadata)

        # Assert
        programmable_constraint.validate.assert_called_once_with(metadata)

    def test__validate_pattern_with_metadata_single_table_error(self):
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
        with pytest.raises(PatternNotMetError, match=expected_msg):
            instance._validate_pattern_with_metadata(metadata)

    def test__validate_pattern_with_data(self):
        """Test the ``_validate_pattern_with_data`` method."""
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {'table': {'columns': {'col_A': {'sdtype': 'numerical'}}}}
        })
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.validate = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._validate_pattern_with_metadata(metadata)

        # Assert
        programmable_constraint.validate.assert_called_once_with(metadata)

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
        data = pd.DataFrame({'col_A': range(5)})
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.fit = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._fit(data, metadata)

        # Assert
        programmable_constraint.fit.assert_called_once_with(data, metadata)

    def test__transform(self):
        """Test the ``_transform`` method."""
        # Setup
        data = pd.DataFrame({'col_A': range(5)})
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.transform = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._transform(data)

        # Assert
        programmable_constraint.transform.assert_called_once_with(data)

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method."""
        # Setup
        data = pd.DataFrame({'col_A': range(5)})
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.reverse_transform = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._reverse_transform(data)

        # Assert
        programmable_constraint.reverse_transform.assert_called_once_with(data)

    def test__is_valid(self):
        """Test the ``_is_valid`` method."""
        # Setup
        data = pd.DataFrame({'col_A': range(5)})
        programmable_constraint = ProgrammableConstraint()
        programmable_constraint.is_valid = Mock()
        instance = ProgrammableConstraintHarness(programmable_constraint)

        # Run
        instance._is_valid(data)

        # Assert
        programmable_constraint.is_valid.assert_called_once_with(data)
