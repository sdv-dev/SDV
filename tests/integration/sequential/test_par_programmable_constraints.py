"""Integration tests for PARSynthesizer with programmable constraints."""

import pandas as pd
import pytest

from sdv.cag import ProgrammableConstraint, SingleTableProgrammableConstraint
from sdv.errors import SynthesizerInputError
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer


class SimpleCustomConstraint(SingleTableProgrammableConstraint):
    """Simple custom constraint for testing."""

    def __init__(self, column_name):
        self.column_name = column_name

    def validate(self, metadata):
        """Validate that the column exists in metadata."""
        table_name = list(metadata.tables.keys())[0]
        if self.column_name not in metadata.tables[table_name].columns:
            raise ValueError(f'Column {self.column_name} not found in metadata')

    def validate_input_data(self, data):
        """Validate the input data."""
        if self.column_name not in data.columns:
            raise ValueError(f'Column {self.column_name} not found in data')

    def fit(self, data, metadata):
        """Fit the constraint."""
        self._median = data[self.column_name].median()

    def transform(self, data):
        """Transform the data by capping values at the median."""
        data = data.copy()
        data[self.column_name] = data[self.column_name].clip(upper=self._median)
        return data

    def reverse_transform(self, transformed_data):
        """Reverse transform - no changes needed."""
        return transformed_data

    def get_updated_metadata(self, metadata):
        """Return the same metadata."""
        return metadata

    def is_valid(self, synthetic_data):
        """Check if synthetic data is valid."""
        return synthetic_data[self.column_name] <= self._median


class ConditionalConstraint(SingleTableProgrammableConstraint):
    """Constraint that enforces conditional logic between columns."""

    def __init__(self, flag_column, target_column):
        self.flag_column = flag_column
        self.target_column = target_column

    def validate(self, metadata):
        """Validate that both columns exist."""
        table_name = list(metadata.tables.keys())[0]
        columns = metadata.tables[table_name].columns
        if self.flag_column not in columns:
            raise ValueError(f'Column {self.flag_column} not found in metadata')
        if self.target_column not in columns:
            raise ValueError(f'Column {self.target_column} not found in metadata')

    def validate_input_data(self, data):
        """Validate the input data."""
        if self.flag_column not in data.columns:
            raise ValueError(f'Column {self.flag_column} not found in data')
        if self.target_column not in data.columns:
            raise ValueError(f'Column {self.target_column} not found in data')

    def fit(self, data, metadata):
        """Fit the constraint."""
        self._typical_value = data[self.target_column].median()

    def transform(self, data):
        """Transform: when flag is True, set target to typical value."""
        data = data.copy()
        mask = data[self.flag_column] == True  # noqa: E712
        data.loc[mask, self.target_column] = self._typical_value
        return data

    def reverse_transform(self, transformed_data):
        """Reverse transform: when flag is True, set target to 0."""
        transformed_data = transformed_data.copy()
        mask = transformed_data[self.flag_column] == True  # noqa: E712
        transformed_data.loc[mask, self.target_column] = 0.0
        return transformed_data

    def get_updated_metadata(self, metadata):
        """Return the same metadata."""
        return metadata

    def is_valid(self, synthetic_data):
        """Check if synthetic data is valid."""
        flag_true = synthetic_data[self.flag_column] == True  # noqa: E712
        target_zero = synthetic_data[self.target_column] == 0.0
        flag_false = synthetic_data[self.flag_column] == False  # noqa: E712

        return (flag_true & target_zero) | flag_false


class MultiTableConstraint(ProgrammableConstraint):
    """Multi-table constraint that should be rejected."""

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, data):
        return data

    def reverse_transform(self, transformed_data):
        return transformed_data

    def get_updated_metadata(self, metadata):
        return metadata

    def is_valid(self, synthetic_data):
        return pd.Series([True] * len(list(synthetic_data.values())[0]))


@pytest.fixture
def sample_sequential_data():
    """Create sample sequential data for testing."""
    data = pd.DataFrame({
        'patient_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'visit_number': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'measurement': [10.5, 12.3, 15.1, 8.2, 9.7, 11.4, 13.8, 16.2, 18.9],
        'has_condition': [False, False, False, True, True, True, False, False, False],
        'treatment_score': [5.0, 6.2, 7.5, 0.0, 0.0, 0.0, 7.1, 8.0, 9.2],
    })
    return data


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    metadata = Metadata.load_from_dict({
        'tables': {
            'patients': {
                'columns': {
                    'patient_id': {'sdtype': 'id'},
                    'visit_number': {'sdtype': 'numerical'},
                    'measurement': {'sdtype': 'numerical'},
                    'has_condition': {'sdtype': 'boolean'},
                    'treatment_score': {'sdtype': 'numerical'},
                },
                'sequence_key': 'patient_id',
                'sequence_index': 'visit_number',
            }
        }
    })
    return metadata


def test_add_single_table_programmable_constraint(sample_sequential_data, sample_metadata):
    """Test that SingleTableProgrammableConstraint can be added to PARSynthesizer."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata, context_columns=['has_condition'])
    constraint = SimpleCustomConstraint('measurement')

    # Run
    synthesizer.add_constraints([constraint])

    # Assert
    constraints = synthesizer.get_constraints()
    assert len(constraints) == 1
    assert isinstance(constraints[0], SimpleCustomConstraint)
    assert constraints[0].column_name == 'measurement'


def test_reject_multi_table_programmable_constraint(sample_metadata):
    """Test that multi-table ProgrammableConstraint is rejected."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata)
    constraint = MultiTableConstraint('measurement')

    # Run & Assert
    with pytest.raises(
        SynthesizerInputError,
        match='not compatible with the single table synthesizers',
    ):
        synthesizer.add_constraints([constraint])


def test_end_to_end_with_simple_constraint(sample_sequential_data, sample_metadata):
    """Test end-to-end workflow with a simple constraint."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata, context_columns=['has_condition'], epochs=1)
    constraint = SimpleCustomConstraint('measurement')
    synthesizer.add_constraints([constraint])

    # Run
    synthesizer.fit(sample_sequential_data)
    synthetic_data = synthesizer.sample(num_sequences=2)

    # Assert
    assert len(synthetic_data) > 0
    assert 'measurement' in synthetic_data.columns
    assert 'patient_id' in synthetic_data.columns
    assert 'has_condition' in synthetic_data.columns

    # Check constraint is satisfied
    original_median = sample_sequential_data['measurement'].median()
    assert all(synthetic_data['measurement'] <= original_median)


def test_end_to_end_with_conditional_constraint(sample_sequential_data, sample_metadata):
    """Test end-to-end workflow with a conditional constraint."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata, context_columns=['has_condition'], epochs=1)
    constraint = ConditionalConstraint('has_condition', 'treatment_score')
    synthesizer.add_constraints([constraint])

    # Run
    synthesizer.fit(sample_sequential_data)
    synthetic_data = synthesizer.sample(num_sequences=2)

    # Assert
    assert len(synthetic_data) > 0

    condition_true = synthetic_data['has_condition'] == True  # noqa: E712
    if condition_true.any():
        assert all(synthetic_data.loc[condition_true, 'treatment_score'] == 0.0)


def test_constraint_with_context_columns(sample_sequential_data, sample_metadata):
    """Test constraint that operates on context columns."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata, context_columns=['has_condition'], epochs=1)

    class BooleanConstraint(SingleTableProgrammableConstraint):
        def __init__(self, column_name):
            self.column_name = column_name

        def validate(self, metadata):
            pass

        def validate_input_data(self, data):
            pass

        def fit(self, data, metadata):
            pass

        def transform(self, data):
            return data

        def reverse_transform(self, transformed_data):
            return transformed_data

        def get_updated_metadata(self, metadata):
            return metadata

        def is_valid(self, synthetic_data):
            return pd.Series([True] * len(synthetic_data))

    constraint = BooleanConstraint('has_condition')
    synthesizer.add_constraints([constraint])

    # Run
    synthesizer.fit(sample_sequential_data)
    synthetic_data = synthesizer.sample(num_sequences=2)

    # Assert
    assert len(synthetic_data) > 0
    assert 'has_condition' in synthetic_data.columns


def test_constraint_with_non_context_columns(sample_sequential_data, sample_metadata):
    """Test constraint that operates on non-context columns."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata, context_columns=['has_condition'], epochs=1)
    constraint = SimpleCustomConstraint('measurement')
    synthesizer.add_constraints([constraint])

    # Run
    synthesizer.fit(sample_sequential_data)
    synthetic_data = synthesizer.sample(num_sequences=2)

    # Assert
    assert len(synthetic_data) > 0
    assert 'measurement' in synthetic_data.columns


def test_multiple_constraints_same_type(sample_sequential_data, sample_metadata):
    """Test adding multiple constraints of the same type."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata, context_columns=['has_condition'], epochs=1)
    constraint1 = SimpleCustomConstraint('measurement')
    constraint2 = SimpleCustomConstraint('treatment_score')
    synthesizer.add_constraints([constraint1, constraint2])

    # Run
    synthesizer.fit(sample_sequential_data)
    synthetic_data = synthesizer.sample(num_sequences=2)

    # Assert
    assert len(synthetic_data) > 0
    constraints = synthesizer.get_constraints()
    assert len(constraints) == 2


def test_validate_constraints(sample_sequential_data, sample_metadata):
    """Test that constraints are properly validated during synthesis."""
    # Setup
    synthesizer = PARSynthesizer(sample_metadata, context_columns=['has_condition'], epochs=1)
    constraint = ConditionalConstraint('has_condition', 'treatment_score')
    synthesizer.add_constraints([constraint])
    synthesizer.fit(sample_sequential_data)

    # Run
    synthetic_data = synthesizer.sample(num_sequences=3)

    # Assert
    is_valid = constraint.is_valid(synthetic_data)
    assert all(is_valid)
