"""Programmable Constraint Integration Tests."""

import pytest

from sdv.cag import FixedCombinations, ProgrammableConstraint, SingleTableProgrammableConstraint
from sdv.datasets.demo import download_demo
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer


class SingleTableIfTrueThenZero(SingleTableProgrammableConstraint):
    """Custom constraint that ensures that if a flag column is True."""

    def __init__(self, target_column, flag_column):
        self.target_column = target_column
        self.flag_column = flag_column

    def validate(self, metadata):
        table_name = metadata._get_single_table_name()
        assert metadata.tables[table_name].columns[self.target_column]['sdtype'] == 'numerical'
        assert metadata.tables[table_name].columns[self.flag_column]['sdtype'] == 'boolean'

    def validate_input_data(self, data):
        return

    def transform(self, data):
        """Transform the data if amenities fee is to be applied."""
        typical_value = data[self.target_column].median()
        data[self.target_column] = data[self.target_column].mask(
            data[self.flag_column], typical_value
        )

        return data

    def reverse_transform(self, transformed_data):
        """Reverse the data if amenities fee is to be applied."""
        transformed_data[self.target_column] = transformed_data[self.target_column].mask(
            transformed_data[self.flag_column], 0.0
        )

        return transformed_data

    def get_updated_metadata(self, metadata):
        return metadata

    def is_valid(self, synthetic_data):
        """Validate that if ``has_rewards`` amenities fee is 0."""
        true_values = (synthetic_data[self.flag_column]) & (
            synthetic_data[self.target_column] == 0.0
        )
        false_values = ~synthetic_data[self.flag_column]
        return (true_values) | (false_values)


@pytest.fixture
def programmable_constraint():
    class MyConstraint(ProgrammableConstraint):
        def __init__(self, column_names, table_name):
            self.column_names = column_names
            self.table_name = table_name
            self._joint_column = '#'.join(self.column_names)
            self._combinations = None

        def _get_single_table_name(self, metadata):
            # Have to define this so that we can re-use existing methods on the constraint
            return self.table_name

        def validate(self, metadata):
            FixedCombinations._validate_pattern_with_metadata(self, metadata)

        def validate_input_data(self, data):
            return

        def fit(self, data, metadata):
            self.metadata = metadata
            FixedCombinations._fit(self, data, metadata)

        def transform(self, data):
            return FixedCombinations._transform(self, data)

        def get_updated_metadata(self, metadata):
            return FixedCombinations._get_updated_metadata(self, metadata)

        def reverse_transform(self, transformed_data):
            return FixedCombinations._reverse_transform(self, transformed_data)

        def is_valid(self, synthetic_data):
            return FixedCombinations._is_valid(self, synthetic_data)

    return MyConstraint


@pytest.fixture
def single_table_programmable_constraint():
    class MyConstraint(SingleTableProgrammableConstraint):
        def __init__(self, column_names, table_name):
            self.column_names = column_names
            self.table_name = table_name
            self._joint_column = '#'.join(self.column_names)
            self._combinations = None

        def _get_single_table_name(self, metadata):
            # Have to define this so that we can re-use existing methods on the constraint
            return self.table_name

        def validate(self, metadata):
            FixedCombinations._validate_pattern_with_metadata(self, metadata)

        def validate_input_data(self, data):
            return

        def fit(self, data, metadata):
            self.metadata = metadata
            data = {self.table_name: data}
            FixedCombinations._fit(self, data, metadata)

        def transform(self, data):
            data = {self.table_name: data}
            transformed = FixedCombinations._transform(self, data)
            return transformed[self.table_name]

        def get_updated_metadata(self, metadata):
            return FixedCombinations._get_updated_metadata(self, metadata)

        def reverse_transform(self, transformed_data):
            transformed_data = {self.table_name: transformed_data}
            reverse_transformed = FixedCombinations._reverse_transform(self, transformed_data)
            return reverse_transformed[self.table_name]

        def is_valid(self, synthetic_data):
            synthetic_data = {self.table_name: synthetic_data}
            is_valid = FixedCombinations._is_valid(self, synthetic_data)
            return is_valid[self.table_name]

    return MyConstraint


def test_end_to_end_programmable_constraint(programmable_constraint):
    """Test using a programmable constraint with a synthesizer end-to-end."""
    data, metadata = download_demo('multi_table', 'fake_hotels')
    my_constraint = programmable_constraint(
        column_names=['has_rewards', 'room_type'], table_name='guests'
    )
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_cag([my_constraint])

    # Run
    synthesizer.fit(data)
    sampled_data = synthesizer.sample(scale=1.0)
    constraints = synthesizer.get_cag()

    # Assert
    original_combinations = set(zip(data['guests']['has_rewards'], data['guests']['room_type']))
    assert (
        set(zip(sampled_data['guests']['has_rewards'], sampled_data['guests']['room_type']))
        == original_combinations
    )
    assert isinstance(constraints[0], programmable_constraint)


def test_end_to_end_single_table_programmable_constraint(single_table_programmable_constraint):
    """Test using a single table programmable constraint with a synthesizer end-to-end."""
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    my_constraint = single_table_programmable_constraint(
        column_names=['has_rewards', 'room_type'], table_name='fake_hotel_guests'
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag([my_constraint])

    # Run
    synthesizer.fit(data)
    sampled_data = synthesizer.sample(1000)
    constraints = synthesizer.get_cag()

    # Assert
    original_combinations = set(zip(data['has_rewards'], data['room_type']))
    assert set(zip(sampled_data['has_rewards'], sampled_data['room_type'])) == original_combinations
    assert isinstance(constraints[0], single_table_programmable_constraint)


def test_end_to_end_simple_constraint_with_no_fit(programmable_constraint):
    """Test using a programmable constraint without a fit method."""
    # Setup
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    custom_constraint = SingleTableIfTrueThenZero(
        target_column='amenities_fee', flag_column='has_rewards'
    )

    # Run
    synthesizer.add_cag([custom_constraint])
    synthesizer.fit(data)
    sampled_data = synthesizer.sample(100)

    # Assert
    true_values = (sampled_data['has_rewards']) & (sampled_data['amenities_fee'] == 0.0)
    false_values = ~sampled_data['has_rewards']
    assert all((true_values) | (false_values))
