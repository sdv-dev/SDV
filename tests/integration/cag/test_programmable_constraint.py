"""Programmable Constraint Integration Tests."""

import pytest

from sdv.cag import FixedCombinations, ProgrammableConstraint, SingleTableProgrammableConstraint
from sdv.datasets.demo import download_demo
from sdv.single_table import GaussianCopulaSynthesizer


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
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    my_constraint = programmable_constraint(
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
