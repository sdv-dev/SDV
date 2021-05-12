from unittest.mock import Mock

import pandas as pd
import pytest

from sdv.constraints.errors import MissingConstraintColumnError
from sdv.metadata import Table


class TestTable:

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
        """Test that the `transform` method calls `_transform_constraints` with right parameters"""
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
        table_mock._transform_constraints.assert_called_once_with(data, 'error')

    def test__transform_constraints(self):
        """Test that method correctly transforms data based on constraints

        The `_transform_constraints` method is expected to loop through constraints
        and call each constraint's `transform` method on the data.

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

        The `_transform_constraints` method is expected to raise a `MissingConstraintColumnError`
        if the constraint transform raises one and `on_missing_column` is set to error.

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

        The `_transform_constraints` method is expected to drop columns associated with
        a constraint its transform raises a MissingConstraintColumnError and `on_missing_column`
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
        expected_result = data.drop(['item 0'], axis=1)

        # Run
        result = Table._transform_constraints(table_mock, data, 'drop')

        # Assert
        assert result.equals(expected_result)
