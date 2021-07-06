from unittest.mock import Mock, patch

import pandas as pd
import pytest
from rdt.transformers.numerical import NumericalTransformer

from sdv.constraints.errors import MissingConstraintColumnError
from sdv.metadata import Table


class TestTable:

    @patch('sdv.metadata.table.rdt.transformers.NumericalTransformer',
           spec_set=NumericalTransformer)
    def test__init__(self, transformer_mock):
        """
        Test that ``__init__`` method passes parameters.

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
