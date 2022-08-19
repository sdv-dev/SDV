from unittest.mock import Mock, call, patch

import pandas as pd

from sdv.data_processing.data_processor import DataProcessor
from sdv.metadata.single_table import SingleTableMetadata


class TestDataProcessor:

    @patch('sdv.data_processing.data_processor.Constraint')
    def test__load_constraints(self, constraint_mock):
        """Test the ``_load_constraints`` method.

        The method should take all the constraints in the passed metadata and
        call the ``Constraint.from_dict`` method on them.

        # Setup
            - Patch the ``Constraint`` module.
            - Mock the metadata to have constraint dicts.

        # Side effects:
            - ``self._constraints`` should be populated.
        """
        # Setup
        data_processor = Mock()
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1_dict = {
            'constraint_name': 'Inequality',
            'low_column_name': 'col1',
            'high_column_name': 'col2'
        }
        constraint2_dict = {
            'constraint_name': 'ScalarInequality',
            'column_name': 'col1',
            'relation': '<',
            'value': 10
        }
        constraint_mock.from_dict.side_effect = [
            constraint1, constraint2
        ]
        data_processor.metadata._constraints = [constraint1_dict, constraint2_dict]

        # Run
        loaded_constraints = DataProcessor._load_constraints(data_processor)

        # Assert
        assert loaded_constraints == [constraint1, constraint2]
        constraint_mock.from_dict.assert_has_calls(
            [call(constraint1_dict), call(constraint2_dict)])

    def test__update_numerical_transformer(self):
        """Test the ``_update_numerical_transformer`` method.

        The ``_transformers_by_sdtype`` dict should be updated based on the
        ``learn_rounding_scheme`` and ``enforce_min_max_values`` parameters.

        Input:
            - learn_rounding_scheme set to False.
            - enforce_min_max_values set to False.
        """
        # Setup
        data_processor = Mock()

        # Run
        DataProcessor._update_numerical_transformer(data_processor, False, False)

        # Assert
        transformer_dict = data_processor._transformers_by_sdtype.update.mock_calls[0][1][0]
        transformer = transformer_dict.get('numerical')
        assert transformer.learn_rounding_scheme is False
        assert transformer.enforce_min_max_values is False

    @patch('sdv.data_processing.data_processor.DataProcessor._load_constraints')
    @patch('sdv.data_processing.data_processor.DataProcessor._update_numerical_transformer')
    def test___init__(self, update_transformer_mock, load_constraints_mock):
        """Test the ``__init__`` method.

        Setup:
            - Patch the ``Constraint`` module.

        Input:
            - A mock for metadata.
            - learn_rounding_scheme set to True.
            - enforce_min_max_values set to False.
        """
        # Setup
        metadata_mock = Mock()
        constraint1_dict = {
            'constraint_name': 'Inequality',
            'low_column_name': 'col1',
            'high_column_name': 'col2'
        }
        constraint2_dict = {
            'constraint_name': 'ScalarInequality',
            'column_name': 'col1',
            'relation': '<',
            'value': 10
        }
        metadata_mock._constraints = [constraint1_dict, constraint2_dict]

        # Run
        data_processor = DataProcessor(
            metadata=metadata_mock,
            learn_rounding_scheme=True,
            enforce_min_max_values=False)

        # Assert
        assert data_processor.metadata == metadata_mock
        update_transformer_mock.assert_called_with(True, False)
        load_constraints_mock.assert_called_once()

    def test__make_ids_unique_field_not_unique(self):
        """Test the ``make_ids_unique`` method.

        Test that key columns are replaced with all unique values if not already unique.
        If they a column is already unique or is numerical, do nothing.

        Setup:
            - Instantiate a ``SingleTableMetadata`` with numerical columns
            containing non-unique ids.
            - Set some of the numerical columns as keys.
            - Create a ``DataProcessor`` instance from this metadata.

        Input:
            - Dataframe with index out of order.

        Output:
            - The original dataframe, but the key columns have only unique values.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')
        metadata.add_column('col2', sdtype='boolean')
        metadata.add_column('col3', sdtype='numerical')
        metadata.add_column('col4', sdtype='numerical')
        metadata.add_column('col5', sdtype='numerical')
        metadata.add_column('col6', sdtype='numerical')
        metadata.set_primary_key(('col1', 'col3'))
        metadata.set_sequence_key('col4')
        metadata.set_alternate_keys(['col2', ('col3', 'col6'), 'col5'])

        instance = DataProcessor.from_dict({'metadata': metadata})
        data = pd.DataFrame({
            'col1': [0, 1, 1, 2, 3, 5, 5, 6],
            'col2': [True, True, False, False, True, False, False, True],
            'col3': [0, 1, 1, 2, 3, 5, 5, 6],
            'col4': [0, 1, 1, 2, 3, 5, 5, 6],
            'col5': [0, 1, 1, 2, 3, 5, 5, 6],
            'col6': [0, 1, 2, 3, 4, 5, 6, 7],
        }, index=[0, 1, 1, 2, 3, 5, 5, 7])

        # Run
        new_data = instance.make_ids_unique(data)

        # Assert
        pd.testing.assert_series_equal(new_data['col2'], data['col2'])
        pd.testing.assert_series_equal(new_data['col6'], data['col6'])
        assert new_data['col1'].is_unique
        assert new_data['col3'].is_unique
        assert new_data['col4'].is_unique
        assert new_data['col5'].is_unique

    def test_to_dict_from_dict(self):
        """Test that ``to_dict`` and ``from_dict`` methods are inverse to each other.

        Run ``from_dict`` on a dict generated by ``to_dict``, and ensure the result
        is the same as the original DataProcessor.

        Setup:
            - A DataProcessor with all its attributes set.

        Input:
            - ``from_dict`` takes the output of ``to_dict``.
        
        Output:
            - The original DataProcessor instance.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')
        metadata.add_constraint('Positive', column_name='col')
        instance = DataProcessor(metadata=metadata)

        # Run
        new_instance = instance.from_dict(instance.to_dict())
    
        # Assert
        assert metadata.to_dict() == new_instance.metadata.to_dict()
