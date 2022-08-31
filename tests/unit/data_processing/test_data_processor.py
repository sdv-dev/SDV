from unittest.mock import Mock, call, patch

from sdv.constraints.tabular import Positive
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

    def test___init___without_mocks(self):
        """Test the ``__init__`` method without using mocks.

        Setup:
            - Create ``SingleTableMetadata`` instance with one column and one constraint.

        Input:
            - The ``SingleTableMetadata``.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')
        metadata.add_constraint('Positive', column_name='col')

        # Run
        instance = DataProcessor(metadata=metadata)

        # Assert
        assert isinstance(instance.metadata, SingleTableMetadata)
        assert instance.metadata._columns == {'col': {'sdtype': 'numerical'}}
        assert instance.metadata._constraints == [
            {'constraint_name': 'Positive', 'column_name': 'col'}
        ]
        assert len(instance._constraints) == 1
        assert isinstance(instance._constraints[0], Positive)
