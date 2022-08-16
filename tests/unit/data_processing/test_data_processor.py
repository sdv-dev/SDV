from unittest.mock import Mock, call, patch

from sdv.data_processing.data_processor import DataProcessor


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

    @patch('sdv.data_processing.data_processor.Constraint')
    def test___init__(self, constraint_mock):
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
            enforce_min_max_values=False
        )

        # Assert
        assert data_processor.metadata == metadata_mock
        constraint_mock.from_dict.assert_has_calls(
            [call(constraint1_dict), call(constraint2_dict)])
        transformer = data_processor._transformers_by_sdtype.get('numerical')
        assert transformer.learn_rounding_scheme is True
        assert transformer.enforce_min_max_values is False
