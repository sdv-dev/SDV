import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.constraints.errors import (
    FunctionError, MissingConstraintColumnError, MultipleConstraintsErrors)
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
            {'constraint_name': 'Positive', 'column_name': 'col'}]
        assert len(instance._constraints) == 1
        assert isinstance(instance._constraints[0], Positive)

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
        instance._constraints_to_reverse = [Positive('col')]

        # Run
        new_instance = instance.from_dict(instance.to_dict())

        # Assert
        assert instance.metadata.to_dict() == new_instance.metadata.to_dict()
        assert instance._model_kwargs == new_instance._model_kwargs
        assert len(new_instance._constraints) == 1
        assert instance._constraints[0].to_dict() == new_instance._constraints[0].to_dict()
        assert len(new_instance._constraints_to_reverse) == 1
        assert instance._constraints_to_reverse[0].to_dict() == \
            new_instance._constraints_to_reverse[0].to_dict()

        for sdtype, transformer in instance._transformers_by_sdtype.items():
            assert repr(transformer) == repr(new_instance._transformers_by_sdtype[sdtype])

    def test_to_json_from_json(self):
        """Test that ``to_json`` and ``from_json`` methods are inverse to each other.

        Run ``from_json`` on a dict generated by ``to_json``, and ensure the result
        is the same as the original DataProcessor.

        Setup:
            - A DataProcessor with all its attributes set.

        Input:
            - ``from_json`` and ``to_json`` take the same file name.

        Output:
            - The original DataProcessor instance.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')
        metadata.add_constraint('Positive', column_name='col')
        instance = DataProcessor(metadata=metadata)
        instance._constraints_to_reverse = [Positive('col')]

        # Run
        instance.to_json('temp.json')
        new_instance = instance.from_json('temp.json')

        # Assert
        assert instance.metadata.to_dict() == new_instance.metadata.to_dict()
        assert instance._model_kwargs == new_instance._model_kwargs
        assert len(new_instance._constraints) == 1
        assert instance._constraints[0].to_dict() == new_instance._constraints[0].to_dict()
        assert len(new_instance._constraints_to_reverse) == 1
        assert instance._constraints_to_reverse[0].to_dict() == \
            new_instance._constraints_to_reverse[0].to_dict()

        for sdtype, transformer in instance._transformers_by_sdtype.items():
            assert repr(transformer) == repr(new_instance._transformers_by_sdtype[sdtype])
    def test__fit_transform_constraints(self):
        """Test the ``_fit_transform_constraints`` method.

        The method should loop through all the constraints, fit them,
        and then call ``transform`` for all of them.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.

        Input:
            - A ``pandas.DataFrame``.

        Output:
            - Same ``pandas.DataFrame``.

        Side effect:
            - Each constraint should be fit and transform the data.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        transformed_data = pd.DataFrame({'a': [4, 5, 6]})
        dp = DataProcessor(SingleTableMetadata())
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1.transform.return_value = transformed_data
        constraint2.transform.return_value = data
        dp._constraints = [constraint1, constraint2]

        # Run
        constrained_data = dp._fit_transform_constraints(data)

        # Assert
        constraint1.fit.assert_called_once_with(data)
        constraint2.fit.assert_called_once_with(data)
        constraint1.transform.assert_called_once_with(data)
        constraint2.transform.assert_called_once_with(transformed_data)
        pd.testing.assert_frame_equal(constrained_data, data)

    def test__fit_transform_constraints_fit_errors(self):
        """Test the ``_fit_transform_constraints`` method when constraints error on fit.

        The method should loop through all the constraints and try to fit them. If
        any errors are raised, they should be caught and surfaced together.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.
            - Set constraint mocks to raise Exceptions when calling fit.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - A ``MultipleConstraintsErrors`` error should be raised.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        dp = DataProcessor(SingleTableMetadata())
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1.fit.side_effect = Exception('error 1')
        constraint2.fit.side_effect = Exception('error 2')
        dp._constraints = [constraint1, constraint2]

        # Run / Assert
        error_message = re.escape('\nerror 1\n\nerror 2')
        with pytest.raises(MultipleConstraintsErrors, match=error_message):
            dp._fit_transform_constraints(data)

    def test__fit_transform_constraints_transform_errors(self):
        """Test the ``_fit_transform_constraints`` method when constraints error on transform.

        The method should loop through all the constraints and try to fit them. Then it
        should loop through again and try to transform. If any errors are raised, they should be
        caught and surfaced together.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.
            - Set constraint mocks to raise Exceptions when calling transform.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - A ``MultipleConstraintsErrors`` error should be raised.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        dp = DataProcessor(SingleTableMetadata())
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1.transform.side_effect = Exception('error 1')
        constraint2.transform.side_effect = Exception('error 2')
        dp._constraints = [constraint1, constraint2]

        # Run / Assert
        error_message = re.escape('\nerror 1\n\nerror 2')
        with pytest.raises(MultipleConstraintsErrors, match=error_message):
            dp._fit_transform_constraints(data)

        constraint1.fit.assert_called_once_with(data)
        constraint2.fit.assert_called_once_with(data)

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test__fit_transform_constraints_missing_columns_error(self, log_mock):
        """Test the ``_fit_transform_constraints`` method when transform raises a errors.

        The method should loop through all the constraints and try to fit them. Then it
        should loop through again and try to transform. If a ``MissingConstraintColumnError`` or
        ``FunctionError`` is raised, a warning should be raised and reject sampling should be used.

        Setup:
            - Set the ``_constraints`` to be a list of mocked constraints.
            - Set constraint mocks to raise ``MissingConstraintColumnError`` and ``FunctionError``
            when calling transform.
            - Mock warnings module.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - ``MissingConstraintColumnError`` and ``FunctionError`` warning messages.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        dp = DataProcessor(SingleTableMetadata())
        constraint1 = Mock()
        constraint2 = Mock()
        constraint3 = Mock()
        constraint1.transform.return_value = data
        constraint2.transform.side_effect = MissingConstraintColumnError(['column'])
        constraint3.transform.side_effect = FunctionError()
        dp._constraints = [constraint1, constraint2, constraint3]

        # Run
        dp._fit_transform_constraints(data)

        # Assert
        constraint1.fit.assert_called_once_with(data)
        constraint2.fit.assert_called_once_with(data)
        constraint3.fit.assert_called_once_with(data)
        assert log_mock.info.call_count == 2
        message1 = (
            "Mock cannot be transformed because columns: ['column'] were not found. Using the "
            'reject sampling approach instead.'
        )
        message2 = 'Error transforming Mock. Using the reject sampling approach instead.'
        log_mock.info.assert_has_calls([call(message1), call(message2)])

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_fit(self, log_mock):
        """Test the ``fit`` method.

        The ``fit`` method should store the dtypes, fit and transform the constraints
        and then fit the ``HyperTransformer``.

        Setup:
            - Mock the ``_fit_transform_constraints`` method.
            - Mock the ``_fit_hyper_transformer`` method.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - The ``self._dtypes`` method should be set.
            - The ``_fit_transform_constraints`` should be called.
            - The ``_fit_hyper_transformer`` should be called.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]}, dtype=np.int64)
        transformed_data = pd.DataFrame({'a': [4, 5, 6], 'b': [1, 2, 3]})
        dp = Mock()
        dp.table_name = 'fake_table'
        dp._fit_transform_constraints.return_value = transformed_data

        # Run
        DataProcessor.fit(dp, data)

        # Assert
        pd.testing.assert_series_equal(dp._dtypes, pd.Series([np.int64], index=['a']))
        dp._fit_transform_constraints.assert_called_once_with(data)
        dp._fit_hyper_transformer.assert_called_once_with(transformed_data, {'b'})
        fitting_call = call('Fitting table fake_table metadata')
        constraint_call = call('Fitting constraints for table fake_table')
        transformer_call = call('Fitting HyperTransformer for table fake_table')
        log_mock.info.assert_has_calls([fitting_call, constraint_call, transformer_call])
