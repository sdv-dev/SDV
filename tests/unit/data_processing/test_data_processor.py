import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
from rdt.errors import ConfigNotSetError
from rdt.errors import NotFittedError as RDTNotFittedError
from rdt.transformers import FloatFormatter, LabelEncoder, UnixTimestampEncoder

from sdv.constraints.errors import MissingConstraintColumnError
from sdv.constraints.tabular import Positive, ScalarRange
from sdv.data_processing.data_processor import DataProcessor
from sdv.data_processing.datetime_formatter import DatetimeFormatter
from sdv.data_processing.errors import InvalidConstraintsError, NotFittedError
from sdv.data_processing.numerical_formatter import NumericalFormatter
from sdv.errors import SynthesizerInputError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.base import BaseSynthesizer


class TestDataProcessor:

    def test__update_numerical_transformer(self):
        """Test the ``_update_numerical_transformer`` method.

        The ``_transformers_by_sdtype`` dict should be updated based on the
        ``enforce_rounding`` and ``enforce_min_max_values`` parameters.

        Input:
            - enforce_rounding set to False.
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

    @patch('sdv.data_processing.data_processor.rdt')
    @patch('sdv.data_processing.data_processor.DataProcessor._update_numerical_transformer')
    def test___init__(self, update_transformer_mock, mock_rdt):
        """Test the ``__init__`` method.

        Setup:
            - Patch the ``Constraint`` module.

        Input:
            - A mock for metadata.
            - enforce_rounding set to True.
            - enforce_min_max_values set to False.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='id')
        metadata.add_column('col_2', sdtype='id')
        metadata.add_alternate_keys(['col_2'])
        metadata.set_primary_key('col')

        # Run
        data_processor = DataProcessor(
            metadata=metadata,
            enforce_rounding=True,
            enforce_min_max_values=False,
            locales='en_US'
        )

        # Assert
        assert data_processor.metadata == metadata
        assert data_processor._enforce_rounding is True
        assert data_processor._enforce_min_max_values is False
        assert data_processor._locales == 'en_US'
        assert data_processor._model_kwargs == {}
        assert data_processor._constraints_list == []
        assert data_processor._constraints == []
        assert data_processor._constraints_to_reverse == []
        assert data_processor.table_name is None
        assert data_processor.fitted is False
        assert data_processor._dtypes is None
        assert data_processor.formatters == {}
        assert data_processor._keys == ['col_2', 'col']
        assert data_processor._prepared_for_fitting is False

        assert data_processor._hyper_transformer == mock_rdt.HyperTransformer.return_value
        update_transformer_mock.assert_called_with(True, False)

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

        # Run
        instance = DataProcessor(metadata=metadata)

        # Assert
        assert isinstance(instance.metadata, SingleTableMetadata)
        assert instance.metadata.columns == {'col': {'sdtype': 'numerical'}}

    def test_filter_valid(self):
        """Test that we are calling the ``filter_valid`` of each constraint over the data."""
        # Setup
        data = pd.DataFrame({
            'numbers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'range': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        })
        instance = Mock()
        scalar_range = ScalarRange('range', low_value=0, high_value=90, strict_boundaries=True)
        positive = Positive('numbers')
        instance._constraints = [scalar_range, positive]

        # Run
        data = DataProcessor.filter_valid(instance, data)

        # Assert
        expected_data = pd.DataFrame({
            'numbers': [1, 2, 3, 4, 5, 6, 7, 8],
            'range': [10, 20, 30, 40, 50, 60, 70, 80]
        }, index=[1, 2, 3, 4, 5, 6, 7, 8])
        pd.testing.assert_frame_equal(expected_data, data)

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
        instance = DataProcessor(metadata=metadata)
        constraints = [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col'}
            }
        ]
        instance.add_constraints(constraints)
        instance._constraints_to_reverse = [Positive('col')]

        # Run
        new_instance = instance.from_dict(instance.to_dict())

        # Assert
        assert instance.metadata.to_dict() == new_instance.metadata.to_dict()
        assert instance._model_kwargs == new_instance._model_kwargs
        assert len(new_instance._constraints_list) == 1
        assert new_instance._constraints_list == [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col'}
            }
        ]

        assert len(new_instance._constraints_to_reverse) == 1
        assert instance._constraints_to_reverse[0].to_dict() == \
            new_instance._constraints_to_reverse[0].to_dict()

        for sdtype, transformer in instance._transformers_by_sdtype.items():
            assert repr(transformer) == repr(new_instance._transformers_by_sdtype[sdtype])

    def test_to_json_from_json(self, tmp_path):
        """Test that ``to_json`` and ``from_json`` methods are inverse to each other.

        Run ``from_json`` on a dict generated by ``to_json``, and ensure the result
        is the same as the original DataProcessor.

        Setup:
            - A DataProcessor with all its attributes set.
            - Use ``TemporaryDirectory`` to store the file.

        Input:
            - ``from_json`` and ``to_json`` take the same file name.

        Output:
            - The original DataProcessor instance.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')
        instance = DataProcessor(metadata=metadata)
        constraints = [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col'}
            }
        ]
        instance.add_constraints(constraints)

        instance._constraints_to_reverse = [Positive('col')]

        # Run
        file_name = tmp_path / 'temp.json'
        instance.to_json(file_name)
        new_instance = instance.from_json(file_name)

        # Assert
        assert instance.metadata.to_dict() == new_instance.metadata.to_dict()
        assert instance._model_kwargs == new_instance._model_kwargs
        assert len(new_instance._constraints_list) == 1
        assert new_instance._constraints_list == [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col'}
            }
        ]

        assert len(new_instance._constraints_to_reverse) == 1
        assert instance._constraints_to_reverse[0].to_dict() == \
            new_instance._constraints_to_reverse[0].to_dict()

        for sdtype, transformer in instance._transformers_by_sdtype.items():
            assert repr(transformer) == repr(new_instance._transformers_by_sdtype[sdtype])

    def test_get_model_kwargs(self):
        """Test the ``get_model_kwargs`` method.

        The method should return a copy of the ``model_kwargs``.

        Input:
            - Model name.

        Output:
            - model key word args.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())
        dp._model_kwargs = {'model': {'arg1': 10, 'arg2': True}}

        # Run
        model_kwargs = dp.get_model_kwargs('model')

        # Assert
        assert model_kwargs == {'arg1': 10, 'arg2': True}

    def test_set_model_kwargs(self):
        """Test the ``set_model_kwargs`` method.

        The method should set the ``model_kwargs`` for the provided model name.

        Input:
            - Model name.
            - Model key word args.

        Side effect:
            - ``_model_kwargs`` should be set.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())

        # Run
        dp.set_model_kwargs('model', {'arg1': 10, 'arg2': True})

        # Assert
        assert dp._model_kwargs == {'model': {'arg1': 10, 'arg2': True}}

    def test_get_sdtypes(self):
        """Test that this returns a mapping of column names and its sdtypes.

        This test ensures that a dictionary is returned with column name as key and
        ``sdtype`` as value. When ``primary_keys`` is ``False`` this should not be included.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        metadata.add_column('col2', sdtype='id')
        metadata.add_column('col3', sdtype='numerical', computer_representation='Int8')
        metadata.set_primary_key('col2')
        dp = DataProcessor(metadata)

        # Run
        sdtypes = dp.get_sdtypes()

        # Assert
        assert sdtypes == {
            'col1': 'categorical',
            'col3': 'numerical'
        }

    def test_get_sdtypes_with_primary_keys(self):
        """Test that this returns a mapping of column names and it's sdtypes.

        This test ensures that a dictionary is returned with column name as key and
        ``sdtype`` as value. When ``primary_keys`` is ``True`` this should be included.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        metadata.add_column('col2', sdtype='id')
        metadata.add_column('col3', sdtype='numerical', computer_representation='Int8')
        metadata.set_primary_key('col2')
        dp = DataProcessor(metadata)

        # Run
        sdtypes = dp.get_sdtypes(primary_keys=True)

        # Assert
        assert sdtypes == {
            'col1': 'categorical',
            'col2': 'id',
            'col3': 'numerical'
        }

    def test__validate_custom_constraints(self):
        """Test that ``_validate_custom_constraints`` doesn't raise an error."""
        # Setup
        class_names = [
            'CustomCons',
            'CustomCons2'
        ]
        filepath = 'example/myfile.py'
        instance = Mock()
        module = Mock()

        # Run and Assert
        DataProcessor._validate_custom_constraints(instance, filepath, class_names, module)

    def test__validate_custom_constraint_name_raises_error(self):
        """Test the method's error case.

        An error should be raised if the name of the custom constraint matches the name of
        a predefined constraint.
        """
        # Setup
        instance = Mock()

        # Run and Assert
        error_msg = re.escape(
            'The provided constraint is invalid:'
            "\nThe name 'Positive' is a reserved constraint name. Please use a different one for "
            'the custom constraint.'
        )
        with pytest.raises(InvalidConstraintsError, match=error_msg):
            DataProcessor._validate_custom_constraint_name(instance, 'Positive')

    def test__validate_custom_constraints_raises_an_error(self):
        """Test the ``_validate_custom_constraints``.

        Ensure that the method will raise an error if the ``class_name`` is within the reserved
        class names (the default ones provided by ``SDV``) and when the constraint class is not
        found in the given ``filepath``.
        """
        # Setup
        class_names = [
            'CustomCons',
            'Positive',
            'CustomCons2'
        ]
        filepath = 'example/myfile.py'
        instance = Mock()
        module = Mock()
        del module.CustomCons2
        name_error = InvalidConstraintsError(
            "The name 'Positive' is a reserved constraint name. Please use a different one for "
            'the custom constraint.'
        )
        instance._validate_custom_constraint_name.side_effect = [None, name_error, None]

        # Run and Assert
        error_msg = re.escape(
            'The provided constraint is invalid:'
            "\nThe name 'Positive' is a reserved constraint name. Please use a different one for "
            'the custom constraint.'
            "\n\nThe constraint 'CustomCons2' is not defined in 'example/myfile.py'."
        )
        with pytest.raises(InvalidConstraintsError, match=error_msg):
            DataProcessor._validate_custom_constraints(instance, filepath, class_names, module)

    @patch('sdv.data_processing.data_processor.load_module_from_path')
    def test_load_custom_constraint_classes(self, load_module_from_path_mock):
        """Test ``load_custom_constraint_classes``.

        Ensure that the method calls ``_validate_custom_constraints`` using the ``filepath`` and
        the ``class_names``. If this are valid, update ``instance._custom_constraint_classes`` with
        the ``class_name`` and the ``filepath`` from where this class can be loaded.
        """
        # Setup
        instance = Mock()
        instance._custom_constraint_classes = {}
        module_mock = Mock()
        custom_constraint_mock = Mock()
        simple_constraint_mock = Mock()
        module_mock.CustomCons = custom_constraint_mock
        module_mock.SimpleCons = simple_constraint_mock
        load_module_from_path_mock.return_value = module_mock

        # Run
        filepath = 'example/myfile.py'
        class_names = ['CustomCons', 'SimpleCons']
        DataProcessor.load_custom_constraint_classes(instance, filepath, class_names)

        # Assert
        instance._validate_custom_constraints.assert_called_once_with(
            'example/myfile.py',
            ['CustomCons', 'SimpleCons'],
            module_mock
        )
        assert instance._custom_constraint_classes == {
            'CustomCons': custom_constraint_mock,
            'SimpleCons': simple_constraint_mock
        }

    def test_add_custom_constraint_class(self):
        """Test that the class object is added to the ``_custom_constraint_classes`` dict."""
        # Setup
        instance = Mock()
        instance._custom_constraint_classes = {}
        custom_constraint = Mock()

        # Run
        DataProcessor.add_custom_constraint_class(instance, custom_constraint, 'custom')

        # Assert
        instance._validate_custom_constraint_name.assert_called_once_with('custom')
        assert instance._custom_constraint_classes == {'custom': custom_constraint}

    @patch('sdv.data_processing.data_processor.Constraint')
    def test__validate_constraint_dict(self, mock_constraint):
        """Ensure that the validation calls the ``Constraint`` class validation."""
        # Setup
        constraint_dict = {
            'constraint_class': 'Positive',
            'constraint_parameters': {'column_name': 'col1'}
        }
        positive_constraint = Mock()
        mock_constraint._get_class_from_dict.return_value = positive_constraint

        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        dp = DataProcessor(metadata)

        # Run
        dp._validate_constraint_dict(constraint_dict)

        # Assert
        positive_constraint._validate_metadata.assert_called_once_with(
            metadata, column_name='col1'
        )
        mock_constraint._get_class_from_dict.assert_called_once_with('Positive')

    def test__validate_constraint_dict_custom_constraint(self):
        """Test that the validation runs properly for a custom constraint class."""
        # Setup
        constraint_dict = {
            'constraint_class': 'CustomCons',
            'constraint_parameters': {'column_name': 'col1'}
        }
        custom_constraint = Mock()

        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        dp = DataProcessor(metadata)
        dp._custom_constraint_classes = {'CustomCons': custom_constraint}

        # Run
        dp._validate_constraint_dict(constraint_dict)

        # Assert
        custom_constraint._validate_metadata.assert_called_once_with(metadata, column_name='col1')

    @patch('sdv.data_processing.data_processor.Constraint')
    def test__validate_constraint_dict_key_error(self, mock_constraint):
        """Validate that an ``InvalidConstraintsError`` is raised when the class is not found."""
        # Setup
        constraint_dict = {
            'constraint_class': 'Positive',
            'constraint_parameters': {'column_name': 'col1'}
        }
        mock_constraint._get_class_from_dict.side_effect = [KeyError]

        metadata = SingleTableMetadata()
        dp = DataProcessor(metadata)

        # Run and Assert
        error_msg = re.escape("Invalid constraint class ('Positive').")
        with pytest.raises(InvalidConstraintsError, match=error_msg):
            dp._validate_constraint_dict(constraint_dict)

    def test_add_constraints(self):
        """Test that constraints are being added when those are valid."""
        # Setup
        constraints = [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col1'}
            },
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col2'}
            }
        ]

        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')
        metadata.add_column('col2', sdtype='numerical')
        dp = DataProcessor(metadata)

        # Run
        dp.add_constraints(constraints)

        # Assert
        assert dp._constraints_list == constraints
        assert id(dp._constraints_list) != id(constraints)

    def test_add_constraints_to_existing_list(self):
        """Test that constraints are being extended with the current ones when those are valid."""
        # Setup
        constraints = [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col1'}
            },
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col2'}
            }
        ]

        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')
        metadata.add_column('col2', sdtype='numerical')
        dp = DataProcessor(metadata)
        dp._constraints_list = [
            {
                'constraint_class': 'UniqueCombinations',
                'constraint_parameters': {'column_names': ['col1', 'col2']}
            }
        ]
        dp._prepared_for_fitting = True

        # Run
        dp.add_constraints(constraints)

        # Assert
        assert dp._constraints_list == [
            {
                'constraint_class': 'UniqueCombinations',
                'constraint_parameters': {'column_names': ['col1', 'col2']}
            },
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col1'}
            },
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'col2'}
            }
        ]
        assert id(dp._constraints_list) != id(constraints)
        assert dp._prepared_for_fitting is False

    def test_add_constraints_raises_invalidconstraintserror(self):
        """Test that constraints raises an ``InvalidConstraintsError`` when those are not valid."""
        # Setup
        constraints = [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {
                    'column_name': 'col1'
                }
            },
            {
                'constraint_class': 'Positiveee',
                'constraint_parameters': {
                    'column_name': 'col2'
                }
            }
        ]

        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        metadata.add_column('col2', sdtype='categorical')
        dp = DataProcessor(metadata)

        # Run and Assert
        error_msg = re.escape(
            'The provided constraint is invalid:\n'
            "A Positive constraint is being applied to an invalid column 'col1'. "
            'This constraint is only defined for numerical columns.\n\n'
            "Invalid constraint class ('Positiveee')."
        )
        with pytest.raises(InvalidConstraintsError, match=error_msg):
            dp.add_constraints(constraints)

    def test_add_constraints_missing_parameters(self):
        """Test error raised when required params are missing."""
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3]})
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        constraint = {'constraint_class': 'Inequality'}
        model = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "A constraint is missing required parameters {'constraint_parameters'}. "
            'Please add these parameters to your constraint definition.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            model.add_constraints([constraint])

    def test_add_constraints_invalid_parameters(self):
        """Test error raised when invalid params are passed."""
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3]})
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        constraint = {
            'constraint_class': 'Inequality',
            'constraint_parameters': {
                'low_column_name': 'col',
                'high_column_name': 'col'
            },
            'invalid': 42
        }
        model = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "Unrecognized constraint parameter {'invalid'}. "
            'Please remove these parameters from your constraint definition.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            model.add_constraints([constraint])

    def test_get_constraints(self):
        """Test that ``get_constraints`` returns a copy of the ``instance._constraints_list``."""
        # Setup
        instance = Mock()
        instance._constraints_list = [
            {
                'constraint_class': 'Positive',
                'constraint_parameters': {'column_name': 'a'}
            }
        ]

        # Run
        result = DataProcessor.get_constraints(instance)

        # Assert
        assert instance._constraints_list == result
        assert id(result) != id(instance._constraints_list)

    @patch('sdv.data_processing.data_processor.get_subclasses')
    @patch('sdv.data_processing.data_processor.Constraint')
    def test__load_constraints(self, constraint_mock, get_subclasses_mock):
        """Test the ``_load_constraints`` method.

        The method should take all the constraints in the passed metadata and
        call the ``Constraint.from_dict`` method on them if the constraints are the
        default provided by ``sdv``. If the constraints are custom constraints then
        it will used the stored class object.
        """
        # Setup
        get_subclasses_mock.return_value = {'Inequality', 'ScalarInequality'}
        data_processor = Mock()
        constraint1 = Mock()
        constraint2 = Mock()
        constraint1_dict = {
            'constraint_class': 'Inequality',
            'constraint_parameters': {
                'low_column_name': 'col1',
                'high_column_name': 'col2'
            }
        }
        constraint2_dict = {
            'constraint_class': 'ScalarInequality',
            'constraint_parameters': {
                'column_name': 'col1',
                'relation': '<',
                'value': 10
            }
        }
        custom_constraint_dict = {
            'constraint_class': 'CustomCons',
            'constraint_parameters': {
                'column_names': ['a', 'b']
            }
        }
        custom_constraint = Mock()

        data_processor._custom_constraint_classes = {'CustomCons': custom_constraint}
        constraint_mock.from_dict.side_effect = [constraint1, constraint2]

        data_processor._constraints_list = [
            constraint1_dict,
            constraint2_dict,
            custom_constraint_dict
        ]

        # Run
        loaded_constraints = DataProcessor._load_constraints(data_processor)

        # Assert
        assert loaded_constraints == [constraint1, constraint2, custom_constraint.return_value]
        custom_constraint.assert_called_once_with(column_names=['a', 'b'])

        constraint_mock.from_dict.assert_has_calls(
            [call(constraint1_dict), call(constraint2_dict)])

    def test__update_transformers_by_sdtypes(self):
        """Test that we update the ``_transformers_by_sdtype`` of the current instance."""
        # Setup
        instance = Mock()
        instance._transformers_by_sdtype = {
            'categorical': 'labelencoder',
            'numerical': 'float',
            'boolean': None
        }

        # Run
        DataProcessor._update_transformers_by_sdtypes(instance, 'categorical', None)

        # Assert
        assert instance._transformers_by_sdtype == {
            'categorical': None,
            'numerical': 'float',
            'boolean': None
        }

    @patch('sdv.data_processing.data_processor.rdt')
    def test_create_regex_generator_regex_generator(self, mock_rdt):
        """Test the ``create_regex_generator`` method.

        Test that when given an ``sdtype`` and ``column_metadata`` that contains ``regex_format``
        this creates and returns an instance of ``RegexGenerator``.

        Input:
            - String representing an ``sdtype``.
            - Dictionary with ``column_metadata`` that contains ``sdtype`` and ``regex_format``.

        Mock:
            - Mock ``rdt``.

        Output:
            - The return value of ``rdt.transformers.RegexGenerator``.
        """
        # Setup
        sdtype = 'id'
        column_metadata = {
            'sdtype': 'id',
            'regex_format': 'ID_00',
        }
        instance = Mock()
        instance._keys = ['ssn']

        # Run
        output = DataProcessor.create_regex_generator(
            instance,
            'ssn',
            sdtype,
            column_metadata,
            False
        )

        # Assert
        assert output == mock_rdt.transformers.RegexGenerator.return_value
        mock_rdt.transformers.RegexGenerator.assert_called_once_with(
            regex_format='ID_00',
            enforce_uniqueness=True
        )

    @patch('sdv.data_processing.data_processor.get_anonymized_transformer')
    def test_create_anonymized_transformer_enforce_uniqueness(self,
                                                              mock_get_anonymized_transformer):
        """Test the ``create_regex_generator`` method.

        Test that when given an ``sdtype`` and ``column_metadata`` that does not contain a
        ``regex_format`` this calls ``create_anonymized_transformer`` with ``enforce_uniqueness``
        set to ``True``.

        Input:
            - String representing an ``sdtype``.
            - Dictionary with ``column_metadata`` that contains ``sdtype``.

        Mock:
            - Mock the ``create_anonymized_transformer``.

        Output:
            - The return value of ``create_anonymized_transformer``.
        """
        # Setup
        sdtype = 'ssn'
        column_metadata = {
            'sdtype': 'ssn',
        }

        # Run
        output = DataProcessor.create_anonymized_transformer(
            sdtype,
            column_metadata,
            True
        )

        # Assert
        mock_get_anonymized_transformer.assert_called_once_with(
            'ssn', {'enforce_uniqueness': True, 'locales': None}
        )
        assert output == mock_get_anonymized_transformer.return_value

    @patch('sdv.data_processing.data_processor.get_anonymized_transformer')
    def test_create_anonymized_transformer_locales(self, mock_get_anonymized_transformer):
        """Test the ``create_anonymized_transformer`` method with locales.

        Test that when given an ``sdtype``, ``column_metadata``, and locales, this calls the
        ``get_anonymized_transformer`` with the ``locales`` keyword arg.
        """
        # Setup
        sdtype = 'ssn'
        column_metadata = {
            'sdtype': 'ssn',
        }

        # Run
        output = DataProcessor.create_anonymized_transformer(
            sdtype,
            column_metadata,
            False,
            locales=['en_US', 'en_CA']
        )

        # Assert
        mock_get_anonymized_transformer.assert_called_once_with(
            'ssn', {'locales': ['en_US', 'en_CA']}
        )
        assert output == mock_get_anonymized_transformer.return_value

    @patch('sdv.data_processing.data_processor.get_anonymized_transformer')
    def test_create_anonymized_transformer(self, mock_get_anonymized_transformer):
        """Test the ``create_anonymized_transformer`` method.

        Test that when given an ``sdtype`` and ``column_metadata`` this calls the
        ``get_anonymized_transformer`` with filtering the ``pii`` and ``sdtype`` keyword args.

        Input:
            - String representing an ``sdtype``.
            - Dictionary with ``column_metadata`` that contains ``sdtype`` and ``pii``.

        Mock:
            - Mock the ``get_anonymized_transformer``.

        Output:
            - The return value of ``get_anonymized_transformer``.
        """
        # Setup
        sdtype = 'email'
        column_metadata = {
            'sdtype': 'email',
            'pii': True,
            'function_kwargs': {'domain': 'gmail.com'}
        }

        # Run
        output = DataProcessor.create_anonymized_transformer(sdtype, column_metadata, False)

        # Assert
        assert output == mock_get_anonymized_transformer.return_value
        mock_get_anonymized_transformer.assert_called_once_with(
            'email', {'function_kwargs': {'domain': 'gmail.com'}, 'locales': None}
        )

    def test__get_transformer_instance_no_kwargs(self):
        """Test the ``_get_transformer_instance`` without keyword args.

        When there are no keyword args this will return a copy of a predefined transformer
        from the dictionary ``self._transformers_by_sdtype``.
        """
        # Setup
        dp = Mock()
        dp._transformers_by_sdtype = {
            'numerical': 'FloatFormatter',
            'categorical': 'LabelEncoder'
        }

        # Run
        result = DataProcessor._get_transformer_instance(dp, 'numerical', {})

        # Assert
        assert result == 'FloatFormatter'

    def test__get_transformer_instance_kwargs(self):
        """Test the ``_get_transformer_instance`` without keyword args.

        When there are extra (allowed) keyword args this will create an instance
        of a transformer with those.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())

        # Run
        result = dp._get_transformer_instance('numerical', {'computer_representation': 'Int32'})

        # Assert
        assert isinstance(result, FloatFormatter)
        assert result.computer_representation == 'Int32'

    @patch('sdv.data_processing.data_processor.LOGGER')
    @patch('sdv.data_processing.data_processor.rdt')
    def test__update_constraint_transformers(self, mock_rdt, mock_log):
        """Test that the transformers are being updated to the new columns in the data.

        Also ensure that we are informing the user that the transformers for given columns that
        have been dropped by the transformer are being removed.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())
        data = pd.DataFrame({
            'id_col': [1, 2],
            'pii_col': [1, 2],
            'low': [1, 2],
            'high': [1, 2],
            'map_col': ['z', 'x'],
            'map_col#cat_col': ['z#a', 'x#b'],
            'low#high': [0.2, 0.5]
        })
        dp._get_transformer_instance = Mock(return_value='LabelEncoder')
        mock_rdt.transformers.FloatFormatter.return_value = 'FloatFormatter'

        config = {
            'transformers': {
                'id_col': 'FloatFormatter',
                'pii_col': 'FloatFormatter',
                'low': 'FloatFormatter',
                'high': 'FloatFormatter',
                'cat_col': 'LabelEncoder',
                'map_col': 'LabelEncoder',
            },
            'sdtypes': {
                'id_col': 'numerical',
                'pii_col': 'pii',
                'low': 'numerical',
                'high': 'numerical',
                'cat_col': 'categorical',
                'map_col': 'categorical',
            }
        }
        columns_created_by_constraints = ['map_col#cat_col', 'low#high']

        # Run
        result = dp._update_constraint_transformers(data, columns_created_by_constraints, config)

        # Assert
        assert result == {
            'transformers': {
                'id_col': 'FloatFormatter',
                'pii_col': 'FloatFormatter',
                'low': 'FloatFormatter',
                'high': 'FloatFormatter',
                'map_col': 'LabelEncoder',
                'map_col#cat_col': 'LabelEncoder',
                'low#high': 'FloatFormatter'
            },
            'sdtypes': {
                'id_col': 'numerical',
                'pii_col': 'pii',
                'low': 'numerical',
                'high': 'numerical',
                'map_col': 'categorical',
                'map_col#cat_col': 'categorical',
                'low#high': 'numerical'
            }
        }
        mock_log.info.assert_called_once_with(
            "A constraint has dropped the column 'cat_col', removing the transformer from the "
            "'HyperTransformer'."
        )

    def test__create_config(self):
        """Test the ``_create_config`` method.

        The method should loop through the columns in the metadata and set the transformer
        for each column based on the sdtype. It should then loop through the columns in the
        ``columns_created_by_constraints`` list and either set them to use a ``FloatFormatter``
        or other transformer depending on their sdtype.

        Setup:
            - Create data with different column types.
            - Mock the metadata's columns.
        Input:
            - Data with columns both in the metadata and not.
            - columns_created_by_constraints as a list of the column names not in the metadata.

        Output:
            - The expected ``HyperTransformer`` config.
        """
        # Setup
        locales = ['en_US', 'en_CA', 'fr_CA']
        data = pd.DataFrame({
            'int': [1, 2, 3],
            'float': [1., 2., 3.],
            'bool': [True, False, True],
            'categorical': ['a', 'b', 'c'],
            'created_int': [4, 5, 6],
            'created_float': [4., 5., 6.],
            'created_bool': [False, True, False],
            'created_categorical': ['d', 'e', 'f'],
            'email': ['a@aol.com', 'b@gmail.com', 'c@gmx.com'],
            'first_name': ['John', 'Doe', 'Johanna'],
            'id': ['ID_001', 'ID_002', 'ID_003'],
            'date': ['2021-02-01', '2022-03-05', '2023-01-31']
        })
        dp = DataProcessor(SingleTableMetadata(), locales=locales)
        dp.metadata = Mock()
        dp.create_anonymized_transformer = Mock()
        dp.create_regex_generator = Mock()
        dp.create_anonymized_transformer.return_value = 'AnonymizedFaker'
        dp.create_regex_generator.return_value = 'RegexGenerator'
        dp.metadata.primary_key = 'id'
        dp._primary_key = 'id'
        dp._keys = ['id']
        dp.metadata.columns = {
            'int': {'sdtype': 'numerical'},
            'float': {'sdtype': 'numerical'},
            'bool': {'sdtype': 'boolean'},
            'categorical': {'sdtype': 'categorical'},
            'email': {'sdtype': 'email', 'pii': True},
            'first_name': {'sdtype': 'first_name'},
            'id': {'sdtype': 'id', 'regex_format': 'ID_\\d{3}[0-9]'},
            'date': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
        }

        # Run
        created_columns = {'created_int', 'created_float', 'created_bool', 'created_categorical'}
        config = dp._create_config(data, created_columns)

        # Assert
        assert config['sdtypes'] == {
            'int': 'numerical',
            'float': 'numerical',
            'bool': 'boolean',
            'categorical': 'categorical',
            'created_int': 'numerical',
            'created_float': 'numerical',
            'created_bool': 'boolean',
            'created_categorical': 'categorical',
            'email': 'pii',
            'first_name': 'pii',
            'id': 'text',
            'date': 'datetime'
        }

        int_transformer = config['transformers']['created_int']
        assert isinstance(int_transformer, FloatFormatter)
        assert int_transformer.missing_value_replacement == 'mean'
        assert int_transformer.missing_value_generation == 'random'

        float_transformer = config['transformers']['created_float']
        assert isinstance(float_transformer, FloatFormatter)
        assert float_transformer.missing_value_replacement == 'mean'
        assert float_transformer.missing_value_generation == 'random'

        assert isinstance(config['transformers']['bool'], LabelEncoder)
        assert isinstance(config['transformers']['created_bool'], LabelEncoder)
        assert isinstance(config['transformers']['categorical'], LabelEncoder)
        assert isinstance(config['transformers']['created_categorical'], LabelEncoder)

        assert isinstance(config['transformers']['int'], FloatFormatter)
        assert isinstance(config['transformers']['float'], FloatFormatter)
        anonymized_transformer = config['transformers']['email']
        primary_regex_generator = config['transformers']['id']
        assert anonymized_transformer == 'AnonymizedFaker'

        first_name_transformer = config['transformers']['first_name']
        assert first_name_transformer == 'AnonymizedFaker'
        assert primary_regex_generator == 'RegexGenerator'

        datetime_transformer = config['transformers']['date']
        assert isinstance(datetime_transformer, UnixTimestampEncoder)
        assert datetime_transformer.missing_value_replacement == 'mean'
        assert datetime_transformer.missing_value_generation == 'random'
        assert datetime_transformer.datetime_format == '%Y-%m-%d'
        assert dp._primary_key == 'id'

        dp.create_anonymized_transformer.calls == [
            call('email', {'sdtype': 'email', 'pii': True, 'locales': locales}),
            call('first_name', {'sdtype': 'first_name', 'locales': locales})

        ]
        dp.create_regex_generator.assert_called_once_with(
            'id',
            'id',
            {'sdtype': 'id', 'regex_format': 'ID_\\d{3}[0-9]'},
            False
        )

    def test_update_transformers_not_fitted(self):
        """Test when ``self._hyper_transformer`` is ``None`` raises a ``NotFittedError``."""
        # Setup
        dp = DataProcessor(SingleTableMetadata())

        # Run and Assert
        error_msg = (
            'The DataProcessor must be prepared for fitting before the transformers can be '
            'updated.'
        )
        with pytest.raises(NotFittedError, match=error_msg):
            dp.update_transformers({'column': None})

    def test_update_transformers_for_key(self):
        """Test when ``transformer`` is not ``AnonymizedFaker`` or ``RegexGenerator`` for keys."""
        # Setup
        dp = DataProcessor(SingleTableMetadata())
        dp._keys = ['pk_column', 'b']
        dp._hyper_transformer = Mock()

        # Run and Assert
        error_msg = re.escape(
            "Invalid transformer 'FloatFormatter' for a primary or alternate key 'pk_column'. "
            "Please use 'AnonymizedFaker' or 'RegexGenerator' instead."
        )
        with pytest.raises(SynthesizerInputError, match=error_msg):
            dp.update_transformers({'pk_column': FloatFormatter()})

    @patch('sdv.data_processing.data_processor.rdt.HyperTransformer')
    def test__fit_hyper_transformer(self, ht_mock):
        """Test the ``_fit_hyper_transformer`` method.

        The method should create a ``HyperTransformer``, create a config from the data and
        set the ``HyperTransformer's`` config to be what was created. Then it should fit the
        ``HyperTransformer`` on the data.

        Setup:
            - Patch the ``HyperTransformer``.
            - Mock the ``_create_config`` method.

        Input:
            - A dataframe.

        Side effects:
            - ``HyperTransformer`` should fit the data.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())
        ht_mock.return_value._fitted = False
        data = pd.DataFrame({'a': [1, 2, 3]})

        # Run
        dp._fit_hyper_transformer(data)

        # Assert
        ht_mock.return_value.fit.assert_called_once_with(data)

    @patch('sdv.data_processing.data_processor.rdt.HyperTransformer')
    def test__fit_hyper_transformer_empty_data(self, ht_mock):
        """Test the ``_fit_hyper_transformer`` method.

        If the data is empty, the ``HyperTransformer`` should not call fit.

        Setup:
            - Patch the ``HyperTransformer``.
            - Mock the ``_create_config`` method.

        Input:
            - An empty dataframe.

        Side effects:
            - ``HyperTransformer`` should not fit the data.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())
        ht_mock.return_value._fitted = False
        ht_mock.return_value.field_transformers = {}
        data = pd.DataFrame()

        # Run
        dp._fit_hyper_transformer(data)

        # Assert
        ht_mock.return_value.fit.assert_not_called()

    @patch('sdv.data_processing.data_processor.rdt.HyperTransformer')
    def test__fit_hyper_transformer_hyper_transformer_is_fitted(self, ht_mock):
        """Test when ``self._hyper_transformer`` is not ``None``.

        This should not re-fit or re-create the ``self._hyper_transformer``.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())
        dp._hyper_transformer = Mock()
        dp._hyper_transformer.field_transformers = {'name': 'categorical'}
        dp._hyper_transformer._fitted = True
        dp._create_config = Mock()
        data = pd.DataFrame({'name': ['John Doe']})

        # Run
        dp._fit_hyper_transformer(data)

        # Assert
        ht_mock.return_value.set_config.assert_not_called()
        ht_mock.return_value.fit.assert_not_called()
        dp._create_config.assert_not_called()

    @patch('sdv.data_processing.data_processor.rdt.HyperTransformer')
    def test__fit_hyper_transformer_hyper_transformer_is_fitted_and_modified_config(self, ht_mock):
        """Test when ``self._hyper_transformer._modified_config is True.

        Tests when both ``self._hyper_transformer._fitted`` and
        ``self._hyper_transformer._modified_config`` are ``True``. This should re-fit or re-create
        the ``self._hyper_transformer``.
        """
        # Setup
        dp = DataProcessor(SingleTableMetadata())
        ht_mock.return_value._fitted = True
        ht_mock.return_value._modified_config = True
        data = pd.DataFrame({'a': [1, 2, 3]})

        # Run
        dp._fit_hyper_transformer(data)

        # Assert
        ht_mock.return_value.fit.assert_called_once_with(data)

    @patch('sdv.data_processing.numerical_formatter.NumericalFormatter.learn_format')
    def test__fit_formatters(self, learn_format_mock):
        """Test the ``_fit_formatters`` method.

        Runs the methods through three columns: a non-numerical column, which should
        be skipped by the method, and two numerical ones (with different values for
        ``computer_representation``), which should create and learn a ``NumericalFormatter``.

        Setup:
            - ``SingleTableMetadata`` describing the three columns.
            - A mock of ``NumericalFormatter.learn_format``.
        """
        # Setup
        data = pd.DataFrame({
            'col1': ['abc', 'def'],
            'col2': [1, 2],
            'col3': [3, 4],
            'date_col1': ['16-05-2023', '14-04-2022'],
            'date_col2': pd.to_datetime(['2021-02-15', '2022-05-16']),
        })
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        metadata.add_column('col2', sdtype='numerical')
        metadata.add_column('col3', sdtype='numerical', computer_representation='Int8')
        metadata.add_column('date_col1', sdtype='datetime')
        metadata.add_column('date_col2', sdtype='datetime', datetime_format='%Y-%d-%M')
        dp = DataProcessor(metadata, enforce_rounding=False, enforce_min_max_values=False)

        # Run
        dp._fit_formatters(data)

        # Assert
        assert list(dp.formatters.keys()) == ['col2', 'col3', 'date_col1', 'date_col2']

        assert isinstance(dp.formatters['col2'], NumericalFormatter)
        assert dp.formatters['col2'].enforce_rounding is False
        assert dp.formatters['col2'].enforce_min_max_values is False
        assert dp.formatters['col2'].computer_representation == 'Float'

        assert isinstance(dp.formatters['col3'], NumericalFormatter)
        assert dp.formatters['col3'].enforce_rounding is False
        assert dp.formatters['col3'].enforce_min_max_values is False
        assert dp.formatters['col3'].computer_representation == 'Int8'

        learn_format_mock.assert_has_calls([call(data['col2']), call(data['col3'])])

        assert isinstance(dp.formatters['date_col1'], DatetimeFormatter)
        assert isinstance(dp.formatters['date_col2'], DatetimeFormatter)
        assert dp.formatters['date_col1']._dtype == 'O'
        assert dp.formatters['date_col1'].datetime_format == '%d-%m-%Y'
        assert dp.formatters['date_col2']._dtype == '<M8[ns]'
        assert dp.formatters['date_col2'].datetime_format == '%Y-%d-%M'

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_prepare_for_fitting(self, log_mock):
        """Test the steps before fitting.

        Test that ``dtypes``, numerical formatters and constraints are being fitted before
        creating the configuration for the ``rdt.HyperTransformer``.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]}, dtype=np.int64)
        transformed_data = pd.DataFrame({'a': [4, 5, 6], 'b': [1, 2, 3]})
        dp = Mock()
        dp.table_name = 'fake_table'
        dp._transform_constraints.return_value = transformed_data
        dp._prepared_for_fitting = False
        dp._hyper_transformer.get_config.return_value = {'sdtypes': {}, 'transformers': {}}
        dp._constraints_list = [1, 2, 3]
        dp._constraints = []

        # Run
        DataProcessor.prepare_for_fitting(dp, data)

        # Assert
        pd.testing.assert_series_equal(dp._dtypes, pd.Series([np.int64], index=['a']))
        dp._transform_constraints.assert_called_once_with(data)
        dp._fit_constraints.assert_called_once_with(data)
        dp._fit_formatters.assert_called_once_with(data)
        dp._hyper_transformer.set_config.assert_called_with(dp._create_config.return_value)
        fitting_call = call('Fitting table fake_table metadata')
        formatter_call = call('Fitting formatters for table fake_table')
        constraint_call = call('Fitting constraints for table fake_table')
        setting_config_call = call(
            'Setting the configuration for the ``HyperTransformer`` for table fake_table')
        log_mock.info.assert_has_calls([
            fitting_call,
            formatter_call,
            constraint_call,
            setting_config_call
        ])

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_prepare_for_fitting_config_already_exists(self, log_mock):
        """Test the steps before fitting.

        Test that ``dtypes``, numerical formatters and constraints are being fitted. If the config
        already exists, it doesn't need to be created again.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]}, dtype=np.int64)
        transformed_data = pd.DataFrame({'column': [4, 5, 6]})
        dp = Mock()
        dp.table_name = 'fake_table'
        dp._constraints_list = []
        dp._constraints = []
        dp._transform_constraints.return_value = transformed_data
        dp._prepared_for_fitting = False
        dp._hyper_transformer.get_config.return_value = {
            'sdtypes': {'column': 'numerical'},
            'transformers': {'column': Mock()}
        }

        # Run
        DataProcessor.prepare_for_fitting(dp, data)

        # Assert
        pd.testing.assert_series_equal(dp._dtypes, pd.Series([np.int64], index=['a']))
        dp._transform_constraints.assert_called_once_with(data)
        dp._fit_formatters.assert_called_once_with(data)
        dp._hyper_transformer.set_config.assert_not_called()
        fitting_call = call('Fitting table fake_table metadata')
        formatter_call = call('Fitting formatters for table fake_table')
        constraint_call = call('Fitting constraints for table fake_table')
        log_mock.info.assert_has_calls([
            fitting_call,
            formatter_call,
            constraint_call
        ])

    @patch('sdv.data_processing.data_processor.rdt')
    def test_prepare_for_fitting_config_already_exists_new_constraint_columns(self, mock_rdt):
        """Test ``prepare_for_fitting`` after new constraints have been added.

        Ensure that this process will call ``self._update_constraint_transformers`` with the
        new missing columns that have been generated by the constraints.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]}, dtype=np.int64)
        transformed_data = pd.DataFrame({
            'column': [4, 5, 6],
            'column#a': [1., 2., 3.]
        })
        dp = Mock()
        dp.table_name = 'fake_table'
        dp._transform_constraints.return_value = transformed_data
        dp._prepared_for_fitting = False
        dp._constraints_list = [1, 2, 3]
        dp._constraints = [1, 2, 3]
        config = {
            'sdtypes': {'column': 'numerical'},
            'transformers': {'column': Mock()}
        }
        dp._hyper_transformer.get_config.return_value = config

        # Run
        DataProcessor.prepare_for_fitting(dp, data)

        # Assert
        pd.testing.assert_series_equal(dp._dtypes, pd.Series([np.int64], index=['a']))
        dp._fit_constraints.assert_not_called()
        dp._transform_constraints.assert_called_once_with(data)
        dp._fit_formatters.assert_called_once_with(data)
        dp._update_constraint_transformers.assert_called_once_with(
            transformed_data,
            {'column#a'},
            config
        )
        assert dp._hyper_transformer == mock_rdt.HyperTransformer.return_value
        dp._hyper_transformer.set_config.assert_called_once_with(
            dp._update_constraint_transformers.return_value)

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_prepare_for_fitting_already_prepared(self, log_mock):
        """Test that if the preparation has already been done, it doesn't do it again."""
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]}, dtype=np.int64)
        dp = Mock()
        dp._constraints_list = []
        dp._constraints = []
        dp._prepared_for_fitting = True

        # Run
        DataProcessor.prepare_for_fitting(dp, data)

        # Assert
        dp._fit_constraints.assert_not_called()
        dp._fit_formatters.assert_not_called()
        log_mock.info.assert_not_called()

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_fit(self, log_mock):
        """Test the ``fit`` method.

        The ``fit`` method should store the dtypes, learn the formatters for each column,
        fit and transform the constraints and then fit the ``HyperTransformer``.

        Setup:
            - Mock the ``prepare_for_fitting`` method.

        Input:
            - A ``pandas.DataFrame``.

        Side effect:
            - The ``_fit_hyper_transformer`` should be called.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]}, dtype=np.int64)
        transformed_data = pd.DataFrame({'a': [4, 5, 6], 'b': [1, 2, 3]})
        dp = Mock()
        dp.table_name = 'fake_table'
        dp._transform_constraints.return_value = transformed_data

        # Run
        DataProcessor.fit(dp, data)

        # Assert
        dp.prepare_for_fitting.assert_called_once_with(data)
        dp._transform_constraints.assert_called_once_with(data)
        dp._fit_hyper_transformer.assert_called_once_with(transformed_data)
        log_mock.info.assert_called_once_with('Fitting HyperTransformer for table fake_table')

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_transform(self, log_mock):
        """Test the ``transform`` method.

        The method should call the ``_transform_constraints`` and
        ``HyperTransformer.transform_subset``.

        Input:
            - Table data.

        Side Effects:
            - Calls ``_transform_constraints``.
            - Calls ``HyperTransformer.transform_subset``.
            - Calls logger with right messages.
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        dp = DataProcessor(SingleTableMetadata(), table_name='table_name')
        dp._transform_constraints = Mock()
        dp._transform_constraints.return_value = data
        dp._hyper_transformer = Mock()
        dp.get_sdtypes = Mock()
        dp.get_sdtypes.return_value = {
            'item 0': 'numerical',
            'item 1': 'boolean'
        }
        dp._hyper_transformer.transform_subset.return_value = data
        dp.fitted = True

        # Run
        dp.transform(data)

        # Assert
        expected_data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        constraint_mock_calls = dp._transform_constraints.mock_calls
        ht_mock_calls = dp._hyper_transformer.transform_subset.mock_calls
        constraint_data, is_condition = constraint_mock_calls[0][1]
        ht_data = ht_mock_calls[0][1][0]
        assert len(constraint_mock_calls) == 1
        assert is_condition is False
        pd.testing.assert_frame_equal(constraint_data, expected_data)
        assert len(ht_mock_calls) == 1
        pd.testing.assert_frame_equal(ht_data, expected_data)
        constraint_call = call('Transforming constraints for table table_name')
        transformer_call = call('Transforming table table_name')
        log_mock.debug.assert_has_calls([constraint_call, transformer_call])

    def test_generate_keys(self):
        """Test the ``genereate_primary_keys``.

        Test that when calling this function this calls the ``instance._hyper_transformer``'s
        ``create_anonymized_columns`` method with the ``num_rows`` and the
        ``instance._primary_keys``.

        Setup:
            - Mock the instance of ``DataProcessor``.
            - Set some ``_primary_keys``.

        Input:
            - ``num_rows``

        Side Effects:
            - ``instance._hyper_transformer.create_anonymized_columns`` has been called with the
              input number and ``column_names`` same as the ``instance._primary_keys``.

        Output:
            - The output should be the return value of the
              ``instance._hyper_transformer.create_anonymized_columns``.
        """
        # Setup
        instance = Mock()
        instance._primary_key = 'a'
        instance._hyper_transformer.field_transformers = {
            'a': object()
        }
        instance._keys = ['a']

        # Run
        result = DataProcessor.generate_keys(instance, 10)

        # Assert
        instance._hyper_transformer.create_anonymized_columns.assert_called_once_with(
            num_rows=10,
            column_names=['a'],
        )

        assert result == instance._hyper_transformer.create_anonymized_columns.return_value

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_transform_primary_key(self, log_mock):
        """Test the ``transform`` method.

        The method should call the ``_transform_constraints`` and
        ``HyperTransformer.transform_subset``.

        Input:
            - Table data.

        Side Effects:
            - Calls ``_transform_constraints``.
            - Calls ``HyperTransformer.transform_subset``.
            - Calls logger with right messages.
        """
        # Setup
        data = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'item 0': [0, 1, 2],
            'item 1': [True, True, False],
        }, index=[0, 1, 2])
        dp = DataProcessor(SingleTableMetadata(), table_name='table_name')
        dp._transform_constraints = Mock()
        dp._transform_constraints.return_value = data
        dp._hyper_transformer = Mock()
        dp._hyper_transformer.transform_subset.return_value = data
        dp._hyper_transformer.field_transformers = {'id': object()}
        dp.get_sdtypes = Mock()
        dp.get_sdtypes.return_value = {
            'id': 'categorical',
            'item 0': 'numerical',
            'item 1': 'boolean'
        }

        dp.fitted = True
        dp._primary_key = 'id'

        primary_key_data = pd.DataFrame({'id': ['a', 'b', 'c']})
        dp._hyper_transformer.create_anonymized_columns.return_value = primary_key_data

        # Run
        transformed = dp.transform(data)

        # Assert
        expected_data = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])

        constraint_mock_calls = dp._transform_constraints.mock_calls
        ht_mock_calls = dp._hyper_transformer.transform_subset.mock_calls
        constraint_data, is_condition = constraint_mock_calls[0][1]
        assert len(constraint_mock_calls) == 1
        assert is_condition is False
        assert len(ht_mock_calls) == 1

        constraint_call = call('Transforming constraints for table table_name')
        transformer_call = call('Transforming table table_name')
        log_mock.debug.assert_has_calls([constraint_call, transformer_call])

        pd.testing.assert_frame_equal(constraint_data, data)
        pd.testing.assert_frame_equal(transformed, expected_data)

    def test_transform_not_fitted(self):
        """Test the ``transform`` method if the ``DataProcessor`` was not fitted.

        The method should raise a ``NotFittedError``.

        Setup:
            - Set ``fitted`` to False.

        Input:
            - Table data.

        Side Effects:
            - Raises ``NotFittedError``.
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        dp = DataProcessor(SingleTableMetadata(), table_name='table_name')

        # Run and Assert
        with pytest.raises(NotFittedError):
            dp.transform(data)

    def test_transform_hyper_transformer_errors(self):
        """Test the ``transform`` method when ``HyperTransformer`` errors.

        The method should catch the error raised by the ``HyperTransformer`` and return
        the data unchanged.

        Input:
            - Table data.

        Output:
            - Same data.
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        dp = DataProcessor(SingleTableMetadata(), table_name='table_name')
        dp._transform_constraints = Mock()
        dp._transform_constraints.return_value = data
        dp._hyper_transformer = Mock()
        dp._hyper_transformer.transform_subset.side_effect = ConfigNotSetError()
        dp.get_sdtypes = Mock()
        dp.get_sdtypes.return_value = {
            'item 0': 'numerical',
            'item 1': 'boolean'
        }
        dp.fitted = True

        # Run
        transformed_data = dp.transform(data)

        # Assert
        expected_data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        constraint_mock_calls = dp._transform_constraints.mock_calls
        constraint_data, is_condition = constraint_mock_calls[0][1]
        assert len(constraint_mock_calls) == 1
        assert is_condition is False
        pd.testing.assert_frame_equal(constraint_data, expected_data)
        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test__transform_constraints(self):
        """Test that ``_transform_constraints`` correctly transforms data based on constraints.

        The method is expected to loop through constraints and call each constraint's ``transform``
        method on the data.

        Input:
            - Table data.
        Output:
            - Transformed data.
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
        dp = DataProcessor(SingleTableMetadata())
        dp._constraints = [first_constraint_mock, second_constraint_mock]

        # Run
        result = dp._transform_constraints(data)

        # Assert
        assert result.equals(transformed_data)
        first_constraint_mock.transform.assert_called_once_with(data)
        second_constraint_mock.transform.assert_called_once_with(transformed_data)
        assert dp._constraints_to_reverse == [
            first_constraint_mock,
            second_constraint_mock
        ]

    def test__transform_constraints_is_condition_drops_columns(self):
        """Test that ``_transform_constraints`` drops columns when necessary.

        The method is expected to drop columns associated with a constraint when its
        transform raises a ``MissingConstraintColumnError`` and the ``is_condition``
        flag is True.

        Input:
            - Table data.
            - ``is_condition`` set to True.
        Output:
            - Table with dropped columns.
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.transform.side_effect = MissingConstraintColumnError(missing_columns=[])
        constraint_mock.constraint_columns = ['item 0']
        dp = DataProcessor(SingleTableMetadata())
        dp._constraints = [constraint_mock]
        dp._constraints_to_reverse = [constraint_mock]

        # Run
        result = dp._transform_constraints(data, True)

        # Assert
        expected_result = pd.DataFrame({
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        assert result.equals(expected_result)
        assert dp._constraints_to_reverse == [constraint_mock]

    def test__transform_constraints_is_condition_false_returns_data(self):
        """Test that ``_transform_constraints`` returns data unchanged when necessary.

        The method is expected to return data unchanged when the constraint transform
        raises a ``MissingConstraintColumnError`` and the ``is_condition`` flag is False.

        Input:
            - Table data.
        Output:
            - Table with dropped columns.
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        constraint_mock = Mock()
        constraint_mock.transform.side_effect = MissingConstraintColumnError(missing_columns=[])
        constraint_mock.constraint_columns = ['item 0']
        dp = DataProcessor(SingleTableMetadata())
        dp._constraints = [constraint_mock]
        dp._constraints_to_reverse = [constraint_mock]

        # Run
        result = dp._transform_constraints(data, False)

        # Assert
        expected_result = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [3, 4, 5]
        }, index=[0, 1, 2])
        assert result.equals(expected_result)
        assert dp._constraints_to_reverse == []

    def test_reverse_transform(self):
        """Test the ``reverse_transform`` method.

        This method should attempt to reverse transform all the columns using the
        ``HyperTransformer``. Then it should loop through the constraints and reverse
        transform all of them. Finally, it should cast all the columns to their original
        dtypes.

        Setup:
            - Mock the ``HyperTransformer``.
            - Set the ``_constraints_to_reverse`` to contain a mock constraint.
            - Set ``fitted`` to True.
            - Mock the ``_dtypes``.

        Input:
            - A dataframe.

        Output:
            - The reverse transformed data.
        """
        # Setup
        constraint_mock = Mock()
        dp = DataProcessor(SingleTableMetadata())
        dp.fitted = True
        dp.metadata = Mock()
        dp.metadata.columns = {'a': None, 'b': None, 'c': None, 'd': None}
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [True, True, False],
            'c': ['d', 'e', 'f'],
        })
        dp._hyper_transformer = Mock()
        dp._hyper_transformer.create_anonymized_columns.return_value = pd.DataFrame({
            'd': ['a@gmail.com', 'b@gmail.com', 'c@gmail.com']
        })
        dp._constraints_to_reverse = [constraint_mock]
        dp._hyper_transformer.reverse_transform_subset.return_value = data
        dp._hyper_transformer._output_columns = ['a', 'b', 'c']
        dp._dtypes = pd.Series(
            [np.float64, np.bool_, np.object_, np.object_], index=['a', 'b', 'c', 'd'])
        constraint_mock.reverse_transform.return_value = data

        # Run
        reverse_transformed = dp.reverse_transform(data)

        # Assert
        input_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [True, True, False],
            'c': ['d', 'e', 'f']
        })
        constraint_mock.reverse_transform.assert_called_once_with(data)
        data_from_call = dp._hyper_transformer.reverse_transform_subset.mock_calls[0][1][0]
        pd.testing.assert_frame_equal(input_data, data_from_call)
        dp._hyper_transformer.reverse_transform_subset.assert_called_once()
        expected_output = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [True, True, False],
            'c': ['d', 'e', 'f'],
            'd': ['a@gmail.com', 'b@gmail.com', 'c@gmail.com']
        })
        dp._hyper_transformer.create_anonymized_columns.assert_called_once_with(
            num_rows=3,
            column_names=['d']
        )
        pd.testing.assert_frame_equal(reverse_transformed, expected_output)

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_reverse_transform_hyper_transformer_errors(self, log_mock):
        """Test the ``reverse_transform`` method.

        A message should be logged if the ``HyperTransformer`` errors.

        Setup:
            - Patch the logger.
            - Mock the ``HyperTransformer``.
            - Set the ``_constraints_to_reverse`` to contain a mock constraint.
            - Set ``fitted`` to True.
            - Mock the ``_dtypes``.

        Input:
            - A dataframe.

        Output:
            - The reverse transformed data.
        """
        # Setup
        constraint_mock = Mock()
        dp = DataProcessor(SingleTableMetadata(), table_name='table_name')
        dp.fitted = True
        dp.metadata = Mock()
        dp.metadata.columns = {'a': None, 'b': None, 'c': None}
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [True, True, False],
            'c': ['d', 'e', 'f']
        })
        dp._hyper_transformer = Mock()
        dp._constraints_to_reverse = [constraint_mock]
        dp._hyper_transformer.reverse_transform_subset.side_effect = RDTNotFittedError
        dp._hyper_transformer._output_columns = ['a', 'b', 'c']
        dp._dtypes = pd.Series([np.float64, np.bool_, np.object_], index=['a', 'b', 'c'])
        constraint_mock.reverse_transform.return_value = data

        # Run
        reverse_transformed = dp.reverse_transform(data)

        # Assert
        input_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [True, True, False],
            'c': ['d', 'e', 'f']
        })
        constraint_mock.reverse_transform.assert_called_once_with(data)
        data_from_call = dp._hyper_transformer.reverse_transform_subset.mock_calls[0][1][0]
        message = 'HyperTransformer has not been fitted for table table_name'
        log_mock.info.assert_called_with(message)
        pd.testing.assert_frame_equal(input_data, data_from_call)
        expected_output = pd.DataFrame({
            'a': [1., 2., 3.],
            'b': [True, True, False],
            'c': ['d', 'e', 'f']
        })
        pd.testing.assert_frame_equal(reverse_transformed, expected_output)

    def test_reverse_transform_not_fitted(self):
        """Test the ``reverse_transform`` method if the ``DataProcessor`` was not fitted.

        The method should raise a ``NotFittedError``.

        Setup:
            - Set ``fitted`` to False.

        Input:
            - Table data.

        Side Effects:
            - Raises ``NotFittedError``.
        """
        # Setup
        data = pd.DataFrame({
            'item 0': [0, 1, 2],
            'item 1': [True, True, False]
        }, index=[0, 1, 2])
        dp = DataProcessor(SingleTableMetadata(), table_name='table_name')

        # Run and Assert
        with pytest.raises(NotFittedError):
            dp.reverse_transform(data)

    def test_reverse_transform_integer_rounding(self):
        """Test the ``reverse_transform`` method correctly rounds.

        Expect the data to be rounded when the ``dtypes`` specifies
        the ``'dtype'`` as ``'integer'``.

        Input:
            - A dataframe.
        Output:
            - The input dictionary rounded.
        """
        # Setup
        data = pd.DataFrame({'bar': [0.2, 1.7, 2]})
        dp = DataProcessor(SingleTableMetadata())
        dp.fitted = True
        dp._hyper_transformer = Mock()
        dp._hyper_transformer._output_columns = []
        dp._hyper_transformer.reverse_transform_subset.return_value = data
        dp._constraints_to_reverse = []
        dp._dtypes = {'bar': 'int'}
        dp.metadata = Mock()
        dp.metadata.columns = {'bar': None}

        # Run
        output = dp.reverse_transform(data)

        # Assert
        expected_data = pd.DataFrame({'bar': [0, 2, 2]})
        pd.testing.assert_frame_equal(output, expected_data, check_dtype=False)

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_reverse_transform_pii_numerical_column(self, mock_logger):
        """Test the ``reverse_transform`` method doesn't break if PII value was set.

        If a ``PII`` value was set to a ``numerical`` value from the dataframe, which will
        result into ``self._dtypes`` to have it as ``numerical``, this should inform the
        user that the value will not be set as numerical.
        """
        # Setup
        data = pd.DataFrame({'bar': ['a', 'b', 'c']})
        dp = DataProcessor(SingleTableMetadata())
        dp.fitted = True
        dp._hyper_transformer = Mock()
        dp._hyper_transformer._output_columns = []
        dp._hyper_transformer.reverse_transform_subset.return_value = data
        dp._constraints_to_reverse = []
        dp._dtypes = {'bar': 'int'}
        dp.metadata = Mock()
        dp.metadata.columns = {'bar': {'sdtype': 'address'}}
        dp.formatters = {'bar': object()}

        # Run
        output = dp.reverse_transform(data)

        # Assert
        expected_data = pd.DataFrame({'bar': ['a', 'b', 'c']})
        pd.testing.assert_frame_equal(output, expected_data)
        message = (
            "The real data in 'bar' was stored as 'int' but the synthetic data "
            'could not be cast back to this type. If this is a problem, please check your input '
            'data and metadata settings.'
        )
        mock_logger.info.assert_called_with(message)
        assert dp.formatters == {}

    @patch('sdv.data_processing.data_processor.LOGGER')
    def test_reverse_transform_value_error_is_raised(self, mock_logger):
        """Test the ``reverse_transform`` method raises a ``ValueError``.

        If ``ValueError`` is raised while trying to set the data as type ``dtype``, if this is
        not ``PII`` column, a value error should be raised.
        """
        # Setup
        data = pd.DataFrame({'bar': ['a', 'b', 'c']})
        dp = DataProcessor(SingleTableMetadata())
        dp.fitted = True
        dp._hyper_transformer = Mock()
        dp._hyper_transformer._output_columns = []
        dp._hyper_transformer.reverse_transform_subset.return_value = data
        dp._constraints_to_reverse = []
        dp._dtypes = {'bar': 'int'}
        dp.metadata = Mock()
        dp.metadata.columns = {'bar': {'sdtype': 'numerical'}}
        dp.formatters = {'bar': object()}

        # Run and Assert
        error_msg = re.escape("invalid literal for int() with base 10: 'a'")
        with pytest.raises(ValueError, match=error_msg):
            dp.reverse_transform(data)

        mock_logger.info.assert_not_called()

    @patch('sdv.data_processing.numerical_formatter.NumericalFormatter')
    @patch('sdv.data_processing.numerical_formatter.NumericalFormatter')
    def test_reverse_transform_numerical_formatter(self, formatter_mock1, formatter_mock2):
        """Test the ``reverse_transform`` correctly applies the ``NumericalFormatter``.

        Runs the method through three columns: a non-numerical column, which should
        be skipped by the method, and two numerical ones which should call the
        ``NumericalFormatter.format_data`` method with a column each and return them
        unchanged.

        Setup:
            - ``SingleTableMetadata`` describing the three columns.
            - Two mocks of ``NumericalFormatter``, one for each numerical column,
            with the appropriate return value for the ``format_data`` method.
            - ``formatters`` attribute should have a dict of the two numerical columns
            mapped to the two mocked ``NumericalFormatters``.
        """
        # Setup
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': ['abc', 'def']})
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')
        metadata.add_column('col2', sdtype='numerical')
        metadata.add_column('col3', sdtype='categorical')

        dp = DataProcessor(metadata)
        dp.formatters = {'col1': formatter_mock1, 'col2': formatter_mock2}
        formatter_mock1.format_data.return_value = np.array([1, 2])
        formatter_mock2.format_data.return_value = np.array([3, 4])

        # Unrelated setup, required so the method doesn't crash
        dp._hyper_transformer = Mock()
        dp._hyper_transformer._output_columns = []
        dp._hyper_transformer.reverse_transform_subset.return_value = data
        dp._dtypes = {'col1': 'int', 'col2': 'int', 'col3': 'str'}
        dp.fitted = True

        # Run
        output = dp.reverse_transform(data)

        # Assert
        formatter_mock1.format_data.assert_called_once()
        np.testing.assert_array_equal(
            formatter_mock1.format_data.call_args[0][0], data['col1'].to_numpy())

        formatter_mock2.format_data.assert_called_once()
        np.testing.assert_array_equal(
            formatter_mock2.format_data.call_args[0][0], data['col2'].to_numpy())

        pd.testing.assert_frame_equal(output, data)

    def test_reverse_transform_datetime_formatter(self):
        """Test that the ``reverse_transform`` calls the ``DatetimeFormatter``.

        Iterate over the ``instance.formatters`` and ensure that applies the
        expected formatting.
        """
        # Setup
        data = pd.DataFrame({
            'col1': ['abc', 'def'],
            'col2': ['16-05-2023', '14-04-2022'],
            'col3': pd.to_datetime(['2021-02-15', '2022-05-16']),
        })
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='categorical')
        metadata.add_column('col2', sdtype='datetime')
        metadata.add_column('col3', sdtype='datetime', datetime_format='%Y-%d-%M')
        dp = DataProcessor(metadata)

        col2_datetime_formatter = Mock()
        col3_datetime_formatter = Mock()
        dp.formatters = {'col2': col2_datetime_formatter, 'col3': col3_datetime_formatter}

        dp._hyper_transformer = Mock()
        dp._hyper_transformer._output_columns = []
        dp._hyper_transformer.reverse_transform_subset.return_value = data
        dp.fitted = True
        dp._dtypes = {'col1': 'str', 'col2': 'str', 'col3': 'str'}

        # Run
        dp.reverse_transform(data)

        # Assert
        col2_datetime_formatter.format_data.assert_called_once()
        col3_datetime_formatter.format_data.assert_called_once()
