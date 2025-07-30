import logging
import os
import re
import warnings
from datetime import date, datetime
from unittest.mock import ANY, MagicMock, Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from copulas.multivariate import GaussianMultivariate
from rdt.transformers import (
    BinaryEncoder,
    FloatFormatter,
    GaussianNormalizer,
    OneHotEncoder,
    RegexGenerator,
)

from sdv import version
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag.programmable_constraint import SingleTableProgrammableConstraint
from sdv.errors import (
    ConstraintsNotMetError,
    InvalidDataError,
    SamplingError,
    SynthesizerInputError,
    VersionError,
)
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata import Metadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sampling.tabular import Condition, DataFrameCondition
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)
from sdv.single_table.base import COND_IDX, BaseSingleTableSynthesizer, BaseSynthesizer
from tests.utils import catch_sdv_logs


class TestBaseSynthesizer:
    def test__update_default_transformers(self):
        """Test that ``instance._data_processor._update_transformers_by_sdtypes`` is called.

        Test when there are ``_model_sdtype_transformers`` set, this method will call
        the data processor and update the default ones.
        """
        # Setup
        instance = Mock()
        instance._model_sdtype_transformers = {'categorical': None, 'numerical': 'FloatTransformer'}

        # Run
        BaseSynthesizer._update_default_transformers(instance)

        # Assert
        call_list = instance._data_processor._update_transformers_by_sdtypes.call_args_list
        assert call_list == [call('categorical', None), call('numerical', 'FloatTransformer')]

    def test__check_metadata_updated(self):
        """Test the ``_check_metadata_updated`` method."""
        # Setup
        instance = Mock()
        instance.metadata = Mock()
        instance.metadata._check_updated_flag.return_value = True
        instance._input_metadata = Mock()

        # Run
        expected_message = re.escape(
            "We strongly recommend saving the metadata using 'save_to_json' for replicability"
            ' in future SDV versions.'
        )
        with pytest.warns(UserWarning, match=expected_message):
            BaseSynthesizer._check_metadata_updated(instance)

        # Assert
        instance.metadata._reset_updated_flag.assert_called_once_with()
        instance.metadata._check_updated_flag.assert_called_once_with()
        instance._input_metadata._reset_updated_flag.assert_called_once_with()

    def test__check_metadata_updated__input_metadata_is_single_table(self):
        """Test the ``_check_metadata_updated`` method when input metadata is SingleTable."""
        # Setup
        instance = Mock()
        instance.metadata = Mock()
        instance.metadata._check_updated_flag.return_value = True
        instance._input_metadata = SingleTableMetadata()

        # Run
        expected_message = re.escape(
            "We strongly recommend saving the metadata using 'save_to_json' for replicability"
            ' in future SDV versions.'
        )
        with pytest.warns(UserWarning, match=expected_message):
            BaseSynthesizer._check_metadata_updated(instance)

        # Assert
        instance.metadata._reset_updated_flag.assert_called_once_with()
        instance.metadata._check_updated_flag.assert_called_once_with()
        assert instance._input_metadata._updated is False

    def test__check_input_metadata_updated_warns_when_updated(self):
        """Test the `_check_input_metadata_updated` method raises a warning."""
        # Setup
        instance = Mock()
        metadata_instance = Mock()
        metadata_instance._convert_to_single_table.return_value = metadata_instance
        metadata_instance._updated = True

        instance.metadata.to_dict.return_value = {'some': 'data'}
        instance._input_metadata = metadata_instance

        expected_message = re.escape(
            'Your metadata has been modified. Metadata modifications cannot be applied to an '
            'existing synthesizer. Please create a new synthesizer with the modified metadata.'
        )

        # Run and Assert
        with pytest.warns(UserWarning, match=expected_message):
            BaseSynthesizer._check_input_metadata_updated(instance)

    def test__check_input_metadata_updated_does_not_warn_if_not_updated(self):
        """Test `_check_input_metadata_updated` does not warn if metadata is not updated."""
        # Setup
        instance = Mock()
        metadata_instance = Mock()
        metadata_instance._convert_to_single_table.return_value = metadata_instance
        metadata_instance._updated = False

        instance.metadata.to_dict.return_value = {'some': 'data'}
        instance._input_metadata = metadata_instance

        # Run and Assert
        with warnings.catch_warnings(record=True) as raised_warnings:
            BaseSynthesizer._check_input_metadata_updated(instance)
            assert not raised_warnings

    @patch('sdv.metadata.Metadata.load_from_dict')
    def test__check_input_metadata_updated_sets_original_metadata_if_missing(
        self, mock_load_from_dict
    ):
        """Test `_check_input_metadata_updated` sets _original_metadata if not present."""
        # Setup
        instance = Mock()
        instance.metadata.to_dict.return_value = {'some': 'data'}
        del instance._input_metadata

        loaded_metadata = Mock()
        loaded_metadata._convert_to_single_table.return_value = loaded_metadata
        loaded_metadata._updated = False
        mock_load_from_dict.return_value = loaded_metadata

        # Run
        BaseSynthesizer._check_input_metadata_updated(instance)

        # Assert
        assert instance._input_metadata == loaded_metadata

    @patch('sdv.single_table.base._check_regex_format')
    def test__validate_regex_format(self, mock_check_regex_format):
        """Test the ``_validate_regex_format`` method."""
        # Setup
        instance = Mock()
        metadata = Mock()
        metadata.get_column_names.return_value = ['id_1', 'id_2']

        columns = {
            'id_1': {'sdtype': 'id', 'regex_format': '[0-9]+'},
            'id_2': {'sdtype': 'id', 'regex_format': '[a-z]{3}'},
        }
        metadata.tables = {}
        metadata.tables['table_1'] = Mock()
        metadata.tables['table_1'].columns = columns

        instance.metadata = metadata
        instance._table_name = 'table_1'

        # Run
        BaseSynthesizer._validate_regex_format(instance)

        # Assert
        mock_check_regex_format.assert_has_calls([
            call('table_1', 'id_1', '[0-9]+'),
            call('table_1', 'id_2', '[a-z]{3}'),
        ])
        assert mock_check_regex_format.call_count == 2

    @patch('sdv.single_table.base.datetime')
    @patch('sdv.single_table.base.generate_synthesizer_id')
    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.BaseSynthesizer._check_metadata_updated')
    @patch('sdv.single_table.base.BaseSynthesizer._validate_regex_format')
    def test___init___l(
        self,
        mock_validate_regex_format,
        mock_check_metadata_updated,
        mock_data_processor,
        mock_generate_synthesizer_id,
        mock_datetime,
        caplog,
    ):
        """Test instantiating with default values."""
        # Setup
        metadata = Metadata()
        synthesizer_id = 'SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        mock_generate_synthesizer_id.return_value = synthesizer_id
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'

        # Run
        with catch_sdv_logs(caplog, logging.INFO, logger='SingleTableSynthesizer'):
            instance = BaseSynthesizer(metadata)

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance._data_processor == mock_data_processor.return_value
        assert instance._random_state_set is False
        assert instance._fitted is False
        assert instance._synthesizer_id == synthesizer_id
        args, kwargs = mock_data_processor.call_args
        assert isinstance(kwargs['metadata'], SingleTableMetadata)
        assert kwargs['enforce_rounding'] == instance.enforce_rounding
        assert kwargs['enforce_min_max_values'] == instance.enforce_min_max_values
        assert kwargs['locales'] == instance.locales
        assert len(kwargs) == 4
        assert not args

        assert instance._input_metadata == metadata
        assert instance._original_metadata != metadata
        assert instance._original_metadata.to_dict() == metadata.to_dict()

        mock_check_metadata_updated.assert_called_once()
        mock_generate_synthesizer_id.assert_called_once_with(instance)
        assert caplog.messages[0] == str({
            'EVENT': 'Instance',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'BaseSynthesizer',
            'SYNTHESIZER ID': 'SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        })

    def test__init__with_old_metadata_future_warning(self):
        """Test that future warning is thrown when using `SingleTableMetadata`.

        This test also ensures that the multiple metadata objects stored in the instance are
        as expected. Where:
            - `_input_metadata` points to the original input one (id matches).
            - `metadata` is a new instance of `Metadata`.
            - `_original_metadata` is a new instance of `Metadata` but the id does not match
              the `metadata` one.
        """
        # Setup
        metadata = SingleTableMetadata.load_from_dict({
            'columns': {
                'a': {'sdtype': 'categorical'},
            }
        })
        warn_msg = re.escape(
            "The 'SingleTableMetadata' is deprecated. Please use the new "
            "'Metadata' class for synthesizers."
        )
        # Run
        with pytest.warns(FutureWarning, match=warn_msg):
            instance = BaseSynthesizer(metadata)

        # Assert
        assert isinstance(instance._input_metadata, SingleTableMetadata)
        assert isinstance(instance._original_metadata, Metadata)
        assert isinstance(instance.metadata, Metadata)
        assert id(instance._input_metadata) == id(metadata)
        assert id(instance._original_metadata) != id(instance.metadata)
        assert instance._original_metadata.to_dict() == instance.metadata.to_dict()
        assert instance._input_metadata.to_dict() != instance.metadata.to_dict()

    def test___init__with_unified_metadata(self):
        """Test initialization with unified metadata."""
        # Setup
        metadata = Metadata.load_from_dict({
            'tables': {
                'table_1': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                    },
                }
            }
        })

        multi_metadata = Metadata.load_from_dict({
            'tables': {
                'table_1': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                    },
                },
                'table_2': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                    },
                },
            }
        })

        # Run and Assert
        BaseSynthesizer(metadata)
        error_msg = re.escape(
            'Metadata contains more than one table, use a MultiTableSynthesizer instead.'
        )

        with pytest.raises(InvalidMetadataError, match=error_msg):
            BaseSynthesizer(multi_metadata)

    @patch('sdv.single_table.base.DataProcessor')
    def test___init__custom(self, mock_data_processor):
        """Test that instantiating with custom parameters are properly stored in the instance."""
        # Setup
        metadata = Metadata()

        # Run
        instance = BaseSynthesizer(
            metadata, enforce_min_max_values=False, enforce_rounding=False, locales='en_CA'
        )

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance.locales == 'en_CA'
        assert instance._data_processor == mock_data_processor.return_value
        args = mock_data_processor.call_args[1]
        assert isinstance(args['metadata'], SingleTableMetadata)
        assert args['enforce_rounding'] == instance.enforce_rounding
        assert args['enforce_min_max_values'] == instance.enforce_min_max_values
        assert args['locales'] == instance.locales
        assert len(args) == 4

    def test___init__invalid_enforce_min_max_values(self):
        """Test it crashes when ``enforce_min_max_values`` is not a boolean."""
        # Run and Assert
        err_msg = re.escape(
            "Invalid value 'invalid' for parameter 'enforce_min_max_values'."
            ' Please provide True or False.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            BaseSynthesizer(Metadata(), enforce_min_max_values='invalid')

    def test___init__invalid_enforce_rounding(self):
        """Test it crashes when ``enforce_rounding`` is not a boolean."""
        # Run and Assert
        err_msg = re.escape(
            "Invalid value 'invalid' for parameter 'enforce_rounding'."
            ' Please provide True or False.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            BaseSynthesizer(Metadata(), enforce_rounding='invalid')

    def test_set_address_columns_warning(self):
        """Test ``set_address_columns`` method when the synthesizer has been fitted."""
        # Setup
        synthesizer = BaseSynthesizer(Metadata())

        # Run and Assert
        expected_message = re.escape(
            '`set_address_columns` is deprecated. Please add these columns directly to your'
            ' metadata using `add_column_relationship`.'
        )
        with pytest.warns(FutureWarning, match=expected_message):
            synthesizer.set_address_columns(
                ['country_column', 'city_column'], anonymization_level='full'
            )

    def test_get_parameters(self):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = Metadata()
        instance = BaseSynthesizer(
            metadata, enforce_min_max_values=False, enforce_rounding=False, locales='en_CA'
        )

        # Run
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {
            'enforce_min_max_values': False,
            'enforce_rounding': False,
            'locales': 'en_CA',
        }

    def test_get_metadata(self):
        """Test that it returns the ``metadata`` object."""
        # Setup
        metadata = Metadata()
        instance = BaseSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # Run
        result = instance.get_metadata()

        # Assert
        assert result.to_dict() == metadata.to_dict()

    def test__validate_transform_constraints(self):
        """Test the ``_validate_transform_constraints`` method."""
        # Setup
        data = pd.DataFrame()
        original_metadata = Metadata()
        metadata_1 = Metadata()
        metadata_2 = Metadata()
        instance = BaseSynthesizer(original_metadata)
        constraint_mock_1 = Mock()
        constraint_mock_1.get_updated_metadata = Mock(return_value=metadata_1)
        constraint_mock_1.transform = Mock(return_value=data)
        constraint_mock_2 = Mock()
        constraint_mock_2.get_updated_metadata = Mock(return_value=metadata_2)
        constraint_mock_2.transform = Mock(return_value=data)
        constraint_mock_3 = Mock()
        instance._chained_constraints = [constraint_mock_1, constraint_mock_2]
        instance._reject_sampling_constraints = [constraint_mock_3]

        # Run and Assert
        instance._validate_transform_constraints(data)

        constraint_mock_1.get_updated_metadata.assert_called_once_with(instance._original_metadata)
        constraint_mock_1.fit.assert_called_once_with(
            data=data, metadata=instance._original_metadata
        )
        constraint_mock_2.fit.assert_called_once_with(data=data, metadata=metadata_1)
        constraint_mock_3.fit.assert_called_once_with(
            data=data, metadata=instance._original_metadata
        )
        assert instance._constraints_fitted is True

        # Reset mock call history
        constraint_mock_1.fit.reset_mock()
        constraint_mock_1.transform.reset_mock()
        constraint_mock_2.fit.reset_mock()
        constraint_mock_2.transform.reset_mock()
        constraint_mock_3.fit.reset_mock()

        # Re-run to check it only transforms when constraints are already fitted
        instance._validate_transform_constraints(data)

        constraint_mock_1.transform.assert_called_once_with(data)
        constraint_mock_2.transform.assert_called_once_with(data)
        constraint_mock_1.fit.assert_not_called()
        constraint_mock_2.fit.assert_not_called()

        # Check the constraints are fitted again with enforce_constraint_fitting=True
        instance._validate_transform_constraints(data, enforce_constraint_fitting=True)

        constraint_mock_1.fit.assert_called_once_with(
            data=data, metadata=instance._original_metadata
        )
        constraint_mock_2.fit.assert_called_once_with(data=data, metadata=metadata_1)
        constraint_mock_3.fit.assert_called_once_with(
            data=data, metadata=instance._original_metadata
        )

    def test_validate(self):
        """Test the appropriate methods are called.

        Mock _validate_metadata, _validate_constraints and _validate, with no errors being raised.
        """
        # Setup
        data = pd.DataFrame()
        metadata = Metadata()
        instance = BaseSynthesizer(metadata)
        instance._validate = Mock(return_value=[])
        instance._validate_transform_constraints = Mock()
        instance._original_metadata.validate_data = Mock()

        # Run
        instance.validate(data)

        # Assert
        instance._original_metadata.validate_data.assert_called_once_with({
            instance._table_name: data
        })
        instance._validate.assert_called_once_with(data)
        instance._validate_transform_constraints.assert_called_once_with(
            data, enforce_constraint_fitting=True
        )

    def test_validate_raises_invalid_data_for_metadata(self):
        """Test that if ``metadata`` validation fails we raise an error for it.

        Mock _validate_metadata, _validate_constraints and _validate, with at least one of them
        returning an error, and ensure that they are all called and the error is surfaced.
        """
        # Setup
        data = pd.DataFrame()
        metadata = Metadata()
        instance = BaseSynthesizer(metadata)
        instance._original_metadata.validate_data = Mock(
            side_effect=InvalidMetadataError('\nThe provided data does not match the metadata.')
        )
        instance._validate = Mock(return_value=[])

        # Run and Assert
        err_msg = 'The provided data does not match the metadata.'
        with pytest.raises(InvalidMetadataError, match=err_msg):
            instance.validate(data)

        # Assert auxiliary methods are called
        instance._original_metadata.validate_data.assert_called_once_with({
            instance._table_name: data
        })
        instance._validate.assert_not_called()

    def test_validate_int_primary_key_regex_starts_with_zero(self):
        """Test that an error is raised if the primary key is an int that can start with 0.

        If the the primary key is stored as an int, but a regex is used with it, it is possible
        that the first character can be a 0. If this happens, then we can get duplicate primary
        key values since two different strings can be the same when converted ints
        (eg. '00123' and '0123').
        """
        # Setup
        data = pd.DataFrame({'key': [1, 2, 3], 'info': ['a', 'b', 'c']})
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('key', 'table', sdtype='id', regex_format='[0-9]{3,4}')
        metadata.add_column('info', 'table', sdtype='categorical')
        metadata.set_primary_key('key', 'table')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        message = (
            'Primary key "key" is stored as an int but the Regex allows it to start with '
            '"0". Please remove the Regex or update it to correspond to valid ints.'
        )
        with pytest.raises(InvalidDataError, match=message):
            instance.validate(data)

    def test_validate_int_primary_key_regex_does_not_start_with_zero(self):
        """Test that no error is raised if the primary key is an int that can't start with 0.

        If the the primary key is stored as an int, but a regex is used with it, it is possible
        that the first character can be a 0. If it isn't possible, then no error should be raised.
        """
        # Setup
        data = pd.DataFrame({'key': [1, 2, 3], 'info': ['a', 'b', 'c']})
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('key', 'table', sdtype='id', regex_format='[1-9]{3,4}')
        metadata.add_column('info', 'table', sdtype='categorical')
        metadata.set_primary_key('key', 'table')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        instance.validate(data)

    def test_auto_assign_transformers(self):
        """Test that the ``DataProcessor.prepare_for_fitting`` is being called."""
        # Setup
        instance = Mock()
        data = pd.DataFrame({'name': ['John', 'Doe', 'Johanna'], 'salary': [80.0, 90.0, 120.0]})
        instance._validate_transform_constraints = Mock(return_value=data)

        # Run
        BaseSynthesizer.auto_assign_transformers(instance, data)

        # Assert
        instance.validate.assert_called_once_with(data)
        instance._validate_transform_constraints.assert_called_once_with(data)
        instance._data_processor.prepare_for_fitting.assert_called_once_with(data)

    def test_auto_assign_transformers_with_invalid_data(self):
        """Test that auto_assign_transformer throws useful error about invalid data"""
        # Setup
        metadata = Metadata.load_from_dict({
            'columns': {
                'a': {'sdtype': 'categorical'},
            }
        })
        synthesizer = GaussianCopulaSynthesizer(metadata)
        # input data that does not match the metadata
        data = pd.DataFrame({'b': list(np.random.choice(['M', 'F'], size=10))})
        error_msg = re.escape(
            'The provided data does not match the metadata:\n'
            'Errors in table:\n'
            "Error: The columns ['b'] are not present in the metadata.\n"
            "Error: The metadata columns ['a'] are not present in the data."
        )

        # Run and Assert
        with pytest.raises(InvalidDataError, match=error_msg):
            synthesizer.auto_assign_transformers(data)

    def test_get_transformers(self):
        """Test that this returns the field transformers from the ``HyperTransformer``."""
        # Setup
        instance = Mock()
        instance._data_processor._hyper_transformer.field_transformers = {
            'name': 'FrequencyEncoder',
            'salary': 'FloatFormatter',
            'salary#name': 'LabelEncoder',
            'address': None,
        }
        instance._get_table_metadata.return_value.columns = {
            'salary': {'sdtype': 'numerical'},
            'name': {'sdtype': 'categorical'},
            'address': {'sdtype': 'address'},
        }

        # Run
        result = BaseSynthesizer.get_transformers(instance)

        # Assert
        assert result == {
            'salary': 'FloatFormatter',
            'name': 'FrequencyEncoder',
            'address': None,
            'salary#name': 'LabelEncoder',
        }

    def test_get_transformers_with_columns_dropped_by_constraint(self):
        """Test that this returns the transformers that are within the ``field_transformers``."""
        # Setup
        instance = Mock()
        instance._data_processor._hyper_transformer.field_transformers = {
            'salary#name': 'LabelEncoder',
            'address': None,
        }
        instance._get_table_metadata.return_value.columns = {
            'salary': {'sdtype': 'numerical'},
            'name': {'sdtype': 'categorical'},
            'address': {'sdtype': 'address'},
        }

        # Run
        result = BaseSynthesizer.get_transformers(instance)

        # Assert
        assert result == {'address': None, 'salary#name': 'LabelEncoder'}

    def test_get_transformers_raises_an_error(self):
        """Test that this raises an error when there are no field transformers."""
        # Setup
        instance = Mock()
        instance._data_processor._hyper_transformer.field_transformers = {}

        # Run and Assert
        error_msg = re.escape(
            "No transformers were returned in 'get_transformers'. Use "
            "'auto_assign_transformers' or 'fit' to create them."
        )
        with pytest.raises(ValueError, match=error_msg):
            BaseSynthesizer.get_transformers(instance)

    @patch('sdv.single_table.base.warnings')
    def test__preprocess_helper(self, mock_warnings):
        """Test the ``_preprocess_helper`` method."""
        # Setup
        instance = Mock()
        instance._fitted = True
        data = pd.DataFrame({'name': ['John', 'Doe', 'John Doe']})
        instance._validate_transform_constraints.side_effect = lambda x: x
        expected_warning = (
            'This model has already been fitted. To use the new preprocessed data, please '
            "refit the model using 'fit' or 'fit_processed_data'."
        )

        # Run
        result = BaseSynthesizer._preprocess_helper(instance, data)

        # Assert
        mock_warnings.warn.assert_called_once_with(expected_warning)
        instance.validate.assert_called_once_with(data)
        instance._validate_transform_constraints.assert_called_once_with(data)
        pd.testing.assert_frame_equal(result, data)

    def test__preprocess(self):
        """Test the method preprocesses the data.

        The method calls the ``validate`` function with the data, then fits the
        ``instance._data_processor`` and returns the output of the transformation.
        """
        # Setup
        instance = Mock()
        instance._fitted = True
        data = pd.DataFrame({'name': ['John', 'Doe', 'John Doe']})

        # Run
        result = BaseSynthesizer._preprocess(instance, data)

        # Assert
        assert result == instance._data_processor.transform.return_value
        instance._data_processor.fit.assert_called_once()
        pd.testing.assert_frame_equal(data, instance._data_processor.fit.call_args_list[0][0][0])
        pd.testing.assert_frame_equal(
            data, instance._data_processor.transform.call_args_list[0][0][0]
        )

    def test_preprocess(self):
        """Test the preprocess method."""
        # Setup
        instance = Mock()
        instance._fitted = True
        data = pd.DataFrame({'name': ['John', 'Doe', 'John Doe']})
        instance._preprocess_helper.return_value = data
        instance._store_and_convert_original_cols = Mock(return_value=False)
        instance._preprocess.return_value = data

        # Run
        result = BaseSynthesizer.preprocess(instance, data)

        # Assert
        instance._store_and_convert_original_cols.assert_called_once_with(data)
        instance._preprocess_helper.assert_called_once_with(data)
        instance._preprocess.assert_called_once_with(data)
        pd.testing.assert_frame_equal(result, data)

    def test_preprocess_int_columns(self):
        """Test the preprocess method.

        Ensure that data with column names as integers are not changed by
        preprocess.
        """
        # Setup
        metadata = Metadata().load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        1: {'sdtype': 'id'},
                        2: {'sdtype': 'id'},
                        'str': {'sdtype': 'id'},
                    },
                },
            },
        })
        instance = BaseSynthesizer(metadata)
        instance._fitted = False
        instance._original_columns = pd.Index([1, 2, 'str'])
        data = pd.DataFrame({
            1: ['John', 'Doe', 'John Doe'],
            2: ['John', 'Doe', 'John Doe'],
            'str': ['John', 'Doe', 'John Doe'],
        })

        # Run
        instance.preprocess(data)

        # Assert
        corrected_frame = pd.DataFrame({
            1: ['John', 'Doe', 'John Doe'],
            2: ['John', 'Doe', 'John Doe'],
            'str': ['John', 'Doe', 'John Doe'],
        })

        pd.testing.assert_frame_equal(data, corrected_frame)

    @patch('sdv.single_table.base.DataProcessor')
    def test__fit(self, mock_data_processor):
        """Test that ``NotImplementedError`` is being raised."""
        # Setup
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('key', 'table', sdtype='id')
        data = Mock()
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._fit(data)

    @patch('sdv.single_table.base.datetime')
    def test_fit_processed_data(self, mock_datetime, caplog):
        """Test that ``fit_processed_data`` calls the ``_fit``."""
        # Setup
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        instance = Mock(
            _fitted_sdv_version=None,
            _fitted_sdv_enterprise_version=None,
            _synthesizer_id='SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        )
        processed_data = pd.DataFrame({'column_a': [1, 2, 3]})

        # Run
        with catch_sdv_logs(caplog, logging.INFO, 'SingleTableSynthesizer'):
            BaseSynthesizer.fit_processed_data(instance, processed_data)

        # Assert
        instance._fit.assert_called_once_with(processed_data)
        assert caplog.messages[0] == str({
            'EVENT': 'Fit processed data',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
            'TOTAL NUMBER OF TABLES': 1,
            'TOTAL NUMBER OF ROWS': 3,
            'TOTAL NUMBER OF COLUMNS': 1,
        })

    def test_fit_processed_data_raises_version_error(self):
        """Test that ``fit`` raises ``VersionError``

        When attempting to refit a model that was created on a previous version of the software
        this will raise an error.
        """
        # Setup
        instance = Mock(
            _fitted_sdv_version='1.0.0',
            _fitted_sdv_enterprise_version=None,
        )
        processed_data = pd.DataFrame({'column_a': [1, 2, 3]})
        instance._random_state_set = True
        instance._fitted = True

        # Run and Assert
        error_msg = (
            f'You are currently on SDV version {version.community} but this synthesizer was '
            'created on version 1.0.0. Fitting this synthesizer again is not supported. Please '
            'create a new synthesizer.'
        )
        with pytest.raises(VersionError, match=error_msg):
            BaseSynthesizer.fit_processed_data(instance, processed_data)

    @patch('sdv.single_table.base.datetime')
    def test_fit(self, mock_datetime, caplog):
        """Test that ``fit`` calls ``preprocess`` and the ``fit_processed_data``.

        When fitting, the synthsizer has to ``preprocess`` the data and with the output
        of this method, call the ``fit_processed_data``
        """
        # Setup
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        instance = Mock(
            _fitted_sdv_version=None,
            _fitted_sdv_enterprise_version=None,
            _synthesizer_id='SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        )
        instance._store_and_convert_original_cols.return_value = False
        data = pd.DataFrame({'column_a': [1, 2, 3], 'name': ['John', 'Doe', 'Johanna']})
        instance._random_state_set = True
        instance._fitted = True

        # Run
        with catch_sdv_logs(caplog, logging.INFO, 'SingleTableSynthesizer'):
            BaseSynthesizer.fit(instance, data)

        # Assert
        assert instance._random_state_set is False
        instance._data_processor.reset_sampling.assert_called_once_with()
        instance.preprocess.assert_called_once_with(data)
        instance.fit_processed_data.assert_called_once_with(instance.preprocess.return_value)
        instance._check_input_metadata_updated.assert_called_once()
        assert caplog.messages[0] == str({
            'EVENT': 'Fit',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
            'TOTAL NUMBER OF TABLES': 1,
            'TOTAL NUMBER OF ROWS': 3,
            'TOTAL NUMBER OF COLUMNS': 2,
        })

    def test_fit_raises_version_error(self):
        """Test that ``fit`` raises ``VersionError``

        When attempting to refit a model that was created on a previous version of the software
        this will raise an error.
        """
        # Setup
        instance = Mock(
            _fitted_sdv_version='1.0.0',
            _fitted_sdv_enterprise_version=None,
        )
        data = pd.DataFrame({'column_a': [1, 2, 3]})
        instance._random_state_set = True
        instance._fitted = True

        # Run and Assert
        error_msg = (
            f'You are currently on SDV version {version.community} but this synthesizer was '
            'created on version 1.0.0. Fitting this synthesizer again is not supported. Please '
            'create a new synthesizer.'
        )
        with pytest.raises(VersionError, match=error_msg):
            BaseSynthesizer.fit(instance, data)

    def test_fit_raises_warning_if_metadata_updated(self):
        """Test that ``fit`` raises a warning if the original metadata was updated."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('column_a', sdtype='numerical')
        instance = BaseSynthesizer(metadata)
        instance._fit = Mock()

        data = pd.DataFrame({'column_a': [1, 2, 3]})
        instance._random_state_set = True
        instance._fitted = True
        metadata.update_column('column_a', sdtype='categorical')

        # Run and Assert
        warn_msg = (
            'Your metadata has been modified. Metadata modifications cannot be applied to an '
            'existing synthesizer. Please create a new synthesizer with the modified metadata.'
        )
        with pytest.warns(UserWarning, match=warn_msg):
            instance.fit(data)

    def test_update_transformers_invalid_keys(self):
        """Test error is raised if passed transformer doesn't match key column.

        The transformers of a key column must be either AnonymizedFaker or RegexGenerator.
        Raise an error if any other transformer is passed.
        """
        # Setup
        column_name_to_transformer = {'col2': RegexGenerator(), 'col3': FloatFormatter()}
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col2', 'table', sdtype='id')
        metadata.add_column('col3', 'table', sdtype='id')
        metadata.set_sequence_key('col2', 'table')

        metadata.add_alternate_keys(['col3'], 'table')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "Column 'col3' is a key. It cannot be preprocessed using "
            "the 'FloatFormatter' transformer."
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            instance.update_transformers(column_name_to_transformer)

    def test_update_transformers_already_fitted(self):
        """Test error is raised if passed transformer was already fitted."""
        # Setup
        fitted_transformer = FloatFormatter()
        fitted_transformer.fit(pd.DataFrame({'col': [1]}), 'col')
        column_name_to_transformer = {'col1': BinaryEncoder(), 'col2': fitted_transformer}
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col1', 'table', sdtype='boolean')
        metadata.add_column('col2', 'table', sdtype='numerical')
        instance = BaseSynthesizer(metadata)

        # Run and Assert
        err_msg = "Transformer for column 'col2' has already been fit on data."
        with pytest.raises(SynthesizerInputError, match=err_msg):
            instance.update_transformers(column_name_to_transformer)

    def test_update_transformers_warns_gaussian_copula(self):
        """Test warning is raised when ohe is used for categorical column in the GaussianCopula."""
        # Setup
        column_name_to_transformer = {'col1': OneHotEncoder(), 'col2': FloatFormatter()}
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col1', 'table', sdtype='categorical')
        metadata.add_column('col2', 'table', sdtype='numerical')
        instance = GaussianCopulaSynthesizer(metadata)
        instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))

        # Run and Assert
        warning = re.escape(
            "Using a OneHotEncoder transformer for column 'col1' "
            'may slow down the preprocessing and modeling times.'
        )
        with pytest.warns(UserWarning, match=warning):
            instance.update_transformers(column_name_to_transformer)

        field_transformers = instance._data_processor._hyper_transformer.field_transformers
        assert len(field_transformers) == 2
        assert isinstance(field_transformers['col1'], OneHotEncoder)
        assert isinstance(field_transformers['col2'], FloatFormatter)

    def test_update_transformers_warns_models(self):
        """Test warning is raised for some models.

        A warning should be raised if a transformer is assigned to boolean/categorical columns for
        the CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer and PARSynthesizer models.
        """
        # Setup
        column_name_to_transformer = {'col1': OneHotEncoder(), 'col2': FloatFormatter()}
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col1', 'table', sdtype='categorical')
        metadata.add_column('col2', 'table', sdtype='numerical')

        # NOTE: when PARSynthesizer is implemented, add it here as well
        for model in [CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer]:
            instance = model(metadata)
            instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))

            # Run and Assert
            warning = re.escape(
                "Replacing the default transformer for column 'col1' might "
                'impact the quality of your synthetic data.'
            )
            with pytest.warns(UserWarning, match=warning):
                instance.update_transformers(column_name_to_transformer)

    def test_update_transformers_warns_fitted(self):
        """Test warning is raised if model is fitted.

        A warning telling the user they need to refit the model should be raised if a transformer
        is updated after the model has been fitted.
        """
        # Setup
        column_name_to_transformer = {'col1': GaussianNormalizer(), 'col2': GaussianNormalizer()}
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col1', 'table', sdtype='numerical')
        metadata.add_column('col2', 'table', sdtype='numerical')
        instance = BaseSynthesizer(metadata)
        instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))
        instance._fitted = True

        # Run and Assert
        warning_msg = re.escape(
            'For this change to take effect, please refit the synthesizer using `fit`.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            instance.update_transformers(column_name_to_transformer)

    def test_update_transformers(self):
        """Test method correctly updates the transformers in the HyperTransformer."""
        # Setup
        column_name_to_transformer = {'col1': GaussianNormalizer(), 'col2': GaussianNormalizer()}
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col1', 'table', sdtype='numerical')
        metadata.add_column('col2', 'table', sdtype='numerical')
        instance = BaseSynthesizer(metadata)
        instance._data_processor.fit(pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]}))

        # Run
        instance.update_transformers(column_name_to_transformer)

        # Assert
        field_transformers = instance._data_processor._hyper_transformer.field_transformers
        assert len(field_transformers) == 2
        assert isinstance(field_transformers['col1'], GaussianNormalizer)
        assert isinstance(field_transformers['col2'], GaussianNormalizer)

    def test_update_transformers_warns_rounding(self):
        """Test warning is raised if model cannot round."""
        # Setup
        column_name_to_transformer = {
            'col1': GaussianNormalizer(learn_rounding_scheme=False),
            'col2': GaussianNormalizer(learn_rounding_scheme=True),
            'col3': GaussianNormalizer(learn_rounding_scheme=False),
        }
        metadata = Metadata()
        instance = BaseSynthesizer(metadata)
        instance._validate_transformers = MagicMock()
        instance._warn_quality_and_performance = MagicMock()
        instance._data_processor = MagicMock()
        instance.enforce_rounding = True
        instance._fitted = False

        # Run and Assert
        warn_msg = re.escape(
            "Unable to turn off rounding scheme for column(s) ['col1', 'col3'], "
            'because the overall synthesizer is enforcing rounding. We recommend '
            "setting the synthesizer's 'enforce_rounding' parameter to False."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            instance.update_transformers(column_name_to_transformer)

    def test_validate_constraints(self):
        """Test the ``validate_constraints`` method with multiple constraints."""
        # Setup
        synthetic_data = pd.DataFrame()
        transformed_data = pd.DataFrame()
        original_metadata = Metadata()
        instance = BaseSynthesizer(original_metadata)
        constraint_mock_1 = Mock()
        constraint_mock_1.transform.return_value = transformed_data
        constraint_mock_2 = Mock()
        instance._chained_constraints = [constraint_mock_1, constraint_mock_2]

        # Run
        instance.validate_constraints(synthetic_data)

        # Assert
        constraint_mock_1.is_valid.assert_called_once_with(data=synthetic_data)
        constraint_mock_1.transform.assert_called_once_with(data=synthetic_data)
        constraint_mock_2.is_valid.assert_called_once_with(data=transformed_data)
        constraint_mock_2.transform.assert_called_once_with(data=transformed_data)

    def test_validate_constraints_raises(self):
        """Test the ``validate_constraints`` method raises an error."""
        # Setup
        synthetic_data = pd.DataFrame()
        original_metadata = Metadata()
        instance = BaseSynthesizer(original_metadata)
        constraint_mock_1 = Mock()
        constraint_mock_1.is_valid.return_value = pd.Series([False, False])
        instance._chained_constraints = [constraint_mock_1]
        msg = 'The mock requirement is not met for row indices: 0, 1.'

        # Run and Assert
        with pytest.raises(ConstraintNotMetError, match=msg):
            instance.validate_constraints(synthetic_data)

    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.ProgrammableConstraintHarness')
    def test_add_constraints(self, mock_programmable_constraint_harness, mock_data_processor):
        """Test adding constraints to the synthesizer."""
        # Setup
        instance = Mock()
        instance._chained_constraints = []
        instance._reject_sampling_constraints = []

        constraint1 = Mock()
        constraint2 = Mock()
        constraint3 = Mock()
        instance._validate_constraints_single_table.side_effect = lambda constraint: constraint
        constraint3.get_updated_metadata.side_effect = [ConstraintNotMetError, None]
        constraint4 = SingleTableProgrammableConstraint()
        mock_harness = Mock()
        mock_programmable_constraint_harness.return_value = mock_harness

        # Run
        BaseSynthesizer.add_constraints(instance, [constraint1, constraint2])
        instance._fitted = True
        BaseSynthesizer.add_constraints(instance, [constraint3, constraint4])

        # Assert
        assert instance._chained_constraints == [constraint1, constraint2, mock_harness]
        assert instance._reject_sampling_constraints == [constraint3]
        mock_programmable_constraint_harness.assert_called_once_with(constraint4)
        mock_data_processor.assert_has_calls([
            call(
                metadata=constraint2.get_updated_metadata()._convert_to_single_table.return_value,
                enforce_rounding=instance.enforce_rounding,
                enforce_min_max_values=instance.enforce_min_max_values,
                locales=instance.locales,
            ),
            call(
                metadata=mock_harness.get_updated_metadata()._convert_to_single_table.return_value,
                enforce_rounding=instance.enforce_rounding,
                enforce_min_max_values=instance.enforce_min_max_values,
                locales=instance.locales,
            ),
        ])

    @patch('sdv.single_table.base.datetime')
    @patch('sdv.single_table.base.cloudpickle')
    def test_save(self, cloudpickle_mock, mock_datetime, tmp_path, caplog):
        """Test that the synthesizer is saved correctly."""
        # Setup
        synthesizer = Mock(
            _synthesizer_id='SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        )
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'

        # Run
        filepath = tmp_path / 'output.pkl'
        with catch_sdv_logs(caplog, logging.INFO, 'SingleTableSynthesizer'):
            BaseSynthesizer.save(synthesizer, filepath)

        # Assert
        cloudpickle_mock.dump.assert_called_once_with(synthesizer, ANY)
        assert caplog.messages[0] == str({
            'EVENT': 'Save',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        })

    def test_save_warning(self, tmp_path):
        """Test that the synthesizer produces a warning if saved without fitting."""
        # Setup
        synthesizer = BaseSynthesizer(Metadata())

        # Run and Assert
        warn_msg = re.escape(
            'You are saving a synthesizer that has not yet been fitted. You will not be able '
            'to sample synthetic data without fitting. We recommend fitting the synthesizer '
            'first and then saving.'
        )
        with pytest.warns(Warning, match=warn_msg):
            filepath = os.path.join(tmp_path, 'output.pkl')
            synthesizer.save(filepath)

    @patch('sdv.single_table.base.datetime')
    @patch('sdv.single_table.base.generate_synthesizer_id')
    @patch('sdv.single_table.base.check_synthesizer_version')
    @patch('sdv.single_table.base.check_sdv_versions_and_warn')
    @patch('sdv.single_table.base.cloudpickle')
    @patch('builtins.open', new_callable=mock_open)
    @patch('sdv.single_table.base.warn_load_deprecated')
    @patch('sdv.single_table.base._validate_correct_synthesizer_loading')
    def test_load(
        self,
        mock_validate_correct_synthesizer_loading,
        mock_warn_load_deprecated,
        mock_file,
        cloudpickle_mock,
        mock_check_sdv_versions_and_warn,
        mock_check_synthesizer_version,
        mock_generate_synthesizer_id,
        mock_datetime,
        caplog,
    ):
        """Test that the ``load`` method loads a stored synthesizer."""
        # Setup
        synthesizer_mock = Mock(_fitted=False, _synthesizer_id=None)
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        synthesizer_id = 'SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        mock_generate_synthesizer_id.return_value = synthesizer_id
        cloudpickle_mock.load.return_value = synthesizer_mock

        # Run
        with catch_sdv_logs(caplog, logging.INFO, 'SingleTableSynthesizer'):
            loaded_instance = BaseSynthesizer.load('synth.pkl')

        # Assert
        mock_validate_correct_synthesizer_loading.assert_called_once_with(
            synthesizer_mock, BaseSynthesizer
        )
        mock_warn_load_deprecated.assert_called_once()
        mock_file.assert_called_once_with('synth.pkl', 'rb')
        cloudpickle_mock.load.assert_called_once_with(mock_file.return_value)
        mock_check_sdv_versions_and_warn.assert_called_once_with(loaded_instance)
        assert loaded_instance == synthesizer_mock
        assert loaded_instance._synthesizer_id == synthesizer_id
        mock_check_synthesizer_version.assert_called_once_with(synthesizer_mock)
        mock_generate_synthesizer_id.assert_called_once_with(synthesizer_mock)
        assert caplog.messages[0] == str({
            'EVENT': 'Load',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'SingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        })

    def test_load_custom_constraint_classes(self):
        """Test that ``load_custom_constraint_classes`` calls the ``DataProcessor``'s method."""
        # Setup
        instance = Mock()

        # Run
        BaseSynthesizer.load_custom_constraint_classes(
            instance, 'path/to/file.py', ['Custom', 'Constr', 'UpperPlus']
        )

        # Assert
        instance._data_processor.load_custom_constraint_classes.assert_called_once_with(
            'path/to/file.py', ['Custom', 'Constr', 'UpperPlus']
        )

    @patch('builtins.open')
    @patch('sdv.single_table.base.cloudpickle')
    def test_load_runtime_error(self, cloudpickle_mock, mock_open):
        """Test that the synthesizer's load method errors with the correct message."""
        # Setup
        cloudpickle_mock.load.side_effect = RuntimeError(
            (
                'Attempting to deserialize object on a CUDA device but '
                'torch.cuda.is_available() is False. If you are running on a CPU-only machine,'
                " please use torch.load with map_location=torch.device('cpu') "
                'to map your storages to the CPU.'
            )
        )

        # Run and Assert
        err_msg = re.escape(
            'This synthesizer was created on a machine with GPU but the current machine is'
            ' CPU-only. This feature is currently unsupported. We recommend sampling on '
            'the same GPU-enabled machine.'
        )
        with pytest.raises(SamplingError, match=err_msg):
            BaseSynthesizer.load('synth.pkl')

    @patch('builtins.open')
    @patch('sdv.single_table.base.cloudpickle')
    def test_load_runtime_error_no_change(self, cloudpickle_mock, mock_open):
        """Test that the synthesizer's load method errors with the correct message."""
        # Setup
        cloudpickle_mock.load.side_effect = RuntimeError('Error')

        # Run and Assert
        with pytest.raises(RuntimeError, match='Error'):
            BaseSynthesizer.load('synth.pkl')

    @patch('sdv.single_table.base.version')
    def test_get_info_no_enterprise(self, mock_sdv_version):
        """Test the correct dictionary is returned.

        Check the return dictionary is valid both before and after fitting the synthesizer.

        Mocks:
            * Unfortunately, ``datetime`` can't be mocked directly. This link explains how to
            do it: https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
        """
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3]})
        mock_sdv_version.community = '1.0.0'
        mock_sdv_version.enterprise = None
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col', 'table', sdtype='numerical')

        with patch('sdv.single_table.base.datetime.datetime') as mock_date:
            mock_date.today.return_value = datetime(2023, 1, 23)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            synthesizer = GaussianCopulaSynthesizer(metadata)

            # Run
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'GaussianCopulaSynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': False,
                'last_fit_date': None,
                'fitted_sdv_version': None,
            }

            # Run
            synthesizer.fit(data)
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'GaussianCopulaSynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': True,
                'last_fit_date': '2023-01-23',
                'fitted_sdv_version': '1.0.0',
            }

    @patch('sdv.single_table.base.version')
    def test_get_info_with_enterprise(self, mock_sdv_version):
        """Test the correct dictionary is returned with the enterprise version.

        Check the return dictionary is valid both before and after fitting the synthesizer.

        Mocks:
            * Unfortunately, ``datetime`` can't be mocked directly. This link explains how to
            do it: https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
        """
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3]})
        mock_sdv_version.community = '1.0.0'
        mock_sdv_version.enterprise = '1.2.0'
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col', 'table', sdtype='numerical')

        with patch('sdv.single_table.base.datetime.datetime') as mock_date:
            mock_date.today.return_value = datetime(2023, 1, 23)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            synthesizer = GaussianCopulaSynthesizer(metadata)

            # Run
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'GaussianCopulaSynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': False,
                'last_fit_date': None,
                'fitted_sdv_version': None,
            }

            # Run
            synthesizer.fit(data)
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'GaussianCopulaSynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': True,
                'last_fit_date': '2023-01-23',
                'fitted_sdv_version': '1.0.0',
                'fitted_sdv_enterprise_version': '1.2.0',
            }


class TestBaseSingleTableSynthesizer:
    @patch('sdv.single_table.base.DataProcessor')
    def test__set_random_state(self, mock_data_processor):
        """Test that ``_model.set_random_state`` is being called with the input value.

        When passing a random state to the method ``_set_random_state``, this calls the
        instance's model ``set_random_state`` method in order to assign it to it.
        """
        # Setup
        rng_seed = Mock()
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('names', 'table', sdtype='categorical')
        instance = BaseSingleTableSynthesizer(metadata)
        instance._model = Mock()

        # Run
        instance._set_random_state(rng_seed)

        # Assert
        instance._model.set_random_state.assert_called_once_with(rng_seed)
        assert instance._random_state_set is True

    def test_reset_sampling(self):
        """Test the ``reset_sampling`` method.

        Ensure that the ``reset_sampling`` sets the ``instance._random_state_set`` is set to
        ``False`` and the ``instance._data_processor.reset_sampling`` is being called.
        """
        # Setup
        instance = Mock()
        instance._random_state_set = True

        # Run
        BaseSingleTableSynthesizer.reset_sampling(instance)

        # Assert
        assert instance._random_state_set is False
        instance._data_processor.reset_sampling.assert_called_once_with()

    def test__filter_conditions(self):
        """Test that the method filters out data that doesn't meet the conditions."""
        # Setup
        sampled = pd.DataFrame({
            'position': [
                'Software Engineer',
                'Data Scientist',
                'Statistician',
                'Computer Systems Analyst',
                'Software Engineer',
                'Software Engineer',
                'Computer Systems Analyst',
                'Software Engineer',
            ],
            'salary': [80.0, 90.0, 50.0, 60.0, 82.0, 80.0, 80.0, 81.0],
        })
        conditions = {'position': 'Software Engineer', 'salary': 80.0}
        float_rtol = 0.01

        # Run
        filtered_data = BaseSingleTableSynthesizer._filter_conditions(
            sampled, conditions, float_rtol
        )

        # Assert
        expected_data = pd.DataFrame(
            {'position': ['Software Engineer', 'Software Engineer'], 'salary': [80.0, 80.0]},
            index=[0, 5],
        )
        pd.testing.assert_frame_equal(filtered_data, expected_data)

    def test__sample_rows_without_conditions(self):
        """Test that sample rows calls ``_sample`` when conditions is ``None``.

        Also ensure that when ``_random_state_set`` is ``False`` this calls
        ``_set_random_state`` with the ``FIXED_RNG_SEED`` which is ``73251``.
        """
        # Setup
        data = pd.DataFrame({'name': ['John', 'Doe', 'John Doe']})
        instance = Mock()
        instance._random_state_set = False
        instance._sample.return_value = pd.DataFrame()
        instance._data_processor.reverse_transform.return_value = data
        instance._data_processor.filter_valid.return_value = data
        instance._data_processor._hyper_transformer._input_columns = []
        instance._reject_sampling_constraints = []
        instance._chained_constraints = []
        instance.reverse_transform_constraints = Mock(side_effect=lambda x: x)

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(instance, 3)

        # Assert
        assert num_valid == 3
        pd.testing.assert_frame_equal(sampled, data)
        instance._sample.assert_called_once_with(3)
        instance._data_processor.reverse_transform.assert_called_once_with(
            instance._sample.return_value, conditions=None
        )
        instance.reverse_transform_constraints.assert_called_once_with(
            instance._data_processor.reverse_transform.return_value
        )
        instance._data_processor.filter_valid.assert_called_once_with(
            instance._data_processor.reverse_transform.return_value
        )
        instance._set_random_state.assert_called_once_with(73251)

    def test__sample_rows_with_conditions(self):
        """Test that sample rows calls with the transformed conditions the ``_sample``."""
        # Setup
        data = pd.DataFrame({'name': ['John', 'Doe', 'John Doe'], 'salary': [90.0, 100.0, 80.0]})
        instance = Mock()
        instance._sample.return_value = pd.DataFrame()
        instance._data_processor.reverse_transform.return_value = data
        instance._data_processor._hyper_transformer._input_columns = []
        instance._filter_conditions.return_value = data[data.name == 'John Doe']
        conditions = {'salary': 80.0}
        transformed_conditions = {'salary': 80.0}
        instance._reject_sampling_constraints = []
        instance._chained_constraints = []
        instance._formatters = {}
        instance.reverse_transform_constraints = Mock(side_effect=lambda x: x)

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(
            instance, 3, conditions=conditions, transformed_conditions=transformed_conditions
        )

        # Assert
        assert num_valid == 1
        pd.testing.assert_frame_equal(sampled, data[data.name == 'John Doe'])
        instance._sample.assert_called_once_with(3, {'salary': 80.0})
        instance._data_processor.reverse_transform.assert_called_once_with(
            instance._sample.return_value, conditions=conditions
        )
        instance._data_processor.filter_valid.assert_called_once_with(
            instance._data_processor.reverse_transform.return_value
        )

    def test__sample_rows_with_previous_rows(self):
        """Test that previous rows are being concatenated when provided to ``_sample``."""
        # Setup
        previous_rows = pd.DataFrame({
            'name': ['John Doe', 'John Doe', 'John Doe'],
            'salary': [80.0, 80.0, 80.0],
        })
        data = pd.DataFrame({'name': ['John', 'Doe', 'John Doe'], 'salary': [90.0, 100.0, 80.0]})

        instance = Mock()
        instance._sample.return_value = pd.DataFrame()
        instance._data_processor._hyper_transformer._input_columns = []
        instance._data_processor.filter_valid = lambda x: x
        instance._data_processor.reverse_transform.return_value = data
        instance._reject_sampling_constraints = []
        instance._chained_constraints = []
        instance._formatters = {}
        instance.reverse_transform_constraints = Mock(side_effect=lambda x: x)

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(
            instance, 3, previous_rows=previous_rows
        )

        # Assert
        expected_data = pd.DataFrame({
            'name': ['John Doe', 'John Doe', 'John Doe', 'John', 'Doe', 'John Doe'],
            'salary': [80.0, 80.0, 80.0, 90.0, 100.0, 80.0],
        })
        assert num_valid == 6
        pd.testing.assert_frame_equal(sampled, expected_data)
        instance._sample.assert_called_once_with(3)
        instance._data_processor.reverse_transform.assert_called_once_with(
            instance._sample.return_value, conditions=None
        )

    def test__sample_rows_notimplementederror(self):
        """Test when the model does not support conditional sampling and raises an error."""
        # Setup
        data = pd.DataFrame({'name': ['John', 'Doe', 'John Doe'], 'salary': [90.0, 100.0, 80.0]})
        instance = Mock()
        instance._data_processor.reverse_transform.return_value = data
        instance._data_processor._hyper_transformer._input_columns = []
        instance._filter_conditions.return_value = data[data.name == 'John Doe']
        instance._reject_sampling_constraints = []
        instance._chained_constraints = []
        conditions = {'salary': 80.0}
        transformed_conditions = {'salary': 80.0}
        instance._sample.side_effect = [NotImplementedError, pd.DataFrame()]

        # Run
        sampled, num_valid = BaseSingleTableSynthesizer._sample_rows(
            instance, 3, conditions=conditions, transformed_conditions=transformed_conditions
        )

        # Assert
        assert num_valid == 1
        pd.testing.assert_frame_equal(sampled, data[data.name == 'John Doe'])
        assert instance._sample.call_args_list == [call(3, {'salary': 80.0}), call(3)]

    def test__sample_rows_sdtypes_is_empty(self):
        """Test when ``_data_processor.get_sdtypes`` with ``primary_keys=False`` is empty.

        This test is when ``_data_processor.get_sdtypes`` returns an empty dictionary, which leads
        to only sampling using ``_data_processor.reverse_transform``.
        """
        # Setup
        instance = Mock()
        instance._data_processor.get_sdtypes.return_value = {}
        instance._data_processor.reverse_transform.side_effect = lambda x: x

        # Run
        sampled, num_rows = BaseSingleTableSynthesizer._sample_rows(instance, 10)

        # Assert
        assert num_rows == 10
        pd.testing.assert_frame_equal(sampled, pd.DataFrame(index=range(10)))
        instance._data_processor.reverse_transform.assert_called_once()

    def test__sample_batch_without_saving_to_file(self):
        """Test the ``_sample_batch`` without storing the samples in a file."""
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe'],
            'salary': [80.0, 60.0, 100.0],
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.return_value = (sampled_data, 3)

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=3,
            max_tries=100,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data)
        _sample_rows_args = instance._sample_rows.call_args[0]
        rows, conditions, trans_cond, float_rtol, sampled, keep_extra_columns = _sample_rows_args
        assert rows == 3
        assert conditions is None
        assert trans_cond is None
        assert float_rtol == 0.01
        pd.testing.assert_frame_equal(sampled, pd.DataFrame())
        assert keep_extra_columns is False

    def test__sample_batch_with_sampled_data_bigger_than_batch_size(self):
        """Test ``sampled_data`` is bigger than the batch size.

        If the sampled data is bigger than the batch size, the returned data frame must be the
        ``batch_size`` size.
        """
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'John Doe John'],
            'salary': [80.0, 60.0, 100.0, 300.0],
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.return_value = (sampled_data, 3)

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=3,
            max_tries=100,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data.head(3))
        _sample_rows_args = instance._sample_rows.call_args[0]
        rows, conditions, trans_cond, float_rtol, sampled, keep_extra_columns = _sample_rows_args
        assert rows == 3
        assert conditions is None
        assert trans_cond is None
        assert float_rtol == 0.01
        pd.testing.assert_frame_equal(sampled, pd.DataFrame())
        assert keep_extra_columns is False

    def test__sample_batch_max_tries_reached(self):
        """Test that when ``max_tries`` is reached, a break occurs."""
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'John Doe John'],
            'salary': [80.0, 60.0, 100.0, 300.0],
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.return_value = (sampled_data, 3)

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=10,
            max_tries=2,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data)
        _sample_rows_args = instance._sample_rows.call_args[0]
        _rows, _conditions, _trans_cond, _float_rtol, _sampled, _keep_extra_columns = (
            _sample_rows_args
        )
        assert instance._sample_rows.call_count == 2

    def test__sample_batch_storing_output_file(self, tmpdir):
        """Test that an output file is properly stored while sampling.

        Test that an output file is properly stored as well as that the progress bar is updated.
        """
        # Setup
        sampled_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'John Doe John'],
            'salary': [80.0, 60.0, 100.0, 300.0],
        })
        instance = Mock()
        instance.metadata.columns.keys.return_value = ['name', 'salary']
        instance._sample_rows.side_effect = [
            (sampled_data, 4),
            (sampled_data, 5),
            (sampled_data, 10),
        ]
        mock_progress_bar = Mock()

        path = tmpdir / 'file.csv'
        # Create an empty csv file
        open(path, mode='a').close()

        # Run
        result = BaseSingleTableSynthesizer._sample_batch(
            instance,
            batch_size=10,
            max_tries=100,
            conditions=None,
            transformed_conditions=None,
            float_rtol=0.01,
            progress_bar=mock_progress_bar,
            output_file_path=path,
        )

        # Assert
        pd.testing.assert_frame_equal(result, sampled_data)
        assert instance._sample_rows.call_count == 3
        expected_stored_data = pd.DataFrame({
            'name': [
                'John',
                'Doe',
                'John Doe',
                'John Doe John',
                'John Doe John',
                'John',
                'Doe',
                'John Doe',
                'John Doe John',
            ],
            'salary': [
                80.0,
                60.0,
                100.0,
                300.0,
                300.0,
                80.0,
                60.0,
                100.0,
                300.0,
            ],
        })
        data = pd.read_csv(path)
        pd.testing.assert_frame_equal(expected_stored_data, data)
        mock_progress_bar.update.call_count == 3

    def test__make_condition_dfs(self):
        """Test _make_condition_dfs works with Condition conditions."""
        # Setup
        condition_a = Condition({'name': 'John Doe'})
        condition_b = Condition({'salary': 80.0})

        # Run
        result = BaseSingleTableSynthesizer._make_condition_dfs([condition_a, condition_b])

        # Assert
        expected_result = [
            pd.DataFrame({'name': ['John Doe']}),
            pd.DataFrame({'salary': [80.0]}),
        ]
        for res, exp in zip(result, expected_result):
            pd.testing.assert_frame_equal(res, exp)

    def test__make_condition_dfs_dataframe_condition(self):
        """Test _make_condition_dfs works with DataFrameCondition conditions."""
        # Setup
        dataframe = pd.DataFrame({
            'name': ['John Doe'],
            'salary': [80.0],
        })
        users_condition = DataFrameCondition(table_name='table', dataframe=dataframe)

        # Run
        result = BaseSingleTableSynthesizer._make_condition_dfs([users_condition])

        # Assert
        expected_result = [dataframe]
        for res, exp in zip(result, expected_result):
            pd.testing.assert_frame_equal(res, exp)

    def test__make_condition_dfs_condition_and_dataframe_conditions(self):
        """Test _make_condition_dfs works with Condition and DataFrameCondition conditions."""
        # Setup
        condition_ = Condition({'name': 'John Doe'})
        dataframe = pd.DataFrame({'salary': [80.0]})
        dataframe_condition = DataFrameCondition(table_name=None, dataframe=dataframe)

        # Run
        result = BaseSingleTableSynthesizer._make_condition_dfs([condition_, dataframe_condition])

        # Assert
        expected_result = [
            pd.DataFrame({'name': ['John Doe']}),
            pd.DataFrame({'salary': [80.0]}),
        ]
        for res, exp in zip(result, expected_result):
            pd.testing.assert_frame_equal(res, exp)

    def test__make_condition_dfs_raises(self):
        """Test _make_condition_dfs raises an error with invalid condition"""

        # Setup
        msg = '`conditions` must be list of Condition or DataFrameCondition'

        class CustomCondition:
            pass

        condition = CustomCondition()

        # Run
        with pytest.raises(ValueError, match=msg):
            BaseSingleTableSynthesizer._make_condition_dfs([condition])

    def test__sample_in_batches(self):
        """Test that this method calls and concatenates the output of ``_sample_batch``."""
        # Setup
        first_data = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe'],
            'salary': [60.0, 70.0, 80.0],
        })
        second_data = pd.DataFrame({
            'name': ['Johana', 'Doe', 'Johana Doe'],
            'salary': [65.0, 75.0, 85.0],
        })
        instance = Mock()
        instance._sample_batch.side_effect = [first_data, second_data]

        # Run
        result = BaseSingleTableSynthesizer._sample_in_batches(
            instance,
            num_rows=6,
            batch_size=3,
            max_tries_per_batch=100,
            conditions='conditions',
            transformed_conditions='transformed_conditions',
            float_rtol=0.02,
            progress_bar='progress_bar',
            output_file_path='output_file_path',
        )

        # Assert
        expected_result = pd.DataFrame({
            'name': ['John', 'Doe', 'John Doe', 'Johana', 'Doe', 'Johana Doe'],
            'salary': [60.0, 70.0, 80.0, 65.0, 75.0, 85.0],
        })
        pd.testing.assert_frame_equal(result, expected_result)
        assert instance._sample_batch.call_count == 2
        expected_call = call(
            batch_size=3,
            max_tries=100,
            conditions='conditions',
            transformed_conditions='transformed_conditions',
            float_rtol=0.02,
            progress_bar='progress_bar',
            output_file_path='output_file_path',
            keep_extra_columns=False,
        )
        assert expected_call == instance._sample_batch.call_args_list[0]
        assert expected_call == instance._sample_batch.call_args_list[1]

    def test__conditionally_sample_rows(self):
        """Test when sampled rows is bigger than 0."""
        # Setup
        transformed_data = pd.DataFrame({COND_IDX: [0, 1, 2], 'salary': [65.0, 75.0, 85.0]})
        data = pd.DataFrame({
            'name': ['Johana', 'Doe', 'Johana Doe'],
            'salary': [65.0, 75.0, 85.0],
        })
        instance = Mock()
        instance._sample_in_batches.return_value = data
        condition = Mock()
        transformed_condition = Mock()

        # Run
        result = BaseSingleTableSynthesizer._conditionally_sample_rows(
            instance,
            transformed_data,
            condition,
            transformed_condition,
        )

        # Assert
        expected_data = pd.DataFrame({
            'name': ['Johana', 'Doe', 'Johana Doe'],
            'salary': [65.0, 75.0, 85.0],
            COND_IDX: [0, 1, 2],
        })
        pd.testing.assert_frame_equal(expected_data, result)
        instance._sample_in_batches.assert_called_once_with(
            num_rows=3,
            batch_size=3,
            max_tries_per_batch=None,
            conditions=condition,
            transformed_conditions=transformed_condition,
            float_rtol=0.01,
            progress_bar=None,
            output_file_path=None,
            keep_extra_columns=False,
        )

    def test__conditionally_sample_rows_no_rows_sampled_error(self):
        """Test that an error is raised when no rows are sampled."""
        # Setup
        transformed_data = pd.DataFrame({COND_IDX: [0, 1, 2], 'salary': [65.0, 75.0, 85.0]})
        instance = Mock()
        instance._sample_in_batches.return_value = []
        condition = Mock()
        transformed_condition = 'fancy_condition'

        # Run and Assert
        expected_message = re.escape(
            'Unable to sample any rows for the given conditions '
            "'fancy_condition'. Try increasing 'max_tries_per_batch' (currently: None) "
            "or increasing 'batch_size' (currently: 3). Note that "
            'increasing these values will also increase the sampling time.'
        )
        with pytest.raises(ValueError, match=expected_message):
            BaseSingleTableSynthesizer._conditionally_sample_rows(
                instance,
                transformed_data,
                condition,
                transformed_condition,
                graceful_reject_sampling=False,
            )

    def test__conditionally_sample_rows_raises_value_error_model_is_gm(self):
        """Test when there are no sampled rows and the model is ``GaussianMultivariate``.

        When no ``graceful_reject_sampling`` is used and the model is ``GaussianMultivariate``
        the sampling raises a ``ValueError`` with a specific message for this model.
        """
        # Setup
        transformed_data = pd.DataFrame({COND_IDX: [0, 1, 2], 'salary': [65.0, 75.0, 85.0]})
        instance = Mock()
        instance._sample_in_batches.return_value = []
        condition = Mock()
        transformed_condition = 'fancy_condition'
        instance._model = GaussianMultivariate()

        # Run and Assert
        expected_message = re.escape(
            'Unable to sample any rows for the given conditions '
            "'fancy_condition'. This may be because the provided values are out-of-bounds in the "
            'current model. \nPlease try again with a different set of values.'
        )
        with pytest.raises(ValueError, match=expected_message):
            BaseSingleTableSynthesizer._conditionally_sample_rows(
                instance,
                transformed_data,
                condition,
                transformed_condition,
                graceful_reject_sampling=False,
            )

    def test__conditionally_sample_rows_with_graceful_reject_sampling(self):
        """Test when no rows are sampled but ``graceful_reject_sampling`` is ``True``."""
        # Setup
        transformed_data = pd.DataFrame({COND_IDX: [0, 1, 2], 'salary': [65.0, 75.0, 85.0]})
        instance = Mock()
        instance._sample_in_batches.return_value = []
        condition = Mock()
        transformed_condition = 'fancy_condition'
        instance._model = GaussianMultivariate()

        # Run
        result = BaseSingleTableSynthesizer._conditionally_sample_rows(
            instance,
            transformed_data,
            condition,
            transformed_condition,
            graceful_reject_sampling=True,
        )

        # Assert
        assert result == []

    def test__sample_with_progress_bar_num_rows_is_none(self):
        """Test that a ``ValueError`` is raised when ``num_rows`` is ``None``."""
        # Setup
        instance = Mock()
        num_rows = None

        # Run and Assert
        expected_message = re.escape(
            'You must specify the number of rows to sample (e.g. num_rows=100).'
        )
        with pytest.raises(ValueError, match=expected_message):
            BaseSingleTableSynthesizer._sample_with_progress_bar(instance, num_rows)

    def test__sample_with_progress_bar_num_rows_is_zero(self):
        """Test that an empty dataframe is returned when ``num_rows`` is 0."""
        # Setup
        instance = Mock()
        num_rows = 0

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, num_rows)

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame())

    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test__sample_with_progress_bar_returns_sampled_data(
        self, mock_validate_file_path, mock_tqdm
    ):
        """Test that ``_sample_in_batches`` is being called and it's output is returned."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        instance._sample_in_batches.return_value = pd.DataFrame({
            'name': ['John', 'Johanna', 'Doe']
        })

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, 10)

        # Assert
        expected_result = pd.DataFrame({'name': ['John', 'Johanna', 'Doe']})
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=10, disable=False)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with('Sampling rows')
        instance._sample_in_batches.assert_called_once_with(
            num_rows=10,
            batch_size=10,
            max_tries_per_batch=100,
            progress_bar=progress_bar.__enter__.return_value,
            output_file_path=mock_validate_file_path.return_value,
        )

    @patch('sdv.single_table.base.handle_sampling_error')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test__sample_with_progress_bar_handle_sampling_error(
        self, mock_validate_file_path, mock_tqdm, mock_handle_sampling_error
    ):
        """Test the error handling when we are using ``_sample_in_batches``."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        keyboard_error = KeyboardInterrupt()
        instance._sample_in_batches.side_effect = [keyboard_error]
        mock_validate_file_path.return_value = 'temp_file'

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, 10)

        # Assert
        expected_result = pd.DataFrame()
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=10, disable=False)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with('Sampling rows')
        instance._sample_in_batches.assert_called_once_with(
            num_rows=10,
            batch_size=10,
            max_tries_per_batch=100,
            progress_bar=progress_bar.__enter__.return_value,
            output_file_path=mock_validate_file_path.return_value,
        )
        mock_handle_sampling_error.assert_called_once_with('temp_file', keyboard_error)

    @patch('sdv.single_table.base.os')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test__sample_with_progress_bar_removing_temp_file(
        self, mock_validate_file_path, mock_tqdm, mock_os
    ):
        """Test that the temporary file is removed after sampling."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        instance._sample_in_batches.return_value = pd.DataFrame()
        mock_validate_file_path.return_value = '.sample.csv.temp'

        # Run
        result = BaseSingleTableSynthesizer._sample_with_progress_bar(instance, 10)

        # Assert
        expected_result = pd.DataFrame()
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=10, disable=False)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with('Sampling rows')
        instance._sample_in_batches.assert_called_once_with(
            num_rows=10,
            batch_size=10,
            max_tries_per_batch=100,
            progress_bar=progress_bar.__enter__.return_value,
            output_file_path=mock_validate_file_path.return_value,
        )

    def test_sample_not_fitted(self):
        """Test that ``sample`` raises an error when the synthesizer is not fitted."""
        # Setup
        instance = Mock()
        instance._fitted = False

        def raise_sampling_error():
            raise SamplingError(
                'This synthesizer has not been fitted. Please fit your synthesizer first before'
                ' sampling synthetic data.'
            )

        instance._validate_fit_before_sample = Mock(side_effect=raise_sampling_error)
        expected_message = re.escape(
            'This synthesizer has not been fitted. Please fit your synthesizer first before'
            ' sampling synthetic data.'
        )

        # Run and Assert
        with pytest.raises(SamplingError, match=expected_message):
            BaseSingleTableSynthesizer.sample(instance, 10)

    def test__sample_with_progress_bar_without_output_filepath(self):
        """Test that ``_sample_with_progress_bar`` raises an error
        when the synthesizer is not fitted.
        """
        # Setup
        instance = Mock()
        instance._fitted = True
        expected_message = re.escape(
            'Error: Sampling terminated. No results were saved due to unspecified '
            '"output_file_path".'
        )
        instance._sample_in_batches.side_effect = RuntimeError('Mocked Error')

        # Run and Assert
        with pytest.raises(RuntimeError, match=expected_message) as exception:
            BaseSingleTableSynthesizer._sample_with_progress_bar(
                instance, output_file_path=None, num_rows=10
            )
        assert isinstance(exception.value.__cause__, RuntimeError)
        assert 'Mocked Error' in str(exception.value.__cause__)

    @patch('sdv.single_table.base.datetime')
    def test_sample(self, mock_datetime, caplog):
        """Test that we use ``_sample_with_progress_bar`` in this method."""
        # Setup
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        num_rows = 10
        max_tries_per_batch = 50
        batch_size = 5
        output_file_path = 'temp.csv'
        instance = Mock(
            _synthesizer_id='BaseSingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        )
        instance.get_metadata.return_value._constraints = False
        instance._sample_with_progress_bar.return_value = pd.DataFrame({'col': [1, 2, 3]})
        instance._reverse_transform_constraints.return_value = pd.DataFrame({'col': [1, 2, 3]})

        # Run
        with catch_sdv_logs(caplog, logging.INFO, logger='SingleTableSynthesizer'):
            result = BaseSingleTableSynthesizer.sample(
                instance,
                num_rows,
                max_tries_per_batch,
                batch_size,
                output_file_path,
            )

        # Assert
        instance._sample_with_progress_bar.assert_called_once_with(
            10, 50, 5, 'temp.csv', show_progress_bar=True
        )
        instance._check_input_metadata_updated.assert_called_once_with()
        pd.testing.assert_frame_equal(result, pd.DataFrame({'col': [1, 2, 3]}))
        assert caplog.messages[0] == str({
            'EVENT': 'Sample',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'BaseSingleTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
            'TOTAL NUMBER OF TABLES': 1,
            'TOTAL NUMBER OF ROWS': 3,
            'TOTAL NUMBER OF COLUMNS': 1,
        })

    @patch('sdv.single_table.base.datetime')
    def test_sample_warns_if_metadata_updated(self, mock_datetime, caplog):
        """Test that if we call sample with updated metadata a warning will be shown."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('column_a', sdtype='numerical')
        instance = BaseSingleTableSynthesizer(metadata)
        instance._sample_with_progress_bar = Mock(return_value=pd.DataFrame())
        instance._fitted = True

        metadata.update_column('column_a', sdtype='categorical')

        # Run and Assert
        warn_msg = (
            'Your metadata has been modified. Metadata modifications cannot be applied to an '
            'existing synthesizer. Please create a new synthesizer with the modified metadata.'
        )
        with pytest.warns(UserWarning, match=warn_msg):
            instance.sample(5)

    def test__validate_conditions_unseen_columns(self):
        """Test that conditions are within the original metadata columns."""
        # Setup
        instance = Mock()
        instance._original_metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'name': {'sdtype': 'categorical'},
                        'surname': {'sdtype': 'categorical'},
                    }
                }
            }
        })
        instance._table_name = 'table'
        conditions = pd.DataFrame({'name': ['Johanna'], 'surname': ['Doe']})

        # Run and Assert
        BaseSingleTableSynthesizer._validate_conditions_unseen_columns(instance, conditions)

    def test__validate_conditions_unseen_columns_backwards_compatability(self):
        """Test that conditions are within the ``data_processor`` fields."""
        # Setup
        instance = Mock()
        delattr(instance, '_original_metadata')
        instance._data_processor.get_sdtypes.return_value = {
            'name': 'categorical',
            'surname': 'categorical',
        }
        conditions = pd.DataFrame({'name': ['Johanna'], 'surname': ['Doe']})

        # Run
        BaseSingleTableSynthesizer._validate_conditions_unseen_columns(instance, conditions)

        # Assert
        instance._data_processor.get_sdtypes.assert_called()

    def test__validate_conditions_unseen_columns_raises_error(self):
        """Test that conditions are not in the original metadata columns."""
        # Setup
        instance = Mock()
        instance._original_metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'name': {'sdtype': 'categorical'},
                        'surname': {'sdtype': 'categorical'},
                    }
                }
            }
        })
        instance._table_name = 'table'
        conditions = pd.DataFrame({'names': ['Johanna'], 'surname': ['Doe']})

        # Run and Assert
        error_msg = re.escape(
            "Unexpected column name 'names'. Use a column name that was present in the "
            'original data.'
        )
        with pytest.raises(ValueError, match=error_msg):
            BaseSingleTableSynthesizer._validate_conditions_unseen_columns(instance, conditions)

    def test__validate_conditions_unseen_columns_raises_error_backwards_compatability(self):
        """Test that conditions are not in the ``data_processor`` fields."""
        # Setup
        instance = Mock()
        delattr(instance, '_original_metadata')
        instance._data_processor.get_sdtypes.return_value = {
            'name': 'categorical',
            'surname': 'categorical',
        }
        conditions = pd.DataFrame({'names': ['Johanna'], 'surname': ['Doe']})

        # Run and Assert
        error_msg = re.escape(
            "Unexpected column name 'names'. Use a column name that was present in the "
            'original data.'
        )
        with pytest.raises(ValueError, match=error_msg):
            BaseSingleTableSynthesizer._validate_conditions_unseen_columns(instance, conditions)

    def test__validate_conditions_primary_key_raises_error(self):
        """Test that conditions containing the primary key error."""
        # Setup
        instance = Mock()
        instance._original_metadata = Metadata.load_from_dict({
            'tables': {
                'table': {
                    'columns': {
                        'name': {'sdtype': 'id'},
                        'surname': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'name',
                },
            }
        })
        instance._table_name = 'table'
        conditions = pd.DataFrame({'name': ['Johanna'], 'surname': ['Doe']})

        # Run and Assert
        error_msg = re.escape(
            "Cannot conditionally sample column name 'name' because it is the primary key."
        )
        with pytest.raises(ValueError, match=error_msg):
            BaseSingleTableSynthesizer._validate_conditions_unseen_columns(instance, conditions)

    def test__transform_conditions_chained_constraints_constraints_not_met(self):
        """Test when conditions are not met."""
        # Setup
        conditions = pd.DataFrame({'name': ['Johanna'], 'salary': [100.0]})
        instance = Mock()
        instance._validate_transform_constraints.side_effect = [ConstraintNotMetError]

        # Run and Assert
        error_msg = 'Provided conditions are not valid for the given constraints.'
        with pytest.raises(ConstraintNotMetError, match=error_msg):
            BaseSingleTableSynthesizer._transform_conditions_chained_constraints(
                instance,
                conditions,
            )

    def test__sample_with_conditions_constraints_not_met_backwards_compatability(self):
        """Test when conditions are not met."""
        # Setup
        conditions = pd.DataFrame({'name': ['Johanna', 'Doe'], 'salary': [100.0, 90.0]})
        instance = Mock()
        delattr(instance, '_chained_constraints')
        instance._data_processor.transform.side_effect = [ConstraintsNotMetError]

        # Run and Assert
        error_msg = 'Provided conditions are not valid for the given constraints.'
        with pytest.raises(ConstraintsNotMetError, match=error_msg):
            BaseSingleTableSynthesizer._sample_with_conditions(
                instance,
                conditions,
                10,
                10,
            )

    def test__sample_with_conditions_transformed_with_transformed_data(self):
        """Test when the condition is transformed, this is being used to conditionally sample."""
        # Setup
        conditions = pd.DataFrame({'name': ['Johanna', 'Doe']})
        instance = Mock()
        instance._transform_conditions_chained_constraints.side_effect = [
            pd.DataFrame({'name': [0.25]}),
            pd.DataFrame({'name': [0.90]}),
        ]
        instance._conditionally_sample_rows.side_effect = [
            pd.DataFrame({'name': ['Johanna'], COND_IDX: [1]}),
            pd.DataFrame({'name': ['Doe'], COND_IDX: [0]}),
        ]

        # Run
        result = BaseSingleTableSynthesizer._sample_with_conditions(
            instance,
            conditions,
            10,
            10,
        )

        # Assert
        sample_calls = instance._conditionally_sample_rows.call_args_list
        first_call_kwargs = sample_calls[0][1]
        second_call_kwargs = sample_calls[1][1]
        first_df = first_call_kwargs.pop('dataframe')
        second_df = second_call_kwargs.pop('dataframe')

        pd.testing.assert_frame_equal(result, pd.DataFrame({'name': ['Doe', 'Johanna']}))

        pd.testing.assert_frame_equal(
            first_df, pd.DataFrame({'name': [0.25], COND_IDX: [1]}, index=[1])
        )
        pd.testing.assert_frame_equal(second_df, pd.DataFrame({'name': [0.90], COND_IDX: [0]}))
        assert first_call_kwargs == {
            'condition': {'name': 'Doe'},
            'transformed_condition': {'name': 0.25},
            'max_tries_per_batch': 10,
            'batch_size': 10,
            'progress_bar': None,
            'output_file_path': None,
            'keep_extra_columns': False,
        }
        assert second_call_kwargs == {
            'condition': {'name': 'Johanna'},
            'transformed_condition': {'name': 0.90},
            'max_tries_per_batch': 10,
            'batch_size': 10,
            'progress_bar': None,
            'output_file_path': None,
            'keep_extra_columns': False,
        }

    def test__transform_conditions_chained_constraints_no_transformed_conditions(self):
        """Test when the conditions are not being transformed.

        Test that when the conditions are not transformable, this calls the conditional
        sampling with the expected values and makes the ``transformed_condition`` to be ``None``.
        """
        # Setup
        conditions = pd.DataFrame({'name': ['Johanna']})
        instance = Mock()
        instance._validate_transform_constraints.side_effect = [KeyError]
        instance._data_processor.transform.side_effect = [
            pd.DataFrame(),
            pd.DataFrame(),
        ]
        instance._conditionally_sample_rows.return_value = pd.DataFrame()

        # Run
        result = BaseSingleTableSynthesizer._transform_conditions_chained_constraints(
            instance,
            conditions,
        )

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame())

    @patch('sdv.single_table.base.os')
    @patch('sdv.single_table.base.check_num_rows')
    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_from_conditions(
        self, mock_validate_file_path, mock_tqdm, mock_data_processor, mock_check_num_rows, mock_os
    ):
        """Test sample conditions with sampled data and reject sampling.

        An instance of ``BaseSingleTableSynthesizer`` is created and it's utility functions are
        being mocked to reach the point of calling ```_sample_with_conditions``. After the
        sampling is done, when the ``temp_file`` is the ``.sample.csv.temp``, this is being
        removed.
        """
        # Setup
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('names', 'table', sdtype='categorical')
        instance = BaseSingleTableSynthesizer(metadata)
        conditions = [Condition({'name': 'John Doe'})]
        mock_validate_file_path.return_value = '.sample.csv.temp'

        instance._validate_conditions = Mock()
        instance._sample_with_conditions = Mock()
        instance._model = GaussianMultivariate()
        instance._sample_with_conditions.return_value = pd.DataFrame({'name': ['John Doe']})
        instance._validate_fit_before_sample = Mock()
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar

        # Run
        result = instance.sample_from_conditions(conditions, 10, 10, '.sample.csv.temp')

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame({'name': ['John Doe']}))

    @patch('sdv.single_table.base.handle_sampling_error')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_from_conditions_handle_sampling_error(
        self, mock_validate_file_path, mock_tqdm, mock_handle_sampling_error
    ):
        """Test the error handling when we are using ``sample_from_conditions``."""
        # Setup
        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar
        instance = Mock()
        instance._make_condition_dfs.side_effect = lambda x: x
        conditions = [Condition({'name': 'John Doe'})]
        keyboard_error = KeyboardInterrupt()
        instance._sample_with_conditions.side_effect = [keyboard_error]
        mock_validate_file_path.return_value = 'temp_file'

        # Run
        result = BaseSingleTableSynthesizer.sample_from_conditions(
            instance, conditions, 10, 10, '.sample.csv.temp'
        )

        # Assert
        expected_result = pd.DataFrame()
        pd.testing.assert_frame_equal(result, expected_result)
        mock_tqdm.tqdm.assert_called_once_with(total=1)
        progress_bar.__enter__.return_value.set_description.assert_called_once_with(
            'Sampling conditions'
        )
        mock_handle_sampling_error.assert_called_once_with('temp_file', keyboard_error)

    @patch('sdv.single_table.base.os')
    @patch('sdv.single_table.base.check_num_rows')
    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_remaining_columns(
        self, mock_validate_file_path, mock_tqdm, mock_data_processor, mock_check_num_rows, mock_os
    ):
        """Test the this method calls ``_sample_with_conditions`` with the ``known_column."""
        # Setup
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('names', 'table', sdtype='categorical')
        instance = BaseSingleTableSynthesizer(metadata)
        known_columns = pd.DataFrame({'name': ['Johanna Doe']})

        instance._validate_known_columns = Mock()
        instance._sample_with_conditions = Mock()
        instance._validate_fit_before_sample = Mock()
        instance._model = GaussianMultivariate()
        instance._sample_with_conditions.return_value = pd.DataFrame({'name': ['John Doe']})
        mock_validate_file_path.return_value = '.sample.csv.temp'

        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar

        # Run
        result = instance.sample_remaining_columns(known_columns, 10, 10, '.sample.csv.temp')

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame({'name': ['John Doe']}))

    @patch('sdv.single_table.base.handle_sampling_error')
    @patch('sdv.single_table.base.check_num_rows')
    @patch('sdv.single_table.base.DataProcessor')
    @patch('sdv.single_table.base.tqdm')
    @patch('sdv.single_table.base.validate_file_path')
    def test_sample_remaining_columns_handles_sampling_error(
        self,
        mock_validate_file_path,
        mock_tqdm,
        mock_data_processor,
        mock_check_num_rows,
        mock_handle_sampling_error,
    ):
        """Test when sample remaining is being interrupted.

        This should properly handle the errors with the ``handle_sampling_error`` function.
        """
        # Setup
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('names', 'table', sdtype='categorical')
        instance = BaseSingleTableSynthesizer(metadata)
        known_columns = pd.DataFrame({'name': ['Johanna Doe']})

        instance._validate_known_columns = Mock()
        instance._sample_with_conditions = Mock()
        instance._model = GaussianMultivariate()
        keyboard_error = KeyboardInterrupt()
        instance._sample_with_conditions.side_effect = [keyboard_error]
        mock_validate_file_path.return_value = 'temp_file'

        progress_bar = MagicMock()
        mock_tqdm.tqdm.return_value = progress_bar

        # Run
        result = instance.sample_remaining_columns(known_columns, 10, 10, 'temp_file')

        # Assert
        pd.testing.assert_frame_equal(result, pd.DataFrame())
        mock_handle_sampling_error.assert_called_once_with('temp_file', keyboard_error)

    def test__validate_fit_before_sample_fitted(self):
        """Test that ``_validate_fit_before_sample`` does nothing when synthesizer is fitted."""
        # Setup
        instance = Mock()
        instance._fitted = True

        # Run
        BaseSingleTableSynthesizer._validate_fit_before_sample(instance)

    def test__validate_fit_before_sample_not_fitted(self):
        """Test that ``_validate_fit_before_sample`` raises SamplingError when not fitted."""
        # Setup
        instance = Mock()
        instance._fitted = False
        expected_message = re.escape(
            'This synthesizer has not been fitted. Please fit your synthesizer first before'
            ' sampling synthetic data.'
        )

        # Run and Assert
        with pytest.raises(SamplingError, match=expected_message):
            BaseSingleTableSynthesizer._validate_fit_before_sample(instance)

    def test_sample_calls_validate_fit_before_sample(self):
        """Test that ``sample`` calls ``_validate_fit_before_sample``."""
        # Setup
        instance = Mock()
        instance._fitted = True
        instance._check_input_metadata_updated = Mock()
        instance._sample_with_progress_bar = Mock(return_value=pd.DataFrame())
        instance._original_columns = pd.Index([])
        instance._validate_fit_before_sample = Mock()

        # Run
        BaseSingleTableSynthesizer.sample(instance, 10)

        # Assert
        instance._validate_fit_before_sample.assert_called_once_with()
