import logging
import re
import warnings
from collections import defaultdict
from datetime import date, datetime
from unittest.mock import ANY, Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from sdv import version
from sdv.errors import (
    ConstraintsNotMetError, InvalidDataError, NotFittedError, SamplingError, SynthesizerInputError,
    VersionError)
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.ctgan import CTGANSynthesizer
from tests.utils import catch_sdv_logs, get_multi_table_data, get_multi_table_metadata


class TestBaseMultiTableSynthesizer:

    def test__initialize_models(self):
        """Test that this method initializes the ``self._synthezier`` for each table.

        This tests that we iterate over the tables within the metadata and if there are
        ``table_parameters`` we use those to initialize the model.
        """
        # Setup
        locales = ['en_CA', 'fr_CA']
        instance = Mock()
        instance._table_synthesizers = {}
        instance._table_parameters = {
            'nesreca': {
                'default_distribution': 'gamma'
            }
        }
        instance.locales = locales
        instance.metadata = get_multi_table_metadata()

        # Run
        BaseMultiTableSynthesizer._initialize_models(instance)

        # Assert
        assert instance._table_synthesizers == {
            'nesreca': instance._synthesizer.return_value,
            'oseba': instance._synthesizer.return_value,
            'upravna_enota': instance._synthesizer.return_value
        }
        instance._synthesizer.assert_has_calls([
            call(metadata=instance.metadata.tables['nesreca'], default_distribution='gamma',
                 locales=locales),
            call(metadata=instance.metadata.tables['oseba'], locales=locales),
            call(metadata=instance.metadata.tables['upravna_enota'], locales=locales)
        ])

    def test__get_pbar_args(self):
        """Test that ``_get_pbar_args`` returns a dictionary with disable opposite to verbose."""
        # Setup
        instance = Mock()
        instance.verbose = False

        # Run
        result = BaseMultiTableSynthesizer._get_pbar_args(instance)

        # Assert
        assert result == {'disable': True}

    def test__get_pbar_args_kwargs(self):
        """Test that ``_get_pbar_args`` returns a dictionary with the given kwargs."""
        # Setup
        instance = Mock()
        instance.verbose = True

        # Run
        result = BaseMultiTableSynthesizer._get_pbar_args(
            instance,
            desc='Process Table',
            position=0
        )

        # Assert
        assert result == {
            'disable': False,
            'desc': 'Process Table',
            'position': 0
        }

    @patch('sdv.multi_table.base.print')
    def test__print(self, mock_print):
        """Test that print info will print a message if verbose is True."""
        # Setup
        instance = Mock(verbose=True)

        # Run
        BaseMultiTableSynthesizer._print(instance, text='Fitting', end='')

        # Assert
        mock_print.assert_called_once_with('Fitting', end='')

    @patch('sdv.multi_table.base.datetime')
    @patch('sdv.multi_table.base.generate_synthesizer_id')
    @patch('sdv.multi_table.base.BaseMultiTableSynthesizer._check_metadata_updated')
    def test___init__(self, mock_check_metadata_updated, mock_generate_synthesizer_id,
                      mock_datetime, caplog):
        """Test that when creating a new instance this sets the defaults.

        Test that the metadata object is being stored and also being validated. Afterwards, this
        calls the ``self._initialize_models`` which creates the initial instances of those.
        """
        # Setup
        synthesizer_id = 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        mock_generate_synthesizer_id.return_value = synthesizer_id
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        metadata = get_multi_table_metadata()
        metadata.validate = Mock()

        # Run
        with catch_sdv_logs(caplog, logging.INFO, 'MultiTableSynthesizer'):
            instance = BaseMultiTableSynthesizer(metadata)

        # Assert
        assert instance.metadata == metadata
        assert isinstance(instance._table_synthesizers['nesreca'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['oseba'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['upravna_enota'], GaussianCopulaSynthesizer)
        assert instance._table_parameters == defaultdict(dict)
        instance.metadata.validate.assert_called_once_with()
        mock_check_metadata_updated.assert_called_once()
        mock_generate_synthesizer_id.assert_called_once_with(instance)
        assert instance._synthesizer_id == synthesizer_id
        assert caplog.messages[0] == str({
            'EVENT': 'Instance',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'BaseMultiTableSynthesizer',
            'SYNTHESIZER ID': 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        })

    def test__init__column_relationship_warning(self):
        """Test that a warning is raised only once when the metadata has column relationships."""
        # Setup
        metadata = get_multi_table_metadata()
        metadata.add_column('nesreca', 'lat', sdtype='latitude')
        metadata.add_column('nesreca', 'lon', sdtype='longitude')

        metadata.add_column_relationship('nesreca', 'gps', ['lat', 'lon'])

        expected_warning = (
            "The metadata contains a column relationship of type 'gps' "
            'which requires the gps add-on. This relationship will be ignored. For higher'
            ' quality data in this relationship, please inquire about the SDV Enterprise tier.'
        )

        # Run
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            BaseMultiTableSynthesizer(metadata)

        # Assert
        column_relationship_warnings = [
            warning for warning in caught_warnings if expected_warning in str(warning.message)
        ]
        assert len(column_relationship_warnings) == 1

    def test___init___synthesizer_kwargs_deprecated(self):
        """Test that the ``synthesizer_kwargs`` method is deprecated."""
        # Setup
        metadata = get_multi_table_metadata()
        metadata.validate = Mock()

        # Run and Assert
        warn_message = (
            'The `synthesizer_kwargs` parameter is deprecated as of SDV 1.2.0 and does not '
            'affect the synthesizer. Please use the `set_table_parameters` method instead.'
        )
        with pytest.warns(FutureWarning, match=warn_message):
            BaseMultiTableSynthesizer(metadata, synthesizer_kwargs={})

    def test__check_metadata_updated(self):
        """Test the ``_check_metadata_updated`` method."""
        # Setup
        instance = Mock()
        instance.metadata = Mock()
        instance.metadata._check_updated_flag = Mock()
        instance.metadata._reset_updated_flag = Mock()

        # Run
        expected_message = re.escape(
            "We strongly recommend saving the metadata using 'save_to_json' for replicability"
            ' in future SDV versions.'
        )
        with pytest.warns(UserWarning, match=expected_message):
            BaseMultiTableSynthesizer._check_metadata_updated(instance)

        # Assert
        instance.metadata._check_updated_flag.assert_called_once()
        instance.metadata._reset_updated_flag.assert_called_once()

    def test_set_address_columns(self):
        """Test the ``set_address_columns`` method."""
        # Setup
        metadata = MultiTableMetadata().load_from_dict({
            'tables': {
                'address_table': {
                    'columns': {
                        'country_column': {'sdtype': 'country_code'},
                        'city_column': {'sdtype': 'city'},
                        'parent_key': {'sdtype': 'id'},
                    },
                    'primary_key': 'parent_key'
                },
                'other_table': {
                    'columns': {
                        'numerical_column': {'sdtype': 'numerical'},
                        'child_foreign_key': {'sdtype': 'id'},
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'address_table',
                    'parent_primary_key': 'parent_key',
                    'child_table_name': 'other_table',
                    'child_foreign_key': 'child_foreign_key'
                }
            ]
        })
        columns = ('country_column', 'city_column')
        metadata.validate = Mock()
        SingleTableMetadata.validate = Mock()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_synthesizers['address_table'].set_address_columns = Mock()

        # Run
        instance.set_address_columns(
            'address_table', columns, anonymization_level='street_address'
        )

        # Assert
        instance._table_synthesizers['address_table'].set_address_columns.assert_called_once_with(
            columns, 'street_address'
        )

    def test_set_address_columns_error(self):
        """Test that ``set_address_columns`` raises an error for unknown table."""
        # Setup
        metadata = MultiTableMetadata()
        columns = ('country_column', 'city_column')
        metadata.validate = Mock()
        SingleTableMetadata.validate = Mock()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        expected_error = re.escape(
            'The provided data does not match the metadata:\n'
            "Table 'address_table' is not present in the metadata."
        )
        with pytest.raises(ValueError, match=expected_error):
            instance.set_address_columns(
                'address_table', columns, anonymization_level='street_address'
            )

    def test_get_table_parameters_empty(self):
        """Test that this method returns an empty dictionary when there are no parameters."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_table_parameters('oseba')

        # Assert
        assert result == {
            'synthesizer_name': 'GaussianCopulaSynthesizer',
            'synthesizer_parameters': {
                'default_distribution': 'beta',
                'enforce_min_max_values': True,
                'enforce_rounding': True,
                'locales': ['en_US'],
                'numerical_distributions': {}
            }
        }

    def test_get_table_parameters_has_parameters(self):
        """Test that this method returns a dictionary with the parameters."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance.set_table_parameters('oseba', {'default_distribution': 'gamma'})

        # Run
        result = instance.get_table_parameters('oseba')

        # Assert
        assert result['synthesizer_parameters'] == {
            'default_distribution': 'gamma',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': ['en_US'],
            'numerical_distributions': {}
        }

    def test_get_parameters(self):
        """Test that the synthesizer's parameters are being returned."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata, locales='en_CA')

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {'locales': 'en_CA', 'synthesizer_kwargs': None}

    def test_set_table_parameters(self):
        """Test that the table's parameters are being updated.

        This test should ensure that the ``self._table_parameters`` for the given table
        are being updated with the given parameters, and also that the model is re-created
        and updated it's parameters as well.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        instance.set_table_parameters('oseba', {'default_distribution': 'gamma'})

        # Assert
        table_parameters = instance.get_table_parameters('oseba')
        assert instance._table_parameters['oseba'] == {'default_distribution': 'gamma'}
        assert table_parameters['synthesizer_name'] == 'GaussianCopulaSynthesizer'
        assert table_parameters['synthesizer_parameters'] == {
            'default_distribution': 'gamma',
            'enforce_min_max_values': True,
            'locales': ['en_US'],
            'enforce_rounding': True,
            'numerical_distributions': {}
        }

    def test_set_table_parameters_invalid_enforce_min_max_values(self):
        """Test it crashes when ``enforce_min_max_values`` is not a boolean."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "Invalid value 'invalid' for parameter 'enforce_min_max_values'."
            ' Please provide True or False.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            instance.set_table_parameters('oseba', {'enforce_min_max_values': 'invalid'})

    def test_set_table_parameters_invalid_enforce_rounding(self):
        """Test it crashes when ``enforce_rounding`` is not a boolean."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "Invalid value 'invalid' for parameter 'enforce_rounding'."
            ' Please provide True or False.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            instance.set_table_parameters('oseba', {'enforce_rounding': 'invalid'})

    def test_get_metadata(self):
        """Test that the metadata object is returned."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_metadata()

        # Assert
        assert metadata == result

    def test_validate(self):
        """Test that no error is being raised when the data is valid."""
        # Setup
        metadata = get_multi_table_metadata()
        data = get_multi_table_data()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        instance.validate(data)

    def test_validate_missing_table(self):
        """Test that an error is being raised when there is a missing table in the dictionary."""
        # Setup
        metadata = get_multi_table_metadata()
        data = get_multi_table_data()
        data.pop('nesreca')

        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = "The provided data is missing the tables {'nesreca'}."
        with pytest.raises(InvalidDataError, match=error_msg):
            instance.validate(data)

    def test_validate_key_error(self):
        """Test that if a ``KeyError`` is raised the code will continue without erroring."""
        # Setup
        metadata = get_multi_table_metadata()
        data = get_multi_table_data()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_synthesizers.popitem()

        # Run and Assert
        instance.validate(data)

    def test_validate_data_is_not_dataframe(self):
        """Test that an error is being raised when the data is not a dataframe."""
        # Setup
        metadata = get_multi_table_metadata()
        data = get_multi_table_data()
        data['nesreca'] = pd.Series({
            'id_nesreca': np.arange(10),
            'upravna_enota': np.arange(10),
        })

        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = "Data must be a DataFrame, not a <class 'pandas.core.series.Series'>."
        with pytest.raises(InvalidDataError, match=error_msg):
            instance.validate(data)

    def test_validate_data_does_not_match(self):
        """Test that an error is being raised when the data does not match the metadata."""
        # Setup
        metadata = get_multi_table_metadata()
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(10),
                'upravna_enota': np.arange(10),
                'nesreca_val': np.arange(10).astype(str)
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10),
                'id_nesreca': np.arange(10),
                'oseba_val': np.arange(10).astype(str)
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
                'upravna_val': np.arange(10).astype(str)
            }),
        }

        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = re.escape(
            'The provided data does not match the metadata:\n'
            "Table: 'nesreca'\n"
            "Error: Invalid values found for numerical column 'nesreca_val': ['0', '1', '2', "
            "'+ 7 more']."
            "\n\nTable: 'oseba'\n"
            "Error: Invalid values found for numerical column 'oseba_val': ['0', '1', '2', "
            "'+ 7 more']."
            "\n\nTable: 'upravna_enota'\n"
            "Error: Invalid values found for numerical column 'upravna_val': ['0', '1', '2', "
            "'+ 7 more']."
        )
        with pytest.raises(InvalidDataError, match=error_msg):
            instance.validate(data)

    def test_validate_missing_foreign_keys(self):
        """Test that errors are being raised when there are missing foreign keys."""
        # Setup
        metadata = get_multi_table_metadata()
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(0, 20, 2),
                'upravna_enota': np.arange(10),
                'nesreca_val': np.arange(10)
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10),
                'id_nesreca': np.arange(10),
                'oseba_val': np.arange(10)
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
                'upravna_val': np.arange(10)
            }),
        }
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = re.escape(
            'The provided data does not match the metadata:\n'
            'Relationships:\n'
            "Error: foreign key column 'id_nesreca' contains unknown references: (1, 3, 5, 7, 9). "
            "Please use the method 'drop_unknown_references' from sdv.utils to clean the data."
        )
        with pytest.raises(InvalidDataError, match=error_msg):
            instance.validate(data)

    def test_validate_constraints_not_met(self):
        """Test that errors are being raised when there are constraints not met."""
        # Setup
        metadata = get_multi_table_metadata()
        data = get_multi_table_data()
        data['nesreca']['val'] = list(range(4))
        metadata.add_column('nesreca', 'val', sdtype='numerical')
        instance = BaseMultiTableSynthesizer(metadata)
        inequality_constraint = {
            'constraint_class': 'Inequality',
            'table_name': 'nesreca',
            'constraint_parameters': {
                'low_column_name': 'nesreca_val',
                'high_column_name': 'val',
                'strict_boundaries': True
            }
        }
        instance.add_constraints([inequality_constraint])

        # Run and Assert
        error_msg = (
            "\nData is not valid for the 'Inequality' constraint:\n"
            '   nesreca_val  val\n'
            '0            0    0\n'
            '1            1    1\n'
            '2            2    2\n'
            '3            3    3'
        )
        with pytest.raises(ConstraintsNotMetError, match=error_msg):
            instance.validate(data)

    def test_validate_table_synthesizers_errors(self):
        """Test that errors are being raised when the table synthesizer is erroring."""
        # Setup
        metadata = get_multi_table_metadata()
        data = get_multi_table_data()
        instance = BaseMultiTableSynthesizer(metadata)
        nesreca_synthesizer = Mock()
        nesreca_synthesizer._validate.return_value = ['Invalid data for PAR synthesizer.']
        instance._table_synthesizers['nesreca'] = nesreca_synthesizer

        # Run and Assert
        error_msg = (
            'The provided data does not match the metadata:\n'
            'Invalid data for PAR synthesizer.'
        )
        with pytest.raises(InvalidDataError, match=error_msg):
            instance.validate(data)

    def test_auto_assign_transformers(self):
        """Test that each table of the data calls its single table auto assign method."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        table1 = pd.DataFrame({'col1': [1, 2]})
        table2 = pd.DataFrame({'col2': [1, 2]})
        data = {
            'nesreca': table1,
            'oseba': table2
        }
        instance._table_synthesizers['nesreca'] = Mock()
        instance._table_synthesizers['oseba'] = Mock()

        # Run
        instance.auto_assign_transformers(data)

        # Assert
        instance._table_synthesizers['nesreca'].auto_assign_transformers.assert_called_once_with(
            table1)
        instance._table_synthesizers['oseba'].auto_assign_transformers.assert_called_once_with(
            table2)

    def test_auto_assign_transformers_foreign_key_none(self):
        """Test that each table's foreign key transformers are set to None."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        data = {
            'nesreca': Mock(),
            'oseba': Mock()
        }
        instance.validate = Mock()
        instance.metadata._get_all_foreign_keys = Mock(return_value=['a', 'b'])
        nesreca_synthesizer = Mock()
        oseba_synthesizer = Mock()
        instance._table_synthesizers['nesreca'] = nesreca_synthesizer
        instance._table_synthesizers['oseba'] = oseba_synthesizer

        # Run
        instance.auto_assign_transformers(data)

        # Assert
        nesreca_synthesizer.update_transformers.assert_called_once_with({'a': None, 'b': None})
        oseba_synthesizer.update_transformers.assert_called_once_with({'a': None, 'b': None})

    def test_auto_assign_transformers_missing_table(self):
        """Test it errors out when the passed table was not seen in the metadata."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        data = {'not_seen': pd.DataFrame({'col': [1, 2]})}

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nTable 'not_seen' is not present in the metadata"
        )
        with pytest.raises(ValueError, match=err_msg):
            instance.auto_assign_transformers(data)

    def test_get_transformers(self):
        """Test that each table of the data calls its single table get_transformers method."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_synthesizers['nesreca'] = Mock()
        instance._table_synthesizers['oseba'] = Mock()

        # Run
        instance.get_transformers('oseba')

        # Assert
        instance._table_synthesizers['nesreca'].get_transformers.assert_not_called()
        instance._table_synthesizers['oseba'].get_transformers.assert_called_once()

    def test_get_transformers_missing_table(self):
        """Test it errors out when the passed table name was not seen in the metadata."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nTable 'not_seen' is not present in the metadata."
        )
        with pytest.raises(ValueError, match=err_msg):
            instance.get_transformers('not_seen')

    def test_update_transformers(self):
        """Test that each table of the data calls its single table update_transformers method."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_synthesizers['nesreca'] = Mock()
        instance._table_synthesizers['oseba'] = Mock()

        # Run
        instance.update_transformers('oseba', {})

        # Assert
        instance._table_synthesizers['nesreca'].update_transformers.assert_not_called()
        instance._table_synthesizers['oseba'].update_transformers.assert_called_once_with({})

    def test_update_transformers_missing_table(self):
        """Test it errors out when the passed table name was not seen in the metadata."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            'The provided data does not match the metadata:'
            "\nTable 'not_seen' is not present in the metadata."
        )
        with pytest.raises(ValueError, match=err_msg):
            instance.update_transformers('not_seen', {})

    def test__model_tables(self):
        """Test that ``_model_tables`` raises a ``NotImplementedError``."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(0, 20, 2),
                'upravna_enota': np.arange(10),
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10),
                'id_nesreca': np.arange(10),
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
            }),
        }

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._model_tables(data)

    def test__assign_table_transformers(self):
        """Test the ``_assign_table_transformers`` method.

        Test that the function creates a dictionary mapping with the columns returned from
        ``_get_all_foreign_keys`` and maps them to the value ``None`` to avoid being transformed.
        Then calls ``update_transformers`` for the given ``synthesizer``.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance.validate = Mock()
        instance.metadata._get_all_foreign_keys = Mock(return_value=['a', 'b'])
        synthesizer = Mock()
        table_data = Mock()

        # Run
        instance._assign_table_transformers(synthesizer, 'oseba', table_data)

        # Assert
        synthesizer.auto_assign_transformers.assert_called_once_with(table_data)
        synthesizer.update_transformers.assert_called_once_with({'a': None, 'b': None})

    def test_preprocess(self):
        """Test that ``preprocess`` iterates over the ``data`` and preprocess it.

        This method should call ``instance.validate`` to validate the data, then
        iterate over the ``data`` dictionary and transform it using the ``synthesizer``
        ``preprocess`` method.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance.validate = Mock()
        instance.metadata._get_all_foreign_keys = Mock(return_value=['a', 'b'])
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(0, 20, 2),
                'upravna_enota': np.arange(10),
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10),
                'id_nesreca': np.arange(10),
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
            }),
        }

        synth_nesreca = Mock()
        synth_oseba = Mock()
        synth_upravna_enota = Mock()
        instance._table_synthesizers = {
            'nesreca': synth_nesreca,
            'oseba': synth_oseba,
            'upravna_enota': synth_upravna_enota
        }

        # Run
        result = instance.preprocess(data)

        # Assert
        assert result == {
            'nesreca': synth_nesreca._preprocess.return_value,
            'oseba': synth_oseba._preprocess.return_value,
            'upravna_enota': synth_upravna_enota._preprocess.return_value
        }
        instance.validate.assert_called_once_with(data)
        assert instance.metadata._get_all_foreign_keys.call_args_list == [
            call('nesreca'),
            call('oseba'),
            call('upravna_enota')
        ]

        synth_nesreca.auto_assign_transformers.assert_called_once_with(data['nesreca'])
        synth_nesreca._preprocess.assert_called_once_with(data['nesreca'])
        synth_nesreca.update_transformers.assert_called_once_with({'a': None, 'b': None})

        synth_oseba._preprocess.assert_called_once_with(data['oseba'])
        synth_oseba._preprocess.assert_called_once_with(data['oseba'])
        synth_oseba.update_transformers.assert_called_once_with({'a': None, 'b': None})

        synth_upravna_enota._preprocess.assert_called_once_with(data['upravna_enota'])
        synth_upravna_enota._preprocess.assert_called_once_with(data['upravna_enota'])
        synth_upravna_enota.update_transformers.assert_called_once_with({'a': None, 'b': None})

    def test_preprocess_int_columns(self):
        """Test the preprocess method.

        Ensure that data with column names as integers are not changed by
        preprocess.
        """
        # Setup
        metadata_dict = {
            'tables': {
                'first_table': {
                    'primary_key': '1',
                    'columns': {
                        '1': {'sdtype': 'id'},
                        '2': {'sdtype': 'categorical'},
                        'str': {'sdtype': 'categorical'}
                    }
                },
                'second_table': {
                    'columns': {
                        '3': {'sdtype': 'id'},
                        'str': {'sdtype': 'categorical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'first_table',
                    'parent_primary_key': '1',
                    'child_table_name': 'second_table',
                    'child_foreign_key': '3'
                }
            ]
        }
        metadata = MultiTableMetadata.load_from_dict(metadata_dict)
        instance = BaseMultiTableSynthesizer(metadata)
        instance.validate = Mock()
        instance._table_synthesizers = {
            'first_table': Mock(),
            'second_table': Mock()
        }
        multi_data = {
            'first_table': pd.DataFrame({
                1: ['abc', 'def', 'ghi'],
                2: ['x', 'a', 'b'],
                'str': ['John', 'Doe', 'John Doe'],
            }),
            'second_table': pd.DataFrame({
                3: ['abc', 'def', 'ghi'],
                'another': ['John', 'Doe', 'John Doe'],
            }),
        }

        # Run
        instance.preprocess(multi_data)

        # Assert
        corrected_frame = {
            'first_table': pd.DataFrame({
                1: ['abc', 'def', 'ghi'],
                2: ['x', 'a', 'b'],
                'str': ['John', 'Doe', 'John Doe'],
            }),
            'second_table': pd.DataFrame({
                3: ['abc', 'def', 'ghi'],
                'another': ['John', 'Doe', 'John Doe'],
            }),
        }

        pd.testing.assert_frame_equal(multi_data['first_table'], corrected_frame['first_table'])
        pd.testing.assert_frame_equal(multi_data['second_table'], corrected_frame['second_table'])

    @patch('sdv.multi_table.base.warnings')
    def test_preprocess_warning(self, mock_warnings):
        """Test that ``preprocess`` warns the user if the model has already been fitted."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance.validate = Mock()
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(0, 20, 2),
                'upravna_enota': np.arange(10),
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10),
                'id_nesreca': np.arange(10),
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
            }),
        }

        synth_nesreca = Mock()
        synth_oseba = Mock()
        synth_upravna_enota = Mock()
        instance._table_synthesizers = {
            'nesreca': synth_nesreca,
            'oseba': synth_oseba,
            'upravna_enota': synth_upravna_enota
        }
        instance._fitted = True

        # Run
        result = instance.preprocess(data)

        # Assert
        assert result == {
            'nesreca': synth_nesreca._preprocess.return_value,
            'oseba': synth_oseba._preprocess.return_value,
            'upravna_enota': synth_upravna_enota._preprocess.return_value
        }
        instance.validate.assert_called_once_with(data)
        synth_nesreca._preprocess.assert_called_once_with(data['nesreca'])
        synth_oseba._preprocess.assert_called_once_with(data['oseba'])
        synth_upravna_enota._preprocess.assert_called_once_with(data['upravna_enota'])
        mock_warnings.warn.assert_called_once_with(
            'This model has already been fitted. To use the new preprocessed data, '
            "please refit the model using 'fit' or 'fit_processed_data'."
        )

    @patch('sdv.multi_table.base.datetime')
    def test_fit_processed_data(self, mock_datetime, caplog):
        """Test that fit processed data calls ``_augment_tables`` and ``_model_tables``.

        Ensure that the ``fit_processed_data`` augments the tables and then models those using
        the ``_model_tables`` method. Then sets the state to fitted.
        """
        # Setup
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        instance = Mock(
            _fitted_sdv_version=None,
            _fitted_sdv_enterprise_version=None,
            _synthesizer_id='BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        )
        processed_data = {
            'table1': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']}),
            'table2': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']})
        }

        # Run
        with catch_sdv_logs(caplog, logging.INFO, 'MultiTableSynthesizer'):
            BaseMultiTableSynthesizer.fit_processed_data(instance, processed_data)

        # Assert
        instance._augment_tables.assert_called_once_with(processed_data)
        instance._model_tables.assert_called_once_with(instance._augment_tables.return_value)
        assert instance._fitted
        assert caplog.messages[0] == str({
            'EVENT': 'Fit processed data',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
            'TOTAL NUMBER OF TABLES': 2,
            'TOTAL NUMBER OF ROWS': 6,
            'TOTAL NUMBER OF COLUMNS': 4
        })

    def test_fit_processed_data_empty_table(self):
        """Test attributes are properly set when data is empty and that _fit is not called."""
        # Setup
        instance = Mock(
            _fitted_sdv_version=None,
            _fitted_sdv_enterprise_version=None
        )
        processed_data = {
            'table1': pd.DataFrame(),
            'table2': pd.DataFrame()
        }

        # Run
        BaseMultiTableSynthesizer.fit_processed_data(instance, processed_data)

        # Assert
        instance._fit.assert_not_called()
        assert instance._fitted
        assert instance._fitted_date
        assert instance._fitted_sdv_version

    def test_fit_processed_data_raises_version_error(self):
        """Test that fit_processed data  will raise a ``VersionError``."""
        # Setup
        instance = Mock(
            _fitted_sdv_version='1.0.0',
            _fitted_sdv_enterprise_version=None
        )
        instance.metadata = Mock()
        processed_data = {
            'table1': pd.DataFrame(),
            'table2': pd.DataFrame()
        }

        # Run and Assert
        error_msg = (
            f'You are currently on SDV version {version.public} but this synthesizer '
            'was created on version 1.0.0. Fitting this synthesizer again is not supported. '
            'Please create a new synthesizer.'
        )
        with pytest.raises(VersionError, match=error_msg):
            BaseMultiTableSynthesizer.fit_processed_data(instance, processed_data)

        # Assert
        instance.preprocess.assert_not_called()
        instance.fit_processed_data.assert_not_called()
        instance._check_metadata_updated.assert_not_called()

    @patch('sdv.multi_table.base.datetime')
    @patch('sdv.multi_table.base._validate_foreign_keys_not_null')
    def test_fit(self, mock_validate_foreign_keys_not_null, mock_datetime, caplog):
        """Test that it calls the appropriate methods."""
        # Setup
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        instance = Mock(
            _fitted_sdv_version=None,
            _fitted_sdv_enterprise_version=None,
            _synthesizer_id='BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        )
        instance.metadata = Mock()
        data = {
            'table1': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']}),
            'table2': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']})
        }

        # Run
        with catch_sdv_logs(caplog, logging.INFO, 'MultiTableSynthesizer'):
            BaseMultiTableSynthesizer.fit(instance, data)

        # Assert
        mock_validate_foreign_keys_not_null.assert_called_once_with(instance.metadata, data)
        instance.preprocess.assert_called_once_with(data)
        instance.fit_processed_data.assert_called_once_with(instance.preprocess.return_value)
        instance._check_metadata_updated.assert_called_once()
        assert caplog.messages[0] == str({
            'EVENT': 'Fit',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
            'TOTAL NUMBER OF TABLES': 2,
            'TOTAL NUMBER OF ROWS': 6,
            'TOTAL NUMBER OF COLUMNS': 4
        })

    def test_fit_raises_version_error(self):
        """Test that fit will raise a ``VersionError`` if the current version is bigger."""
        # Setup
        instance = Mock(
            _fitted_sdv_version='1.0.0',
            _fitted_sdv_enterprise_version=None
        )
        instance.metadata = Mock()
        data = {
            'table1': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']}),
            'table2': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']})
        }

        # Run and Assert
        error_msg = (
            f'You are currently on SDV version {version.public} but this synthesizer '
            'was created on version 1.0.0. Fitting this synthesizer again is not supported. '
            'Please create a new synthesizer.'
        )
        with pytest.raises(VersionError, match=error_msg):
            BaseMultiTableSynthesizer.fit(instance, data)

        # Assert
        instance.preprocess.assert_not_called()
        instance.fit_processed_data.assert_not_called()
        instance._check_metadata_updated.assert_not_called()

    def test_reset_sampling(self):
        """Test that ``reset_sampling`` resets the numpy seed and the synthesizers."""
        # Setup
        instance = Mock()
        instance._numpy_seed = object()
        users_mock = Mock()
        sessions_mock = Mock()
        transactions_mock = Mock()
        instance._table_synthesizers = {
            'users': users_mock,
            'sessions': sessions_mock,
            'transactions': transactions_mock,
        }

        # Run
        BaseMultiTableSynthesizer.reset_sampling(instance)

        # Assert
        assert instance._numpy_seed == 73251
        users_mock.reset_sampling.assert_called_once_with()
        sessions_mock.reset_sampling.assert_called_once_with()
        transactions_mock.reset_sampling.assert_called_once_with()

    def test__sample(self):
        """Test that ``_sample`` raises a ``NotImplementedError``."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._sample(scale=1.0)

    def test_sample_validate_input(self):
        """Test that SynthesizerInputError is raised if 'scale' is not >0.0."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._fitted = True
        instance._sample = Mock()
        scales = ['Test', True, -1.2, np.nan]

        # Run and Assert
        msg_1 = re.escape(
            "Invalid parameter for 'scale' (Test). Please provide a number that is >0.0."
        )
        msg_2 = re.escape(
            "Invalid parameter for 'scale' (True). Please provide a number that is >0.0."
        )
        msg_3 = re.escape(
            "Invalid parameter for 'scale' (-1.2). Please provide a number that is >0.0."
        )
        msg_4 = re.escape(
            "Invalid parameter for 'scale' (nan). Please provide a number that is >0.0."
        )
        err_msg = [msg_1, msg_2, msg_3, msg_4]

        for scale, msg in zip(scales, err_msg):
            with pytest.raises(SynthesizerInputError, match=msg):
                instance.sample(scale=scale)

    def test_sample_raises_sampling_error(self):
        """Test that ``sample`` will raise ``SamplingError`` when not fitted."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = (
            'This synthesizer has not been fitted. Please fit your synthesizer first before '
            'sampling synthetic data.'
        )
        with pytest.raises(SamplingError, match=error_msg):
            instance.sample(1)

    @patch('sdv.multi_table.base.datetime')
    def test_sample(self, mock_datetime, caplog):
        """Test that ``sample`` calls the ``_sample`` with the given arguments."""
        # Setup
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._fitted = True
        data = {
            'table1': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']}),
            'table2': pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Johanna', 'Doe']})
        }
        instance._sample = Mock(return_value=data)

        synth_id = 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        instance._synthesizer_id = synth_id

        # Run
        with catch_sdv_logs(caplog, logging.INFO, logger='MultiTableSynthesizer'):
            instance.sample(scale=1.5)

        # Assert
        instance._sample.assert_called_once_with(scale=1.5)
        assert caplog.messages[0] == str({
            'EVENT': 'Sample',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'BaseMultiTableSynthesizer',
            'SYNTHESIZER ID': 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
            'TOTAL NUMBER OF TABLES': 2,
            'TOTAL NUMBER OF ROWS': 6,
            'TOTAL NUMBER OF COLUMNS': 4
        })

    def test_get_learned_distributions_raises_an_unfitted_error(self):
        """Test that ``get_learned_distributions`` raises an error when model is not fitted."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = re.escape(
            "Distributions have not been learned yet. Please fit your model first using 'fit'."
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.get_learned_distributions('nesreca')

    def test_get_learned_distributions_raises_non_parametric_error(self):
        """Test that ``get_learned_distributions`` errors for non-parametric synthesizers."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_synthesizers['nesreca'] = CTGANSynthesizer(metadata.tables['nesreca'])

        # Run and Assert
        msg = re.escape(
            "Learned distributions are not available for the 'nesreca' table because it uses the "
            "'CTGANSynthesizer'."
        )
        with pytest.raises(SynthesizerInputError, match=msg):
            instance.get_learned_distributions('nesreca')

    def test_get_loss_values_bad_table_name(self):
        """Test the ``get_loss_values`` errors if bad ``table_name`` provided."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = "Table 'bad_table' is not present in the metadata."
        with pytest.raises(ValueError, match=error_msg):
            instance.get_loss_values('bad_table')

    def test_get_loss_values_unfitted_error(self):
        """Test the ``get_loss_values`` errors if synthesizer has not been fitted."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_synthesizers['nesreca'] = CTGANSynthesizer(metadata.tables['nesreca'])

        # Run and Assert
        error_msg = 'Loss values are not available yet. Please fit your synthesizer first.'
        with pytest.raises(NotFittedError, match=error_msg):
            instance.get_loss_values('nesreca')

    def test_get_loss_values_unsupported_synthesizer_error(self):
        """Test the ``get_loss_values`` errors if synthesizer not GAN-based."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_synthesizers['nesreca']._fitted = True

        # Run and Assert
        msg = re.escape(
            "Loss values are not available for table 'nesreca' "
            'because the table does not use a GAN-based model.'
        )
        with pytest.raises(SynthesizerInputError, match=msg):
            instance.get_loss_values('nesreca')

    def test_add_constraint_warning(self):
        """Test a warning is raised when the synthesizer had already been fitted."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._fitted = True

        # Run and Assert
        warn_msg = (
            "For these constraints to take effect, please refit the synthesizer using 'fit'."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            instance.add_constraints([])

    def test_add_constraints(self):
        """Test a list of constraints can be added to the synthesizer."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        metadata.add_column('nesreca', 'positive_int', sdtype='numerical')
        metadata.add_column('oseba', 'negative_int', sdtype='numerical')
        positive_constraint = {
            'constraint_class': 'Positive',
            'table_name': 'nesreca',
            'constraint_parameters': {
                'column_name': 'nesreca_val',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'table_name': 'oseba',
            'constraint_parameters': {
                'column_name': 'oseba_val',
                'strict_boundaries': False
            }
        }

        # Run
        instance.add_constraints([positive_constraint, negative_constraint])

        # Assert
        positive_constraint = {
            'constraint_class': 'Positive',
            'constraint_parameters': {
                'column_name': 'nesreca_val',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'constraint_parameters': {
                'column_name': 'oseba_val',
                'strict_boundaries': False
            }
        }
        output_nesreca = instance._table_synthesizers['nesreca'].get_constraints()
        assert output_nesreca == [positive_constraint]

        output_oseba = instance._table_synthesizers['oseba'].get_constraints()
        assert output_oseba == [negative_constraint]

    def test_add_constraints_unique(self):
        """Test an error is raised when a ``Unique`` constraint is passed."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        unique_constraint = {
            'constraint_class': 'Unique',
            'table_name': 'oseba',
            'constraint_parameters': {
                'column_name': 'id_nesreca',
            }
        }

        # Run and Assert
        err_msg = re.escape(
            "The constraint class 'Unique' is not currently supported for multi-table"
            ' synthesizers. Please remove the constraint for this synthesizer.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            instance.add_constraints([unique_constraint])

    def test_get_constraints(self):
        """Test a list of constraints is returned by the method."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        metadata.add_column('nesreca', 'positive_int', sdtype='numerical')
        metadata.add_column('oseba', 'negative_int', sdtype='numerical')
        positive_constraint = {
            'constraint_class': 'Positive',
            'table_name': 'nesreca',
            'constraint_parameters': {
                'column_name': 'nesreca_val',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'table_name': 'oseba',
            'constraint_parameters': {
                'column_name': 'oseba_val',
                'strict_boundaries': False
            }
        }
        constraints = [positive_constraint, negative_constraint]
        instance.add_constraints(constraints)

        # Run
        output = instance.get_constraints()

        # Assert
        assert output == constraints

    def test_add_constraints_missing_table_name(self):
        """Test error raised when ``table_name`` is missing."""
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3]})
        metadata = MultiTableMetadata()
        metadata.detect_table_from_dataframe('table', data)
        constraint = {'constraint_class': 'Inequality'}
        model = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        err_msg = re.escape(
            "A constraint is missing required parameter 'table_name'. "
            'Please add this parameter to your constraint definition.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            model.add_constraints([constraint])

    def test_load_custom_constraint_classes(self):
        """Test that the method calls the single table synthesizer's version of the method."""
        # Setup
        instance = Mock()
        table_synth_mock = Mock()
        instance._table_synthesizers = {'table': table_synth_mock}

        # Run
        BaseMultiTableSynthesizer.load_custom_constraint_classes(
            instance,
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )

        # Assert
        table_synth_mock.load_custom_constraint_classes.assert_called_once_with(
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )

    def test_load_custom_constraint_classes_multi_tables(self):
        """Check that ``load_custom_constraint_classes`` is called for every tables."""
        # Setup
        instance = Mock()
        table_synth_mock = Mock()
        table_synth_mock_2 = Mock()
        instance._table_synthesizers = {'table': table_synth_mock, 'table_2': table_synth_mock_2}

        # Run
        BaseMultiTableSynthesizer.load_custom_constraint_classes(
            instance,
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )

        # Assert
        table_synth_mock.load_custom_constraint_classes.assert_called_once_with(
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )
        table_synth_mock_2.load_custom_constraint_classes.assert_called_once_with(
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )

    def test_add_custom_constraint_class(self):
        """Test that this method calls the single table synthesizer's version of the method."""
        # Setup
        instance = Mock()
        constraint_mock = Mock()
        table_synth_mock = Mock()
        instance._table_synthesizers = {'table': table_synth_mock}

        # Run
        BaseMultiTableSynthesizer.add_custom_constraint_class(
            instance,
            constraint_mock,
            'custom'
        )

        # Assert
        table_synth_mock.add_custom_constraint_class.assert_called_once_with(
            constraint_mock,
            'custom'
        )

    def test_add_custom_constraint_class_multi_tables(self):
        """Check that ``add_custom_constraint_class`` is called for every tables."""
        # Setup
        instance = Mock()
        constraint_mock = Mock()
        table_synth_mock = Mock()
        table_synth_mock_2 = Mock()
        instance._table_synthesizers = {'table': table_synth_mock, 'table_2': table_synth_mock_2}

        # Run
        BaseMultiTableSynthesizer.add_custom_constraint_class(
            instance,
            constraint_mock,
            'custom'
        )

        # Assert
        table_synth_mock.add_custom_constraint_class.assert_called_once_with(
            constraint_mock,
            'custom'
        )
        table_synth_mock_2.add_custom_constraint_class.assert_called_once_with(
            constraint_mock,
            'custom'
        )

    @patch('sdv.multi_table.base.version')
    def test_get_info(self, mock_version):
        """Test the correct dictionary is returned.

        Check the return dictionary is valid both before and after fitting the synthesizer.

        Mocks:
            * Unfortunately, ``datetime`` can't be mocked directly. This link explains how to
            do it: https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
        """
        # Setup
        data = {'tab': pd.DataFrame({'col': [1, 2, 3]})}
        metadata = MultiTableMetadata()
        metadata.add_table('tab')
        metadata.add_column('tab', 'col', sdtype='numerical')
        mock_version.public = '1.0.0'
        mock_version.enterprise = None

        with patch('sdv.multi_table.base.datetime.datetime') as mock_date:
            mock_date.today.return_value = datetime(2023, 1, 23)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            synthesizer = HMASynthesizer(metadata)

            # Run
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'HMASynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': False,
                'last_fit_date': None,
                'fitted_sdv_version': None
            }

            # Run
            synthesizer.fit(data)
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'HMASynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': True,
                'last_fit_date': '2023-01-23',
                'fitted_sdv_version': '1.0.0'
            }

    @patch('sdv.multi_table.base.version')
    def test_get_info_with_enterprise(self, mock_version):
        """Test the correct dictionary is returned.

        Check the return dictionary is valid both before and after fitting the synthesizer.

        Mocks:
            * Unfortunately, ``datetime`` can't be mocked directly. This link explains how to
            do it: https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
        """
        # Setup
        data = {'tab': pd.DataFrame({'col': [1, 2, 3]})}
        metadata = MultiTableMetadata()
        metadata.add_table('tab')
        metadata.add_column('tab', 'col', sdtype='numerical')
        mock_version.public = '1.0.0'
        mock_version.enterprise = '1.1.0'

        with patch('sdv.multi_table.base.datetime.datetime') as mock_date:
            mock_date.today.return_value = datetime(2023, 1, 23)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            synthesizer = HMASynthesizer(metadata)

            # Run
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'HMASynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': False,
                'last_fit_date': None,
                'fitted_sdv_version': None
            }

            # Run
            synthesizer.fit(data)
            info = synthesizer.get_info()

            # Assert
            assert info == {
                'class_name': 'HMASynthesizer',
                'creation_date': '2023-01-23',
                'is_fit': True,
                'last_fit_date': '2023-01-23',
                'fitted_sdv_version': '1.0.0',
                'fitted_sdv_enterprise_version': '1.1.0'
            }

    @patch('sdv.multi_table.base.datetime')
    @patch('sdv.multi_table.base.cloudpickle')
    def test_save(self, cloudpickle_mock, mock_datetime, tmp_path, caplog):
        """Test that the synthesizer is saved correctly."""
        # Setup
        synthesizer = Mock(
            _synthesizer_id='BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        )
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'

        # Run
        filepath = tmp_path / 'output.pkl'
        with catch_sdv_logs(caplog, logging.INFO, 'MultiTableSynthesizer'):
            BaseMultiTableSynthesizer.save(synthesizer, filepath)

        # Assert
        cloudpickle_mock.dump.assert_called_once_with(synthesizer, ANY)
        assert caplog.messages[0] == str({
            'EVENT': 'Save',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        })

    @patch('sdv.multi_table.base.datetime')
    @patch('sdv.multi_table.base.generate_synthesizer_id')
    @patch('sdv.multi_table.base.check_synthesizer_version')
    @patch('sdv.multi_table.base.check_sdv_versions_and_warn')
    @patch('sdv.multi_table.base.cloudpickle')
    @patch('builtins.open', new_callable=mock_open)
    def test_load(self, mock_file, cloudpickle_mock,
                  mock_check_sdv_versions_and_warn, mock_check_synthesizer_version,
                  mock_generate_synthesizer_id, mock_datetime, caplog):
        """Test that the ``load`` method loads a stored synthesizer."""
        # Setup
        synthesizer_id = 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5'
        mock_datetime.datetime.now.return_value = '2024-04-19 16:20:10.037183'
        mock_generate_synthesizer_id.return_value = synthesizer_id
        synthesizer_mock = Mock(_fitted=False, _synthesizer_id=None)
        cloudpickle_mock.load.return_value = synthesizer_mock

        # Run
        with catch_sdv_logs(caplog, logging.INFO, 'MultiTableSynthesizer'):
            loaded_instance = BaseMultiTableSynthesizer.load('synth.pkl')

        # Assert
        mock_file.assert_called_once_with('synth.pkl', 'rb')
        mock_check_sdv_versions_and_warn.assert_called_once_with(loaded_instance)
        cloudpickle_mock.load.assert_called_once_with(mock_file.return_value)
        assert loaded_instance == synthesizer_mock
        mock_check_synthesizer_version.assert_called_once_with(synthesizer_mock)
        assert loaded_instance._synthesizer_id == synthesizer_id
        mock_generate_synthesizer_id.assert_called_once_with(synthesizer_mock)
        assert caplog.messages[0] == str({
            'EVENT': 'Load',
            'TIMESTAMP': '2024-04-19 16:20:10.037183',
            'SYNTHESIZER CLASS NAME': 'Mock',
            'SYNTHESIZER ID': 'BaseMultiTableSynthesizer_1.0.0_92aff11e9a5649d1a280990d1231a5f5',
        })

    @patch('builtins.open')
    @patch('sdv.multi_table.base.cloudpickle')
    def test_load_runtime_error(self, cloudpickle_mock, mock_open):
        """Test that the synthesizer's load method errors with the correct message."""
        # Setup
        cloudpickle_mock.load.side_effect = RuntimeError

        # Run and Assert
        err_msg = re.escape(
            'This synthesizer was created on a machine with GPU but the current machine is'
            ' CPU-only. This feature is currently unsupported. We recommend sampling on '
            'the same GPU-enabled machine.'
        )
        with pytest.raises(SamplingError, match=err_msg):
            BaseMultiTableSynthesizer.load('synth.pkl')
