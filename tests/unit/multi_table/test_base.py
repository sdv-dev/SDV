import re
from collections import defaultdict
from datetime import date, datetime
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.errors import SynthesizerInputError
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.multi_table.hma import HMASynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.errors import InvalidDataError
from tests.utils import get_multi_table_data, get_multi_table_metadata


class TestBaseMultiTableSynthesizer:

    def test__initialize_models(self):
        """Test that this method initializes the ``self._synthezier`` for each table.

        This tests that we iterate over the tables within the metadata and if there are
        ``table_parameters`` we use those to initialize the model.
        """
        # Setup
        instance = Mock()
        instance._table_synthesizers = {}
        instance._table_parameters = {
            'nesreca': {
                'default_distribution': 'gamma'
            }
        }
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
            call(metadata=instance.metadata.tables['nesreca'], default_distribution='gamma'),
            call(metadata=instance.metadata.tables['oseba']),
            call(metadata=instance.metadata.tables['upravna_enota'])
        ])

    def test___init__(self):
        """Test that when creating a new instance this sets the defaults.

        Test that the metadata object is being stored and also being validated. Afterwards, this
        calls the ``self._initialize_models`` which creates the initial instances of those.
        """
        # Setup
        metadata = get_multi_table_metadata()
        metadata.validate = Mock()

        # Run
        instance = BaseMultiTableSynthesizer(metadata)

        # Assert
        assert instance.metadata == metadata
        assert isinstance(instance._table_synthesizers['nesreca'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['oseba'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['upravna_enota'], GaussianCopulaSynthesizer)
        assert instance._table_parameters == defaultdict(dict)
        instance.metadata.validate.assert_called_once_with()

    def test_get_table_parameters_empty(self):
        """Test that this method returns an empty dictionary when there are no parameters."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_table_parameters('oseba')

        # Assert
        assert result == {}

    def test_get_table_parameters_has_parameters(self):
        """Test that this method returns a dictionary with the parameters."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._table_parameters['oseba'] = {'default_distribution': 'gamma'}

        # Run
        result = instance.get_table_parameters('oseba')

        # Assert
        assert result == {'default_distribution': 'gamma'}

    def test_get_parameters(self):
        """Test that the table's synthesizer parameters are being returned."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_parameters('oseba')

        # Assert
        assert result == {
            'default_distribution': 'beta',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'numerical_distributions': {}
        }

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
        assert instance._table_parameters['oseba'] == {'default_distribution': 'gamma'}
        assert instance.get_parameters('oseba') == {
            'default_distribution': 'gamma',
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'numerical_distributions': {}
        }

    def test_get_metadata(self):
        """Test that the metadata object is returned."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance.get_metadata()

        # Assert
        assert metadata == result

    def test__get_all_foreign_keys(self):
        """Test that this method returns all the foreign keys for a given table name."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance._get_all_foreign_keys('nesreca')

        # Assert
        assert result == ['upravna_enota']

    def test__validate_foreign_keys(self):
        """Test that when the data matches as expected there are no errors."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        data = get_multi_table_data()

        # Run
        result = instance._validate_foreign_keys(data)

        # Assert
        assert result is None

    def test__validate_foreign_keys_missing_keys(self):
        """Test that errors are being returned.

        When the values of the foreign keys are not within the values of the parent
        primary key, a list of errors must be returned indicating the values that are missing.
        """
        # Setup
        metadata = get_multi_table_metadata()
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(0, 20, 2),
                'upravna_enota': np.arange(10),
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10, 20),
                'id_nesreca': np.arange(10),
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
            }),
        }
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        result = instance._validate_foreign_keys(data)

        # Assert
        missing_upravna_enota = (
            'Relationships:\n'
            "Error: foreign key column 'upravna_enota' contains unknown references: "
            '(10, 11, 12, 13, 14, + more). '
            'All the values in this column must reference a primary key.\n'
            "Error: foreign key column 'id_nesreca' contains unknown references: (1, 3, 5, 7, 9)."
            ' All the values in this column must reference a primary key.'
        )
        assert result == missing_upravna_enota

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
                'id_nesreca': np.arange(10).astype(str),
                'upravna_enota': np.arange(10),
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10).astype(str),
                'id_nesreca': np.arange(10),
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10).astype(str),
            }),
        }

        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = re.escape(
            'The provided data does not match the metadata:\n'
            "Table: 'nesreca'\n"
            "Error: Invalid values found for numerical column 'id_nesreca': ['0', '1', '2', "
            "'+ 7 more']."
            "\n\nTable: 'oseba'\n"
            "Error: Invalid values found for numerical column 'upravna_enota': ['0', '1', '2', "
            "'+ 7 more']."
            "\n\nTable: 'upravna_enota'\n"
            "Error: Invalid values found for numerical column 'id_upravna_enota': ['0', '1', '2', "
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
            }),
            'oseba': pd.DataFrame({
                'upravna_enota': np.arange(10),
                'id_nesreca': np.arange(10),
            }),
            'upravna_enota': pd.DataFrame({
                'id_upravna_enota': np.arange(10),
            }),
        }
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = re.escape(
            'The provided data does not match the metadata:\n'
            'Relationships:\n'
            "Error: foreign key column 'id_nesreca' contains unknown references: (1, 3, 5, 7, 9). "
            'All the values in this column must reference a primary key.'
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
        with pytest.raises(InvalidDataError, match=err_msg):
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
        with pytest.raises(InvalidDataError, match=err_msg):
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
        with pytest.raises(InvalidDataError, match=err_msg):
            instance.update_transformers('not_seen', {})

    def test__fit(self):
        """Test that ``_fit`` raises a ``NotImplementedError``."""
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
            instance._fit(data)

    def test__skip_foreign_key_transformations(self):
        """Test the ``_skip_foreign_key_transformations`` method.

        Test that the function creates a dictionary mapping with the columns returned from
        ``_get_all_foreign_keys`` and maps them to the value ``None`` to avoid being transformed.
        Then calls ``update_transformers`` for the given ``synthesizer``.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance.validate = Mock()
        instance._get_all_foreign_keys = Mock(return_value=['a', 'b'])
        synthesizer = Mock()
        table_data = Mock()

        # Run
        instance._skip_foreign_key_transformations(synthesizer, 'oseba', table_data)

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
        instance._get_all_foreign_keys = Mock(return_value=['a', 'b'])
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
            'nesreca': synth_nesreca.preprocess.return_value,
            'oseba': synth_oseba.preprocess.return_value,
            'upravna_enota': synth_upravna_enota.preprocess.return_value
        }
        instance.validate.assert_called_once_with(data)
        assert instance._get_all_foreign_keys.call_args_list == [
            call('nesreca'),
            call('oseba'),
            call('upravna_enota')
        ]

        synth_nesreca.auto_assign_transformers.assert_called_once_with(data['nesreca'])
        synth_nesreca.preprocess.assert_called_once_with(data['nesreca'])
        synth_nesreca.update_transformers.assert_called_once_with({'a': None, 'b': None})

        synth_oseba.preprocess.assert_called_once_with(data['oseba'])
        synth_oseba.preprocess.assert_called_once_with(data['oseba'])
        synth_oseba.update_transformers.assert_called_once_with({'a': None, 'b': None})

        synth_upravna_enota.preprocess.assert_called_once_with(data['upravna_enota'])
        synth_upravna_enota.preprocess.assert_called_once_with(data['upravna_enota'])
        synth_upravna_enota.update_transformers.assert_called_once_with({'a': None, 'b': None})

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
            'nesreca': synth_nesreca.preprocess.return_value,
            'oseba': synth_oseba.preprocess.return_value,
            'upravna_enota': synth_upravna_enota.preprocess.return_value
        }
        instance.validate.assert_called_once_with(data)
        synth_nesreca.preprocess.assert_called_once_with(data['nesreca'])
        synth_oseba.preprocess.assert_called_once_with(data['oseba'])
        synth_upravna_enota.preprocess.assert_called_once_with(data['upravna_enota'])
        mock_warnings.warn.assert_called_once_with(
            'This model has already been fitted. To use the new preprocessed data, '
            "please refit the model using 'fit' or 'fit_processed_data'."
        )

    def test_fit_processed_data(self):
        """Test that fit processed data calls ``_fit`` and sets ``_fitted`` to ``True``."""
        # Setup
        instance = Mock()
        data = Mock()

        # Run
        BaseMultiTableSynthesizer.fit_processed_data(instance, data)

        # Assert
        instance._fit.assert_called_once_with(data)
        assert instance._fitted

    def test_fit(self):
        """Test that ``fit`` calls ``preprocess`` and then ``fit_processed_data``."""
        # Setup
        instance = Mock()
        data = Mock()

        # Run
        BaseMultiTableSynthesizer.fit(instance, data)

        # Assert
        instance.preprocess.assert_called_once_with(data)
        instance.fit_processed_data.assert_called_once_with(instance.preprocess.return_value)

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

    def test_validate_input_sample(self):
        """Test that ``_sample`` raises
        errors if scale parameter is not a number >=0.0.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
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

    def test_sample(self):
        """Test that ``sample`` calls the ``_sample`` with the given arguments."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._sample = Mock()

        # Run
        instance.sample(scale=1.5)

        # Assert
        instance._sample.assert_called_once_with(scale=1.5)

    def test_get_learned_distributions_raises_an_error(self):
        """Test that ``get_learned_distributions`` raises an error."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        error_msg = re.escape(
            "Distributions have not been learned yet. Please fit your model first using 'fit'."
        )
        with pytest.raises(ValueError, match=error_msg):
            instance.get_learned_distributions('nesreca')

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
        positive_constraint = {
            'constraint_class': 'Positive',
            'table_name': 'nesreca',
            'constraint_parameters': {
                'column_name': 'id_nesreca',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'table_name': 'oseba',
            'constraint_parameters': {
                'column_name': 'id_nesreca',
                'strict_boundaries': False
            }
        }

        # Run
        instance.add_constraints([positive_constraint, negative_constraint])

        # Assert
        positive_constraint = {
            'constraint_class': 'Positive',
            'constraint_parameters': {
                'column_name': 'id_nesreca',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'constraint_parameters': {
                'column_name': 'id_nesreca',
                'strict_boundaries': False
            }
        }
        assert instance._table_synthesizers['nesreca'].get_constraints() == [positive_constraint]
        assert instance._table_synthesizers['oseba'].get_constraints() == [negative_constraint]

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
        positive_constraint = {
            'constraint_class': 'Positive',
            'table_name': 'nesreca',
            'constraint_parameters': {
                'column_name': 'id_nesreca',
                'strict_boundaries': True
            }
        }
        negative_constraint = {
            'constraint_class': 'Negative',
            'table_name': 'oseba',
            'constraint_parameters': {
                'column_name': 'id_nesreca',
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
            'table',
            'path/to/file.py',
            ['Custom', 'Constr', 'UpperPlus']
        )

        # Assert
        table_synth_mock.load_custom_constraint_classes.assert_called_once_with(
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
            'table',
            constraint_mock,
            'custom'
        )

        # Assert
        table_synth_mock.add_custom_constraint_class.assert_called_once_with(
            constraint_mock,
            'custom'
        )

    def _pkg_mock(self, lib):
        if lib == 'sdv':
            class Distribution:
                version = '1.0.0'

            return Distribution

    @patch('pkg_resources.get_distribution')
    def test_get_info(self, pkg_mock):
        """Test the correct dictionary is returned.

        Check the return dictionary is valid both before and after fitting the synthesizer.

        Mocks:
            * Mock ``pkg_resources`` so we don't have to rewrite this test for every new release.
            * Unfortunately, ``datetime`` can't be mocked directly. This link explains how to
            do it: https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
        """
        # Setup
        data = {'tab': pd.DataFrame({'col': [1, 2, 3]})}
        pkg_mock.side_effect = self._pkg_mock
        metadata = MultiTableMetadata()
        metadata.add_table('tab')
        metadata.add_column('tab', 'col', sdtype='numerical')

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
