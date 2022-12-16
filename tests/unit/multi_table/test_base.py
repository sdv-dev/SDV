import re
from collections import defaultdict
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdv.multi_table.base import BaseMultiTableSynthesizer
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
            call(metadata=instance.metadata._tables['nesreca'], default_distribution='gamma'),
            call(metadata=instance.metadata._tables['oseba']),
            call(metadata=instance.metadata._tables['upravna_enota'])
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

    def test_update_table_parameters(self):
        """Test that the table's parameters are being updated.

        This test should ensure that the ``self._table_parameters`` for the given table
        are being updated with the given parameters, and also that the model is re-created
        and updated it's parameters as well.
        """
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run
        instance.update_table_parameters('oseba', {'default_distribution': 'gamma'})

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
        synth_nesreca.preprocess.assert_called_once_with(data['nesreca'])
        synth_oseba.preprocess.assert_called_once_with(data['oseba'])
        synth_upravna_enota.preprocess.assert_called_once_with(data['upravna_enota'])

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

    def test__sample(self):
        """Test that ``_sample`` raises a ``NotImplementedError``."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)

        # Run and Assert
        with pytest.raises(NotImplementedError, match=''):
            instance._sample(scale=1.0, randomize_samples=False)

    def test_sample(self):
        """Test that ``sample`` calls the ``_sample`` with the given arguments."""
        # Setup
        metadata = get_multi_table_metadata()
        instance = BaseMultiTableSynthesizer(metadata)
        instance._sample = Mock()

        # Run
        instance.sample(scale=1.5, randomize_samples=True)

        # Assert
        instance._sample.assert_called_once_with(scale=1.5, randomize_samples=True)
