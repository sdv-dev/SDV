import re
from collections import defaultdict
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.errors import InvalidDataError
from tests.utils import get_multi_table_metadata


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
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(10),
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

        # Run
        result = instance._validate_foreign_keys(data)

        # Assert
        assert result == None

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
            '(10, 11, 12, 13, 14, + more).\n'
            "Error: foreign key column 'id_nesreca' contains unknown references: (1, 3, 5, 7, 9)."
        )
        assert result == missing_upravna_enota

    def test_validate(self):
        """Test that no error is being raised when the data is valid."""
        # Setup
        metadata = get_multi_table_metadata()
        data = {
            'nesreca': pd.DataFrame({
                'id_nesreca': np.arange(10),
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
        instance.validate(data)

    def test_validate_missing_table(self):
        """Test that an error is being raised when there is a missing table in the dictionary."""
        # Setup
        metadata = get_multi_table_metadata()
        data = {
            'nesrecas': pd.DataFrame({
                'id_nesreca': np.arange(10),
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
        error_msg = "The provided data is missing the tables {'nesreca'}."
        with pytest.raises(InvalidDataError, match=error_msg):
            instance.validate(data)

    def test_validate_data_is_not_dataframe(self):
        """Test that an error is being raised when the data is not a dataframe."""
        # Setup
        metadata = get_multi_table_metadata()
        data = {
            'nesreca': pd.Series({
                'id_nesreca': np.arange(10),
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
            "Invalid values found for numerical column 'id_nesreca': ['0', '1', '2', '+ 7 more']."
            "\n\nTable: 'oseba'\n"
            "Invalid values found for numerical column 'upravna_enota': ['0', '1', '2', "
            "'+ 7 more']."
            "\n\nTable: 'upravna_enota'\n"
            "Invalid values found for numerical column 'id_upravna_enota': ['0', '1', '2', "
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
            "Error: foreign key column 'id_nesreca' contains unknown references: (1, 3, 5, 7, 9)."
        )
        with pytest.raises(InvalidDataError, match=error_msg):
            instance.validate(data)
