import io
from unittest.mock import MagicMock, Mock, mock_open, patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from tests.utils import DataFrameMatcher


class TestSingleTablePreset:

    def test___init__invalid_name(self):
        """Test the method with an invalid name.

        If the name isn't one of the expected presets, a ``ValueError`` should be thrown.
        """
        # Run and Assert
        with pytest.raises(ValueError, match=r"'name' must be one of *"):
            SingleTablePreset(metadata=SingleTableMetadata(), name='invalid')

    @patch('sdv.lite.single_table.rdt.transformers')
    @patch('sdv.lite.single_table.GaussianCopulaSynthesizer')
    def test__init__speed_passes_correct_parameters(self, gaussian_copula_mock, transformers_mock):
        """Tests the method with the speed preset.

        The method should pass the parameters to the ``GaussianCopulaSynthesizer`` class.
        """
        # Setup
        metadata_mock = MagicMock(spec_set=SingleTableMetadata)

        # Run
        SingleTablePreset(metadata=metadata_mock, name='FAST_ML')

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            metadata=metadata_mock,
            default_distribution='norm',
            enforce_rounding=False,
            locales=None
        )

    @patch('sdv.lite.single_table.rdt.transformers')
    @patch('sdv.lite.single_table.GaussianCopulaSynthesizer')
    def test__init__passes_correct_locales(self, gaussian_copula_mock, transformers_mock):
        """Tests the method with locales.

        The method should pass the locales parameter to the ``GaussianCopulaSynthesizer`` class.
        """
        # Setup
        metadata_mock = MagicMock(spec_set=SingleTableMetadata)

        # Run
        SingleTablePreset(metadata=metadata_mock, name='FAST_ML', locales=['en_US', 'fr_CA'])

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            metadata=metadata_mock,
            default_distribution='norm',
            enforce_rounding=False,
            locales=['en_US', 'fr_CA']
        )

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_parameters(self, mock_data_processor):
        """Test that it returns every ``init`` parameter without the ``metadata``."""
        # Setup
        metadata = Mock()
        instance = SingleTablePreset(metadata, name='FAST_ML')

        # Run
        parameters = instance.get_parameters()

        # Assert
        assert 'metadata' not in parameters
        assert parameters == {
            'default_distribution': 'norm',
            'enforce_min_max_values': True,
            'enforce_rounding': False,
            'locales': None,
            'numerical_distributions': {}
        }

    @patch('sdv.single_table.base.DataProcessor')
    def test_get_metadata(self, mock_data_processor):
        """Test that it returns the ``metadata`` object."""
        # Setup
        metadata = Mock()
        instance = SingleTablePreset(metadata, 'FAST_ML')

        # Run
        result = instance.get_metadata()

        # Assert
        assert result == metadata

    def test_fit(self):
        """Test that the synthesizer's fit method is called with the expected args."""
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {}}
        synthesizer = Mock()
        synthesizer._metadata = metadata
        preset = Mock()
        preset._synthesizer = synthesizer

        # Run
        SingleTablePreset.fit(preset, pd.DataFrame())

        # Assert
        synthesizer.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame()))

    def test_fit_null_column_true(self):
        """Test the method with modeling null columns.

        Expect that the synthesizer's fit method is called with the expected args when
        ``_null_column`` is set to ``True``.
        """
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {}}
        synthesizer = Mock()
        synthesizer._metadata = metadata
        preset = Mock()
        preset._synthesizer = synthesizer

        # Run
        SingleTablePreset.fit(preset, pd.DataFrame())

        # Assert
        synthesizer.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame()))

    def test_fit_with_null_values(self):
        """Test the method with null values.

        Expect that the model's fit method is called with the expected args, and that
        the null percentage is calculated correctly.
        """
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {'a': {}}}
        synthesizer = Mock()
        synthesizer._metadata = metadata
        preset = Mock()
        preset._synthesizer = synthesizer

        data = {'a': [1, 2, np.nan]}

        # Run
        SingleTablePreset.fit(preset, pd.DataFrame(data))

        # Assert
        synthesizer.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame(data)))

    def test_sample(self):
        """Test that the synthesizer's sample method is called with the expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer

        # Run
        SingleTablePreset.sample(preset, 5)

        # Assert
        synthesizer.sample.assert_called_once_with(5, 100, None, None)

    def test_sample_from_conditions(self):
        """Test that ``sample_from_conditions`` is called with the expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer
        conditions = [Mock()]

        # Run
        SingleTablePreset.sample_from_conditions(preset, conditions)

        # Assert
        synthesizer.sample_from_conditions.assert_called_once_with(conditions, 100, None, None)

    def test_sample_from_conditions_with_max_tries(self):
        """Test the method with max tries.

        Expect that the synthesizer's ``sample_from_conditions`` is called with the expected args.
        """
        # Setup
        synthesizer = MagicMock(spec=GaussianCopulaSynthesizer)
        preset = Mock()
        preset._synthesizer = synthesizer
        conditions = [Mock()]

        # Run
        SingleTablePreset.sample_from_conditions(
            preset,
            conditions,
            max_tries_per_batch=2,
            batch_size=5
        )

        # Assert
        synthesizer.sample_from_conditions.assert_called_once_with(conditions, 2, 5, None)

    def test_sample_remaining_columns(self):
        """Test the synthesizer's ``sample_remaining_columns`` is called with expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer
        conditions = pd.DataFrame({'a': [1, 2, 3]})

        # Run
        SingleTablePreset.sample_remaining_columns(preset, conditions)

        # Assert
        synthesizer.sample_remaining_columns.assert_called_once_with(conditions, 100, None, None)

    def test_sample_remaining_columns_with_max_tries(self):
        """Test the method with max tries.

        Expect that the synthesizer's ``sample_remaining_columns`` is called with the expected
        args.
        """
        # Setup
        synthesizer = MagicMock(spec=GaussianCopulaSynthesizer)
        preset = Mock()
        preset._synthesizer = synthesizer
        conditions = pd.DataFrame({'a': [1, 2, 3]})

        # Run
        SingleTablePreset.sample_remaining_columns(
            preset, conditions, max_tries_per_batch=2, batch_size=5)

        # Assert
        synthesizer.sample_remaining_columns.assert_called_once_with(conditions, 2, 5, None)

    def test_list_available_presets(self):
        """Tests that the method prints all the available presets."""
        # Setup
        out = io.StringIO()
        expected = (
            "Available presets:\n{'FAST_ML': 'Use this preset to minimize the time "
            "needed to create a synthetic data model.'}\n\nSupply the desired "
            'preset using the `name` parameter.\n\nHave any requests for '
            'custom presets? Contact the SDV team to learn more an SDV Premium license.'
        )

        # Run
        SingleTablePreset.list_available_presets(out)

        # Assert
        assert out.getvalue().strip() == expected

    @patch('sdv.lite.single_table.cloudpickle')
    def test_save(self, cloudpickle_mock, tmp_path):
        """Test that the synthesizer's save method is called with the expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer
        open_mock = mock_open(read_data=cloudpickle.dumps('test'))

        # Run
        with patch('sdv.lite.single_table.open', open_mock):
            SingleTablePreset.save(preset, tmp_path)

        # Assert
        open_mock.assert_called_once_with(tmp_path, 'wb')
        cloudpickle_mock.dump.assert_called_once_with(preset, open_mock())

    @patch('sdv.lite.single_table.cloudpickle')
    def test_load(self, cloudpickle_mock):
        """Test that the synthesizer's load method is called with the expected args."""
        # Setup
        default_synthesizer = Mock()
        SingleTablePreset._default_synthesizer = default_synthesizer
        open_mock = mock_open(read_data=cloudpickle.dumps('test'))

        # Run
        with patch('sdv.lite.single_table.open', open_mock):
            loaded = SingleTablePreset.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == cloudpickle_mock.load.return_value

    def test___repr__(self):
        """Test that a string of format 'SingleTablePreset(name=<name>)' is returned"""
        # Setup
        instance = SingleTablePreset(metadata=SingleTableMetadata(), name='FAST_ML')

        # Run
        res = repr(instance)

        # Assert
        assert res == 'SingleTablePreset(name=FAST_ML)'
