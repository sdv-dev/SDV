import io
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, mock_open, patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from tests.utils import DataFrameMatcher


class TestTabularPreset:

    def test___init__invalid_name(self):
        """Test the method with an invalid name.

        If the name isn't one of the expected presets, a ``ValueError`` should be thrown.
        """
        # Run and Assert
        with pytest.raises(ValueError, match=r'`name` must be one of *'):
            SingleTablePreset(metadata=SingleTableMetadata(), name='invalid')

    @patch('sdv.lite.single_table.rdt.transformers')
    @patch('sdv.lite.single_table.GaussianCopulaSynthesizer')
    def test__init__speed_passes_correct_parameters(self, gaussian_copula_mock, transformers_mock):
        """Tests the method with the speed preset.

        The method should pass the parameters to the ``GaussianCopulaSynthesizer`` class.
        """
        # Setup
        metadata_mock = MagicMock(spec_set=SingleTableMetadata)
        processor_mock = Mock()
        gaussian_copula_mock.return_value._data_processor = processor_mock

        # Run
        SingleTablePreset(metadata=metadata_mock, name='FAST_ML')

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            metadata=metadata_mock,
            default_distribution='norm',
            enforce_rounding=False,
        )
        processor_mock._transformers_by_sdtype.update.assert_called_once_with({
            'numerical': transformers_mock.FloatFormatter(
                missing_value_replacement=None,
                model_missing_values=False,
                enforce_min_max_values=True,
            ),
            'categorical': transformers_mock.FrequencyEncoder(add_noise=True),
            'boolean': transformers_mock.BinaryEncoder(
                missing_value_replacement=None,
                model_missing_values=False
            ),
            'datetime': transformers_mock.UnixTimestampEncoder(
                missing_value_replacement=None,
                model_missing_values=False
            )
        })

    @patch('sdv.lite.single_table.rdt.transformers')
    @patch('sdv.lite.single_table.GaussianCopulaSynthesizer')
    def test__init__with_constraints(self, gaussian_copula_mock, transformers_mock):
        """Tests the method with constraints."""
        # Setup
        constraints = [Mock()]
        metadata = Mock()
        processor_mock = Mock()
        gaussian_copula_mock.return_value._data_processor = processor_mock
        metadata_dict = {'constraints': constraints}
        metadata.to_dict.return_value = metadata_dict

        # Run
        preset = SingleTablePreset(metadata=metadata, name='FAST_ML')

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            metadata=metadata,
            default_distribution='norm',
            enforce_rounding=False,
        )
        processor_mock._transformers_by_sdtype.update.assert_called_once_with({
            'numerical': transformers_mock.FloatFormatter(
                missing_value_replacement='mean',
                model_missing_values=False,
                enforce_min_max_values=True,
            ),
            'categorical': transformers_mock.FrequencyEncoder(add_noise=True),
            'boolean': transformers_mock.BinaryEncoder(
                missing_value_replacement=-1,
                model_missing_values=False
            ),
            'datetime': transformers_mock.UnixTimestampEncoder(
                missing_value_replacement='mean',
                model_missing_values=False
            )
        })
        assert preset._null_column is True

    def test_fit(self):
        """Test that the synthesizer's fit method is called with the expected args."""
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {}}
        synthesizer = Mock()
        synthesizer._metadata = metadata
        preset = Mock()
        preset._synthesizer = synthesizer
        preset._null_percentages = None

        # Run
        SingleTablePreset.fit(preset, pd.DataFrame())

        # Assert
        synthesizer.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame()))
        assert preset._null_percentages is None

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
        preset._null_column = True
        preset._null_percentages = None

        # Run
        SingleTablePreset.fit(preset, pd.DataFrame())

        # Assert
        synthesizer.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame()))
        assert preset._null_percentages is None

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
        preset._null_column = False
        preset._null_percentages = None

        data = {'a': [1, 2, np.nan]}

        # Run
        SingleTablePreset.fit(preset, pd.DataFrame(data))

        # Assert
        synthesizer.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame(data)))
        assert preset._null_percentages == {'a': 1.0 / 3}

    def test__postprocess_sampled_with_null_values(self):
        """Test the method with null percentages.

        Expect that null values are inserted back into the sampled data..
        """
        # Setup
        sampled = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        preset = Mock()
        # Convoluted example - 100% percent chance of nulls to make test deterministic.
        preset._null_percentages = {'a': 1}

        # Run
        sampled_with_nulls = SingleTablePreset._postprocess_sampled(preset, sampled)

        # Assert
        assert sampled_with_nulls['a'].isna().sum() == 5

    def test_sample(self):
        """Test that the synthesizer's sample method is called with the expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer
        preset._null_percentages = None

        # Run
        SingleTablePreset.sample(preset, 5)

        # Assert
        synthesizer.sample.assert_called_once_with(5, True, 100, None, None, None)

    def test_sample_conditions(self):
        """Test that the synthesizer's ``sample_conditions`` is called with the expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer
        preset._null_percentages = None
        conditions = [Mock()]

        # Run
        SingleTablePreset.sample_conditions(preset, conditions)

        # Assert
        synthesizer.sample_conditions.assert_called_once_with(conditions, 100, None, True, None)

    def test_sample_conditions_with_max_tries(self):
        """Test the method with max tries.

        Expect that the synthesizer's ``sample_conditions`` is called with the expected args.
        """
        # Setup
        synthesizer = MagicMock(spec=GaussianCopulaSynthesizer)
        preset = Mock()
        preset._synthesizer = synthesizer
        preset._null_percentages = None
        conditions = [Mock()]

        # Run
        SingleTablePreset.sample_conditions(
            preset,
            conditions,
            max_tries_per_batch=2,
            batch_size=5
        )

        # Assert
        synthesizer.sample_conditions.assert_called_once_with(conditions, 2, 5, True, None)

    def test_sample_remaining_columns(self):
        """Test the synthesizer's ``sample_remaining_columns`` is called with expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer
        preset._null_percentages = None
        conditions = pd.DataFrame({'a': [1, 2, 3]})

        # Run
        SingleTablePreset.sample_remaining_columns(preset, conditions)

        # Assert
        synthesizer.sample_remaining_columns.assert_called_once_with(
            conditions, 100, None, True, None)

    def test_sample_remaining_columns_with_max_tries(self):
        """Test the method with max tries.

        Expect that the synthesizer's ``sample_remaining_columns`` is called with the expected
        args.
        """
        # Setup
        synthesizer = MagicMock(spec=GaussianCopulaSynthesizer)
        preset = Mock()
        preset._synthesizer = synthesizer
        preset._null_percentages = None
        conditions = pd.DataFrame({'a': [1, 2, 3]})

        # Run
        SingleTablePreset.sample_remaining_columns(
            preset, conditions, max_tries_per_batch=2, batch_size=5)

        # Assert
        synthesizer.sample_remaining_columns.assert_called_once_with(conditions, 2, 5, True, None)

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
    def test_save(self, cloudpickle_mock):
        """Test that the synthesizer's save method is called with the expected args."""
        # Setup
        synthesizer = Mock()
        preset = Mock()
        preset._synthesizer = synthesizer
        open_mock = mock_open(read_data=cloudpickle.dumps('test'))

        # Run
        with TemporaryDirectory() as temp_dir:
            with patch('sdv.lite.single_table.open', open_mock):
                SingleTablePreset.save(preset, temp_dir)

        # Assert
        open_mock.assert_called_once_with(temp_dir, 'wb')
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
