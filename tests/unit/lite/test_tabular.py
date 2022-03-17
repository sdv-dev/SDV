import io
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.lite import TabularPreset
from sdv.tabular import GaussianCopula
from tests.utils import DataFrameMatcher


class TestTabularPreset:

    def test___init__missing_optimize_for(self):
        """Test the ``TabularPreset.__init__`` method with no parameters.

        Side Effects:
        - ValueError should be thrown
        """
        # Run and Assert
        with pytest.raises(
            ValueError,
            match=('You must provide the name of a preset using the `optimize_for` parameter. '
                   r'Use `TabularPreset.list_available_presets\(\)` to browse through '
                   'the options.')):
            TabularPreset()

    def test___init__invalid_optimize_for(self):
        """Test the ``TabularPreset.__init__`` method with an invalid arg value.

        Input:
        - optimize_for = invalid parameter

        Side Effects:
        - ValueError should be thrown
        """
        # Run and Assert
        with pytest.raises(ValueError, match=r'`optimize_for` must be one of *'):
            TabularPreset(optimize_for='invalid')

    @patch('sdv.lite.tabular.GaussianCopula', spec_set=GaussianCopula)
    def test__init__speed_passes_correct_parameters(self, gaussian_copula_mock):
        """Tests the ``TabularPreset.__init__`` method with the speed preset.

        The method should pass the parameters to the ``GaussianCopula`` class.

        Input:
        - optimize_for = speed
        Side Effects:
        - GaussianCopula should receive the correct parameters
        """
        # Run
        TabularPreset(optimize_for='SPEED')

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            table_metadata=None,
            categorical_transformer='categorical',
            default_distribution='gaussian',
            rounding=None,
        )
        metadata = gaussian_copula_mock.return_value._metadata
        assert metadata._dtype_transformers.update.call_count == 1

    def test_fit(self):
        """Test the ``TabularPreset.fit`` method.

        Expect that the model's fit method is called with the expected args.

        Input:
        - fit data

        Side Effects:
        - The model's fit method is called with the same data.
        """
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {}}
        model = Mock()
        model._metadata = metadata
        preset = Mock()
        preset._model = model
        preset._null_percentages = None

        # Run
        TabularPreset.fit(preset, pd.DataFrame())

        # Assert
        model.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame()))
        assert preset._null_percentages == {}

    def test_fit_with_null_values(self):
        """Test the ``TabularPreset.fit`` method with null values.

        Expect that the model's fit method is called with the expected args, and that
        the null percentage is calculated correctly.

        Input:
        - fit data

        Side Effects:
        - The model's fit method is called with the same data.
        """
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {'a': {}}}
        model = Mock()
        model._metadata = metadata
        preset = Mock()
        preset._model = model
        preset._null_percentages = None

        data = {'a': [1, 2, np.nan]}

        # Run
        TabularPreset.fit(preset, pd.DataFrame(data))

        # Assert
        model.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame(data)))
        assert preset._null_percentages == {'a': 1.0 / 3}

    def test_sample(self):
        """Test the ``TabularPreset.sample`` method.

        Expect that the model's sample method is called with the expected args.

        Input:
        - num_rows=5

        Side Effects:
        - The model's sample method is called with the same data.
        """
        # Setup
        model = Mock()
        preset = Mock()
        preset._model = model
        preset._null_percentages = None

        # Run
        TabularPreset.sample(preset, 5)

        # Assert
        model.sample.assert_called_once_with(5)

    def test_sample_with_null_values(self):
        """Test the ``TabularPreset.sample`` method with null percentages.

        Expect that the model's sample method is called with the expected args, and that
        null values are inserted back into the sampled data.

        Input:
        - num_rows=5

        Side Effects:
        - The model's sample method is called with the expected number of rows.
        """
        # Setup
        model = Mock()
        model.sample.return_value = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        preset = Mock()
        preset._model = model
        # Convoluted example - 100% percent chance of nulls to make test deterministic.
        preset._null_percentages = {'a': 1}

        # Run
        sampled = TabularPreset.sample(preset, 5)

        # Assert
        model.sample.assert_called_once_with(5)
        assert sampled['a'].isna().sum() == 5

    def test_list_available_presets(self):
        """Tests the ``TabularPreset.list_available_presets`` method.

        This method should print all the available presets.

        Side Effects:
        - The available presets should be printed.
        """
        # Setup
        out = io.StringIO()
        expected = ('Available presets:\n{\'SPEED\': \'Use this preset to minimize the time '
                    'needed to create a synthetic data model.\'}\n\nSupply the desired '
                    'preset using the `opimize_for` parameter.\n\nHave any requests for '
                    'custom presets? Contact the SDV team to learn more an SDV Premium license.')

        # Run
        TabularPreset(optimize_for='SPEED').list_available_presets(out)

        # Assert
        assert out.getvalue().strip() == expected
