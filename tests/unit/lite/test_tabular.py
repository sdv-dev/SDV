import io
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.lite import TabularPreset
from sdv.metadata import Table
from sdv.tabular import GaussianCopula
from tests.utils import DataFrameMatcher


class TestTabularPreset:

    def test___init__missing_name(self):
        """Test the ``TabularPreset.__init__`` method with no parameters.

        Side Effects:
        - ValueError should be thrown
        """
        # Run and Assert
        with pytest.raises(
            ValueError,
            match=('You must provide the name of a preset using the `name` parameter. '
                   r'Use `TabularPreset.list_available_presets\(\)` to browse through '
                   'the options.')):
            TabularPreset()

    def test___init__invalid_name(self):
        """Test the ``TabularPreset.__init__`` method with an invalid arg value.

        Input:
        - name = invalid parameter

        Side Effects:
        - ValueError should be thrown
        """
        # Run and Assert
        with pytest.raises(ValueError, match=r'`name` must be one of *'):
            TabularPreset(name='invalid')

    @patch('sdv.lite.tabular.GaussianCopula', spec_set=GaussianCopula)
    def test__init__speed_passes_correct_parameters(self, gaussian_copula_mock):
        """Tests the ``TabularPreset.__init__`` method with the speed preset.

        The method should pass the parameters to the ``GaussianCopula`` class.

        Input:
        - name of the speed preset
        Side Effects:
        - GaussianCopula should receive the correct parameters
        """
        # Run
        TabularPreset(name='FAST_ML')

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            table_metadata=None,
            constraints=None,
            categorical_transformer='categorical_fuzzy',
            default_distribution='gaussian',
            rounding=None,
        )
        metadata = gaussian_copula_mock.return_value._metadata
        assert metadata._dtype_transformers.update.call_count == 1

    @patch('sdv.lite.tabular.GaussianCopula', spec_set=GaussianCopula)
    def test__init__with_metadata(self, gaussian_copula_mock):
        """Tests the ``TabularPreset.__init__`` method with the speed preset.

        The method should pass the parameters to the ``GaussianCopula`` class.

        Input:
        - name of the speed preset
        Side Effects:
        - GaussianCopula should receive the correct parameters
        """
        # Setup
        metadata = MagicMock(spec_set=Table)

        # Run
        TabularPreset(name='FAST_ML', metadata=metadata)

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            table_metadata=metadata.to_dict(),
            constraints=None,
            categorical_transformer='categorical_fuzzy',
            default_distribution='gaussian',
            rounding=None,
        )

    @patch('sdv.lite.tabular.GaussianCopula', spec_set=GaussianCopula)
    def test__init__with_constraints(self, gaussian_copula_mock):
        """Tests the ``TabularPreset.__init__`` method with constraints.

        The constraints should be added to the metadata.

        Input:
        - constraints

        Side Effects:
        - GaussianCopula should receive args, including the constraints.
        """
        # Setup
        constraint = Mock()

        # Run
        preset = TabularPreset(name='FAST_ML', metadata=None, constraints=[constraint])

        # Assert
        gaussian_copula_mock.assert_called_once_with(
            table_metadata=None,
            constraints=[constraint],
            categorical_transformer='categorical_fuzzy',
            default_distribution='gaussian',
            rounding=None,
        )
        metadata = gaussian_copula_mock.return_value._metadata
        assert metadata._dtype_transformers.update.call_count == 1
        assert preset._null_column is True

    @patch('sdv.lite.tabular.GaussianCopula', spec_set=GaussianCopula)
    def test__init__with_constraints_and_metadata(self, gaussian_copula_mock):
        """Tests the ``TabularPreset.__init__`` method with constraints and metadata.

        The constraints should be added to the metadata.

        Input:
        - constraints
        - metadata

        Side Effects:
        - GaussianCopula should receive metadata with the constraints added.
        """
        # Setup
        metadata = {'name': 'test_table', 'fields': []}
        constraint = Mock()

        # Run
        preset = TabularPreset(name='FAST_ML', metadata=metadata, constraints=[constraint])

        # Assert
        expected_metadata = metadata.copy()
        expected_metadata['constraints'] = [constraint.to_dict.return_value]

        gaussian_copula_mock.assert_called_once_with(
            table_metadata=expected_metadata,
            constraints=None,
            categorical_transformer='categorical_fuzzy',
            default_distribution='gaussian',
            rounding=None,
        )
        metadata = gaussian_copula_mock.return_value._metadata
        assert metadata._dtype_transformers.update.call_count == 1
        assert preset._null_column is True

    def test_fit(self):
        """Test the ``TabularPreset.fit`` method.

        Expect that the model's fit method is called with the expected args.

        Input:
        - fit data

        Side Effects:
        - The model's ``fit`` method is called with the same data.
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
        assert preset._null_percentages is None

    def test_fit_null_column_True(self):
        """Test the ``TabularPreset.fit`` method with modeling null columns.

        Expect that the model's fit method is called with the expected args when
        ``_null_column`` is set to ``True``.

        Setup:
        - _null_column is True

        Input:
        - fit data

        Side Effects:
        - The model's ``fit`` method is called with the same data.
        - ``_null_percentages`` is ``None``
        """
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {}}
        model = Mock()
        model._metadata = metadata
        preset = Mock()
        preset._model = model
        preset._null_column = True
        preset._null_percentages = None

        # Run
        TabularPreset.fit(preset, pd.DataFrame())

        # Assert
        model.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame()))
        assert preset._null_percentages is None

    def test_fit_with_null_values(self):
        """Test the ``TabularPreset.fit`` method with null values.

        Expect that the model's fit method is called with the expected args, and that
        the null percentage is calculated correctly.

        Input:
        - fit data

        Side Effects:
        - The model's ``fit`` method is called with the same data.
        """
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {'fields': {'a': {}}}
        model = Mock()
        model._metadata = metadata
        preset = Mock()
        preset._model = model
        preset._null_column = False
        preset._null_percentages = None

        data = {'a': [1, 2, np.nan]}

        # Run
        TabularPreset.fit(preset, pd.DataFrame(data))

        # Assert
        model.fit.assert_called_once_with(DataFrameMatcher(pd.DataFrame(data)))
        assert preset._null_percentages == {'a': 1.0 / 3}

    def test__postprocess_sampled_with_null_values(self):
        """Test the ``TabularPreset._postprocess_sampled`` method with null percentages.

        Expect that null values are inserted back into the sampled data.

        Setup:
        - _null_percentages has a valid entry

        Input:
        - sampled data

        Output:
        - sampled data with nulls that represents the expected null percentages.
        """
        # Setup
        sampled = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        preset = Mock()
        # Convoluted example - 100% percent chance of nulls to make test deterministic.
        preset._null_percentages = {'a': 1}

        # Run
        sampled_with_nulls = TabularPreset._postprocess_sampled(preset, sampled)

        # Assert
        assert sampled_with_nulls['a'].isna().sum() == 5

    def test_sample(self):
        """Test the ``TabularPreset.sample`` method.

        Expect that the model's sample method is called with the expected args.

        Input:
        - num_rows=5

        Side Effects:
        - The model's ``sample`` method is called with the same data.
        """
        # Setup
        model = Mock()
        preset = Mock()
        preset._model = model
        preset._null_percentages = None

        # Run
        TabularPreset.sample(preset, 5)

        # Assert
        model.sample.assert_called_once_with(5, True, None, None, None)

    def test_sample_conditions(self):
        """Test the ``TabularPreset.sample_conditions`` method.

        Expect that the model's sample_conditions method is called with the expected args.

        Input:
        - num_rows=5

        Side Effects:
        - The model's ``sample_conditions`` method is called with the same data.
        """
        # Setup
        model = Mock()
        preset = Mock()
        preset._model = model
        preset._null_percentages = None
        conditions = [Mock()]

        # Run
        TabularPreset.sample_conditions(preset, conditions)

        # Assert
        model.sample_conditions.assert_called_once_with(conditions, 100, None, True, None)

    def test_sample_remaining_columns(self):
        """Test the ``TabularPreset.sample_remaining_columns`` method.

        Expect that the model's sample_remaining_columns method is called with the expected args.

        Input:
        - num_rows=5

        Side Effects:
        - The model's ``sample_remaining_columns`` method is called with the same data.
        """
        # Setup
        model = Mock()
        preset = Mock()
        preset._model = model
        preset._null_percentages = None
        conditions = pd.DataFrame({'a': [1, 2, 3]})

        # Run
        TabularPreset.sample_remaining_columns(preset, conditions)

        # Assert
        model.sample_remaining_columns.assert_called_once_with(conditions, 100, None, True, None)

    def test_list_available_presets(self):
        """Tests the ``TabularPreset.list_available_presets`` method.

        This method should print all the available presets.

        Side Effects:
        - The available presets should be printed.
        """
        # Setup
        out = io.StringIO()
        expected = ('Available presets:\n{\'FAST_ML\': \'Use this preset to minimize the time '
                    'needed to create a synthetic data model.\'}\n\nSupply the desired '
                    'preset using the `name` parameter.\n\nHave any requests for '
                    'custom presets? Contact the SDV team to learn more an SDV Premium license.')

        # Run
        TabularPreset.list_available_presets(out)

        # Assert
        assert out.getvalue().strip() == expected

    def test_save(self):
        """Test the ``TabularPreset.save`` method.

        Expect that the model's save method is called with the expected args.

        Input:
        - path

        Side Effects:
        - The model's ``save`` method is called with the same argument.
        """
        # Setup
        model = Mock()
        preset = Mock()
        preset._model = model

        # Run
        TabularPreset.save(preset, 'test-path')

        # Assert
        model.save.assert_called_once_with('test-path')

    def test_load(self):
        """Test the ``TabularPreset.load`` method.

        Expect that the model's load method is called with the expected args.

        Input:
        - path

        Side Effects:
        - The default model's ``load`` method is called with the same argument.
        """
        # Setup
        default_model = Mock()
        TabularPreset._default_model = default_model

        # Run
        TabularPreset.load('test-file.pkl')

        # Assert
        default_model.load.called_once_with('test-file.pkl')

    def test___repr__(self):
        """Test the ``TabularPreset.__repr__`` method.

        Output:
        - Expect a string 'TabularPreset(name=<name>)'
        """
        # Setup
        instance = TabularPreset('FAST_ML')

        # Run
        res = repr(instance)

        # Assert
        assert res == 'TabularPreset(name=FAST_ML)'
