import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import scipy
from copulas.univariate import BetaUnivariate, GammaUnivariate, UniformUnivariate

from sdv.errors import SynthesizerInputError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


class TestGaussianCopulaSynthesizer:

    def test_get_distribution_class_str(self):
        """Test that when a ``str`` is passed, the class from the ``DISTRIBUTIONS`` is returned."""
        # Setup
        distribution = 'beta'

        # Run
        result = GaussianCopulaSynthesizer.get_distribution_class(distribution)

        # Assert
        assert result == BetaUnivariate

    def test_get_distribution_class_not_in_distributions(self):
        """Test that ``ValueError`` is raised when the given distribution is not supported."""
        # Setup
        distribution = 'student'

        # Run and Assert
        with pytest.raises(ValueError, match="Invalid distribution specification 'student'."):
            GaussianCopulaSynthesizer.get_distribution_class(distribution)

    def test___init__(self):
        """Test creating an instance of ``GaussianCopulaSynthesizer``."""
        # Setup
        metadata = SingleTableMetadata()
        enforce_min_max_values = True
        enforce_rounding = True
        numerical_distributions = None
        default_distribution = None

        # Run
        instance = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution,
        )

        # Assert
        assert instance.enforce_min_max_values is True
        assert instance.enforce_rounding is True
        assert instance.numerical_distributions == {}
        assert instance.default_distribution == 'beta'
        assert instance._default_distribution == BetaUnivariate
        assert instance._numerical_distributions == {}
        assert instance._num_rows is None

    def test___init__custom(self):
        """Test creating an instance of ``GaussianCopulaSynthesizer`` with custom parameters."""
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('field', sdtype='numerical')
        enforce_min_max_values = False
        enforce_rounding = False
        numerical_distributions = {'field': 'gamma'}
        default_distribution = 'uniform'

        # Run
        instance = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution,
        )

        # Assert
        assert instance.enforce_min_max_values is False
        assert instance.enforce_rounding is False
        assert instance.numerical_distributions == {'field': 'gamma'}
        assert instance.default_distribution == 'uniform'
        assert instance._default_distribution == UniformUnivariate
        assert instance._numerical_distributions == {'field': GammaUnivariate}

    def test___init__incorrect_numerical_distributions(self):
        """Test it crashes when ``numerical_distributions`` receives a non-dictionary."""
        # Setup
        metadata = SingleTableMetadata()
        numerical_distributions = 'invalid'

        # Run
        err_msg = 'numerical_distributions can only be None or a dict instance.'
        with pytest.raises(TypeError, match=err_msg):
            GaussianCopulaSynthesizer(metadata, numerical_distributions=numerical_distributions)

    def test___init__incorrect_column_numerical_distributions(self):
        """Test it crashes when ``numerical_distributions`` includes invalid columns."""
        # Setup
        metadata = SingleTableMetadata()
        numerical_distributions = {'totally_fake_column_name': 'beta'}

        # Run
        err_msg = re.escape(
            'Invalid column names found in the numerical_distributions dictionary '
            "{'totally_fake_column_name'}. The column names you provide must be present "
            'in the metadata.'
        )
        with pytest.raises(SynthesizerInputError, match=err_msg):
            GaussianCopulaSynthesizer(metadata, numerical_distributions=numerical_distributions)

    def test_get_parameters(self):
        """Test that inherited method ``get_parameters`` returns the specified init parameters."""
        # Setup
        metadata = SingleTableMetadata()
        instance = GaussianCopulaSynthesizer(metadata)

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'locales': None,
            'numerical_distributions': {},
            'default_distribution': 'beta'
        }

    @patch('sdv.single_table.copulas.LOGGER')
    def test__fit_logging(self, mock_logger):
        """Test a message is logged.

        A message should be logged if the columns passed in ``numerical_distributions``
        were renamed/dropped during preprocessing.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')
        numerical_distributions = {'col': 'gamma'}
        instance = GaussianCopulaSynthesizer(
            metadata, numerical_distributions=numerical_distributions)
        processed_data = pd.DataFrame({'updated_col': [1, 2, 3]})

        # Run
        instance._fit(processed_data)

        # Assert
        mock_logger.info.assert_called_once_with(
            "Requested distribution 'gamma' cannot be applied to column 'col' "
            'because it no longer exists after preprocessing.'
        )

    @patch('sdv.single_table.copulas.warnings')
    @patch('sdv.single_table.copulas.multivariate')
    def test__fit(self, mock_multivariate, mock_warnings):
        """Test the ``_fit``.

        Test that when fitting, numerical distributions are being generated for any missing column
        or new one that be generated from the ``preprocess`` step. The model should be created with
        the ``numerical_distributions``.
        """
        # Setup
        metadata = SingleTableMetadata()
        metadata.add_column('name', sdtype='numerical')
        metadata.add_column('user.id', sdtype='numerical')
        numerical_distributions = {'name': 'uniform', 'user.id': 'gamma'}

        processed_data = pd.DataFrame({
            'name': np.arange(10),
            'user.id': np.arange(10),
            'account_balance': np.arange(10)
        })
        instance = GaussianCopulaSynthesizer(
            metadata,
            numerical_distributions=numerical_distributions
        )

        # Run
        instance._fit(processed_data)

        # Assert
        expected_numerical_distributions = {
            'name': UniformUnivariate,
            'name': UniformUnivariate,
            'user.id': GammaUnivariate,
            'account_balance': BetaUnivariate,
        }

        mock_multivariate.GaussianMultivariate.assert_called_once_with(
            distribution=expected_numerical_distributions
        )
        instance._model.fit.assert_called_once_with(processed_data)
        mock_warnings.filterwarnings.assert_called_once_with('ignore', module='scipy')
        mock_warnings.catch_warnings.assert_called_once()
        instance._num_rows == 10

    def test__get_nearest_correlation_matrix_valid(self):
        """Test ``_get_nearest_correlation_matrix`` with a psd input.

        If the matrix is positive semi-definite, do nothing.

        Input:
        - matrix which is positive semi-definite.

        Expected Output:
        - the input, unmodified.
        """
        # Run
        correlation_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        output = GaussianCopulaSynthesizer._get_nearest_correlation_matrix(correlation_matrix)

        # Assert
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        assert expected == output.tolist()
        assert output is correlation_matrix

    def test__get_nearest_correlation_matrix_invalid(self):
        """Test ``_get_nearest_correlation_matrix`` with a non psd input.

        If the matrix is not positive semi-definite, modify it to make it PSD.

        Input:
        - matrix which is not positive semi-definite.

        Expected Output:
        - modified matrix which is positive semi-definite.
        """
        # Run
        not_psd_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ])
        output = GaussianCopulaSynthesizer._get_nearest_correlation_matrix(not_psd_matrix)

        # Assert
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        assert expected == output.tolist()

        not_psd_eigenvalues = scipy.linalg.eigh(not_psd_matrix)[0]
        output_eigenvalues = scipy.linalg.eigh(output)[0]
        assert (not_psd_eigenvalues < 0).any()
        assert (output_eigenvalues >= 0).all()

    def test__rebuild_correlation_matrix_valid(self):
        """Test ``_rebuild_correlation_matrix`` with a valid correlation input.

        If the input contains values between -1 and 1, the method is expected
        to simply rebuild the square matrix with the same values.

        Input:
        - list of lists with values between -1 and 1

        Expected Output:
        - numpy array with the square correlation matrix
        """
        # Run
        triangular_correlation = [
            [0.1],
            [0.2, 0.3]
        ]
        correlation = GaussianCopulaSynthesizer._rebuild_correlation_matrix(triangular_correlation)

        # Assert
        expected = [
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ]
        assert expected == correlation

    def test__rebuild_correlation_matrix_outside(self):
        """Test ``_rebuild_correlation_matrix`` with an invalid correlation input.

        If the input contains values outside -1 and 1, the method is expected
        to scale them down to the valid range.

        Input:
        - list of lists with values outside of -1 and 1

        Expected Output:
        - numpy array with the square correlation matrix
        """
        # Run
        triangular_correlation = [
            [1.0],
            [2.0, 1.0]
        ]
        correlation = GaussianCopulaSynthesizer._rebuild_correlation_matrix(triangular_correlation)

        # Assert
        expected = [
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 0.5],
            [1.0, 0.5, 1.0]
        ]
        assert expected == correlation

    def test__rebuild_gaussian_copula(self):
        """Test the ``GaussianCopulaSynthesizer._rebuild_gaussian_copula`` method.

        The ``test__rebuild_gaussian_copula`` method is expected to:
        - Rebuild a square correlation matrix out of a triangular one.

        Input:
        - numpy array, Triangular correlation matrix

        Expected Output:
        - numpy array, Square correlation matrix
        """
        # Setup
        metadata = SingleTableMetadata()
        gaussian_copula = GaussianCopulaSynthesizer(metadata)
        model_parameters = {
            'univariates': {
                'foo': {
                    'scale': 0.0,
                    'loc': 0.0
                },
                'bar': {
                    'scale': 1.0,
                    'loc': 1.0
                },
                'baz': {
                    'scale': 2.0,
                    'loc': 2.0
                },
            },
            'correlation': [[0.1], [0.2, 0.3]],
            'distribution': 'beta',
        }

        # Run
        result = GaussianCopulaSynthesizer._rebuild_gaussian_copula(
            gaussian_copula,
            model_parameters
        )

        # Asserts
        expected = {
            'univariates': [
                {
                    'scale': 0.0,
                    'loc': 0.0,
                    'type': BetaUnivariate
                },
                {
                    'scale': 1.0,
                    'loc': 1.0,
                    'type': BetaUnivariate
                },
                {
                    'scale': 2.0,
                    'loc': 2.0,
                    'type': BetaUnivariate
                },
            ],
            'correlation': [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0]
            ],
            'distribution': 'beta',
            'columns': ['foo', 'bar', 'baz'],
        }
        assert result == expected

    @patch('sdv.single_table.copulas.multivariate')
    @patch('sdv.single_table.copulas.unflatten_dict')
    def test___set_parameters(self, mock_unflatten_dict, mock_multivariate):
        """Test that parameters are properly set and that number of rows is set properly."""
        # Setup
        parameters = {
            'correlation': [
                [0.0],
                [0.0, 0.0]
            ],
            'num_rows': 4.59,
            'univariates': {
                'amount': {
                    'loc': 85.62233142690933,
                    'scale': 0.0
                },
                'cancelled': {
                    'loc': 0.48772424778255064,
                    'scale': 0.0
                },
                'timestamp': {
                    'loc': 1.5475359249730097e+18,
                    'scale': 0.0
                }
            }
        }
        instance = Mock()
        mock_unflatten_dict.return_value = parameters

        # Run
        GaussianCopulaSynthesizer._set_parameters(instance, parameters)

        # Assert
        mock_unflatten_dict.assert_called_once_with(parameters)
        expected_parameters = {
            'correlation': [
                [0.0],
                [0.0, 0.0]
            ],
            'univariates': {
                'amount': {
                    'loc': 85.62233142690933,
                    'scale': 0.0
                },
                'cancelled': {
                    'loc': 0.48772424778255064,
                    'scale': 0.0
                },
                'timestamp': {
                    'loc': 1.5475359249730097e+18,
                    'scale': 0.0
                }
            }
        }

        instance._rebuild_gaussian_copula.assert_called_once_with(expected_parameters)
        model = mock_multivariate.GaussianMultivariate.from_dict.return_value
        assert instance._model == model
        assert instance._num_rows == 5
        mock_multivariate.GaussianMultivariate.from_dict.assert_called_once_with(
            instance._rebuild_gaussian_copula.return_value
        )

    def test__get_valid_columns_from_metadata(self):
        """Test that it returns a list with columns that are from the metadata."""
        # Seutp
        instance = Mock()
        instance.metadata.columns = {
            'a_value': object(),
            'n_value': object(),
            'b_value': object()
        }
        columns = ['a', 'a_value.is_null', '__b_value', '__a_value__b_value', 'n_value']

        # Run
        result = GaussianCopulaSynthesizer._get_valid_columns_from_metadata(instance, columns)

        # Assert
        assert result == ['a_value.is_null', 'n_value']

    def test_get_learned_distributions(self):
        """Test that ``get_learned_distributions`` returns a dict.

        Test that it returns a dictionary with the name of the columns and the learned
        distribution and it's parameters.
        """
        # Setup
        data = pd.DataFrame({
            'zero': [0, 0, 0],
            'one': [1, 1, 1]
        })
        stm = SingleTableMetadata()
        stm.detect_from_dataframe(data)
        gcs = GaussianCopulaSynthesizer(stm, numerical_distributions={'one': 'uniform'})
        gcs.fit(data)

        # Run
        result = gcs.get_learned_distributions()

        # Assert
        assert result == {
            'zero': {
                'distribution': 'beta',
                'learned_parameters': {
                    'a': 1.0,
                    'b': 1.0,
                    'loc': 0.0,
                    'scale': 0.0
                }
            },
            'one': {
                'distribution': 'uniform',
                'learned_parameters': {
                    'loc': 1.0,
                    'scale': 0.0
                }
            }
        }

    def test_get_learned_distributions_raises_an_error(self):
        """Test that ``get_learned_distributions`` returns a dict.

        Test that it returns a dictionary with the name of the columns and the learned
        distribution and it's parameters.
        """
        # Setup
        data = pd.DataFrame({
            'zero': [0, 0, 0],
            'one': [1, 1, 1]
        })
        stm = SingleTableMetadata()
        stm.detect_from_dataframe(data)
        gcs = GaussianCopulaSynthesizer(stm)

        # Run and Assert
        error_msg = re.escape(
            "Distributions have not been learned yet. Please fit your model first using 'fit'."
        )
        with pytest.raises(ValueError, match=error_msg):
            gcs.get_learned_distributions()

    def test__get_likelihood(self):
        """Test that ``_get_likelihood`` returns the ``model.probability_density`` of the input."""
        # Setup
        table_rows = pd.Series([1, 2, 3])
        instance = Mock()

        # Run
        result = GaussianCopulaSynthesizer._get_likelihood(instance, table_rows)

        # Assert
        assert result == instance._model.probability_density.return_value
        instance._model.probability_density.assert_called_once_with(table_rows)
