from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy
from copulas.univariate import BetaUnivariate, GammaUnivariate, UniformUnivariate

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


class TestGaussianCopulaSynthesizer:

    def test__validate_distribution_str(self):
        """Test that when a ``str`` is passed, the class from the ``DISTRIBUTIONS`` is returned."""
        # Setup
        distribution = 'beta'

        # Run
        result = GaussianCopulaSynthesizer._validate_distribution(distribution)

        # Assert
        assert result == BetaUnivariate

    def test__validate_distribution_not_in_distributions(self):
        """Test that ``ValueError`` is raised when the given distribution is not supported."""
        # Setup
        distribution = 'student'

        # Run and Assert
        with pytest.raises(ValueError, match="Invalid distribution specification 'student'."):
            GaussianCopulaSynthesizer._validate_distribution(distribution)

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

    def test___init__custom(self):
        """Test creating an instance of ``GaussianCopulaSynthesizer`` with custom parameters."""
        # Setup
        metadata = SingleTableMetadata()
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

    def test_get_params(self):
        """Test that inherited method ``get_params`` returns all the specific init parameters."""
        # Setup
        metadata = SingleTableMetadata()
        instance = GaussianCopulaSynthesizer(metadata)

        # Run
        result = instance.get_parameters()

        # Assert
        assert result == {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'numerical_distributions': {},
            'default_distribution': 'beta'
        }

    @patch('sdv.single_table.copulas.warnings')
    @patch('sdv.single_table.copulas.copulas.multivariate')
    def test__fit(self, mock_multivariate, mock_warnings):
        """Test the ``_fit``.

        Test that when fitting, numerical distributions are being generated for any missing column
        or new one that be generated from the ``preprocess`` step. The model should be created with
        the ``numerical_distributions``.
        """
        # Setup
        metadata = SingleTableMetadata()
        numerical_distributions = {'name': 'uniform', 'user.id': 'gamma'}

        processed_data = pd.DataFrame({
            'name.value': np.arange(10),
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
            'name.value': UniformUnivariate,
            'user.id': GammaUnivariate,
            'account_balance': BetaUnivariate,
        }

        mock_multivariate.GaussianMultivariate.assert_called_once_with(
            distribution=expected_numerical_distributions
        )
        instance._model.fit.assert_called_once_with(processed_data)
        mock_warnings.filterwarnings.assert_called_once_with('ignore', module='scipy')
        mock_warnings.catch_warnings.assert_called_once()

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
        triangular_covariance = [
            [0.1],
            [0.2, 0.3]
        ]
        correlation = GaussianCopulaSynthesizer._rebuild_correlation_matrix(triangular_covariance)

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
        triangular_covariance = [
            [1.0],
            [2.0, 1.0]
        ]
        correlation = GaussianCopulaSynthesizer._rebuild_correlation_matrix(triangular_covariance)

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
        - Rebuild a square covariance matrix out of a triangular one.

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
            'covariance': [[0.1], [0.2, 0.3]],
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
                    'type': 'beta'
                },
                {
                    'scale': 1.0,
                    'loc': 1.0,
                    'type': 'beta'
                },
                {
                    'scale': 2.0,
                    'loc': 2.0,
                    'type': 'beta'
                },
            ],
            'covariance': [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0]
            ],
            'distribution': 'beta',
            'columns': ['foo', 'bar', 'baz'],
        }
        assert result == expected
