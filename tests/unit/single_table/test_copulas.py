from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
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
