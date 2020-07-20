"""Tests for the sdv.models.copulas module."""
from unittest.mock import Mock

import numpy as np

from sdv.models.copulas import GaussianCopula


def test__prepare_sampled_covariance():
    """Test prepare_sampler_covariante."""
    # Run
    covariance = [[0, 1], [1]]
    result = GaussianCopula._prepare_sampled_covariance(Mock(), covariance)

    # Asserts
    expected = np.array([[1., 1.], [1., 1.0]])
    np.testing.assert_almost_equal(result, expected)


def test__unflatten_gaussian_copula():
    """Test unflatte gaussian copula."""
    # Setup
    sdvmodel = Mock(autospec=GaussianCopula)
    sdvmodel._prepare_sampled_covariance.return_value = [[0.4, 0.2], [0.2, 0.0]]

    # Run
    model_parameters = {
        'univariates': {
            'foo': {
                'scale': 0.0,
                'loc': 5
            },
        },
        'covariance': [[0.4, 0.1], [0.1]],
        'distribution': 'GaussianUnivariate'
    }
    result = GaussianCopula._unflatten_gaussian_copula(sdvmodel, model_parameters)

    # Asserts
    expected = {
        'univariates': [
            {
                'scale': 1.0,
                'loc': 5,
                'type': 'GaussianUnivariate'
            }
        ],
        'columns': ['foo'],
        'distribution': 'GaussianUnivariate',
        'covariance': [[0.4, 0.2], [0.2, 0.0]]
    }
    assert result == expected
