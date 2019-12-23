from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from sdv.models.copulas import GaussianCopula


class TestGaussianCopula(TestCase):

    def test__prepare_sampled_covariance(self):
        """Test prepare_sampler_covariante"""
        # Run
        covariance = [[0, 1], [1]]
        result = GaussianCopula._prepare_sampled_covariance(Mock(), covariance)

        # Asserts
        expected = np.array([[1., 1.], [1., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__unflatten_gaussian_copula(self):
        """Test unflatte gaussian copula"""
        # Setup
        sdvmodel = Mock(autospec=GaussianCopula)
        sdvmodel._prepare_sampled_covariance.return_value = [[0.4, 0.2], [0.2, 0.0]]

        # Run
        model_parameters = {
            'distribs': {
                'foo': {'std': 0.5}
            },
            'covariance': [[0.4, 0.1], [0.1]],
            'distribution': 'GaussianUnivariate'
        }
        result = GaussianCopula._unflatten_gaussian_copula(sdvmodel, model_parameters)

        # Asserts
        expected = {
            'distribs': {
                'foo': {
                    'fitted': True,
                    'std': 1.6487212707001282,
                    'type': 'GaussianUnivariate'
                }
            },
            'distribution': 'GaussianUnivariate',
            'covariance': [[0.4, 0.2], [0.2, 0.0]]
        }
        assert result == expected
