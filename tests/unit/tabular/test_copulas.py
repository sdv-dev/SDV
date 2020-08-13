from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from copulas.multivariate.gaussian import GaussianMultivariate

from sdv.tabular.copulas import GaussianCopula


class TestGaussianCopula:

    @patch('sdv.tabular.copulas.copulas.multivariate.GaussianMultivariate',
           spec_set=GaussianMultivariate)
    def test__fit(self, gm_mock):
        """Test the GaussianCopula._fit method.

        The GaussianCopula._fit method is expected to:
        - Create a GaussianMultivariate object with the indicated distribution value
        (``self._distribution``)
        - Store the GaussianMultivariate instance in the `self._model` attribute.
        - Fit the GaussianMultivariate instance with the given table data, unmodified.
        - Call the `_update_metadata` method.

        Input:
        - pandas.DataFrame
        Expected Output:
        - None
        Side Effects:
        - GaussianMultivariate is called with self._distribution as input
        - GaussianMultivariate output is stored as `self._model`
        - self._model.fit is called with the input dataframe
        - self._update_metadata is called without arguments
        """

        # setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        gaussian_copula._distribution = 'a_distribution'

        # run
        data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        out = GaussianCopula._fit(gaussian_copula, data)

        # asserts
        assert out is None
        gm_mock.assert_called_once_with(distribution='a_distribution')

        assert gaussian_copula._model == gm_mock.return_value
        expected_data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        call_args = gaussian_copula._model.fit.call_args_list
        passed_table_data = call_args[0][0][0]

        pd.testing.assert_frame_equal(expected_data, passed_table_data)
        gaussian_copula._update_metadata.assert_called_once_with()

    def test__rebuild_covariance_matrix_positive_definite(self):
        # Run
        covariance = [[1], [0, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[1., 0.], [0., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_covariance_matrix_not_positive_definite(self):
        # Run
        covariance = [[-1], [0, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[0., 0.], [0., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_gaussian_copula(self):
        # Setup
        sdvmodel = Mock(autospec=GaussianCopula)
        sdvmodel._rebuild_covariance_matrix.return_value = [[0.4, 0.17], [0.17, 0.07]]
        sdvmodel._distribution = {'foo': 'GaussianUnivariate'}

        # Run
        model_parameters = {
            'univariates': {
                'foo': {
                    'scale': 0.0,
                    'loc': 5
                },
            },
            'covariance': [[0.1], [0.4, 0.1]],
            'distribution': 'GaussianUnivariate',
        }
        result = GaussianCopula._rebuild_gaussian_copula(sdvmodel, model_parameters)

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
            'covariance': [[0.4, 0.17], [0.17, 0.07]]
        }
        assert result == expected
