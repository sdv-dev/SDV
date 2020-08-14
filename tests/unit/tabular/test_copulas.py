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
        """Test the  GaussianCopula._rebuild_covariance_matrix
        method for a covariance matrix in triangular format.
        This method return Square matrix positive definide.

        The _rebuild_covariance_matrix method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.
        - Evaluate if the covariance matrix is positive definide by the
          method check_matrix_symmetric_positive_definite
        - If matrix is not positive definide, apply the method make_positive_definite.

        Input
        - Symmetric positive definite matrix triangular format

        output
        - Square matrix positive definite

        Side Effects:
        - make_positive_definite is not called.
        """
        # Run
        covariance = [[1], [0, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[1., 0.], [0., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_covariance_matrix_not_positive_definite(self):
        """Test the GaussianCopula._rebuild_covariance_matrix method for a covariance matrix in
           triangular format. This method return Square matrix positive definite.

        The _rebuild_covariance_matrix method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.
        - Evaluate if the covariance matrix is positive definide by the
          method check_matrix_symmetric_positive_definite
        - If matrix is not positive definide, apply the method make_positive_definite.

        Input
        - No Symmetric positive definite matrix triangular format

        Output
        - Square matrix positive definite

        Side Effects:
        - Make_positive_definite is called.
        """
        # Run
        covariance = [[-1], [0, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[0., 0.], [0., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_gaussian_copula(self):
        """Test the GaussianCopula._rebuild_gaussian_copula method.

        The test__rebuild_gaussian_copula method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.

        Input:
        - Triangular covariance matrix
        Output:
        - Square covariance matrix
        """
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

    def test__sample(self):
        """Test the GaussianCopula._sample method.

        The GaussianCopula._sample method is expected to:
        - call self._model.sample method passing the given num_rows.
        - Return the output from the self._model.sample call.

        Input:
        - Integer
        Output:
        - self._model.sample.return_value
        Side Effects:
        - self._model.sample is called with the given integer as input
        """

    def test__get_distribution_none(self):
        """Test the GaussianCopula._get_distribution method passing None.

        The GaussianCopula._get_distribution method is expected to:
        - Return a GaussianCopula._DISTRIBUTIONS['parametric'].

        Input:
        - None

        Output:
        - GaussianCopula._DISTRIBUTIONS['parametric']
        """

    def test__get_distribution_str(self):
        """Test the GaussianCopula._get_distribution method passing str.

        The GaussianCopula._get_distribution method is expected to:
        - Return a dictionary of copulas.univariate instances.

        Input:
        - String

        Output:
        - copulas.univariate instance
        """

    def test__get_distribution_dict_distribution(self):
        """ Test the GaussianCopula._get_distribution method for a dictionari input.

        The GaussianCopula._get_distribution method is expected to:
        - Return a dictionary of copulas.univariate instances.

        Input:
        - Dictionary

        Output:
        - Dictionary
        """

    def test__get_distribution_unknown(self):
        """Test the GaussianCopula._get_distribution method for an unknown input.

        This method return the input value.

        The `GaussianCopula._get_distribution` method is expected to:
        - Return a dictionary of `copulas.univariate` instances.

        Input:
        - Input

        Output:
        - Input without changes
        """

    def _update_metadata(self):
        """Test the GaussianCopula._update_metadata method with model data previously introduced.

        The _update_metadata method is expected to:
        - Apply the method 'self._metadata.get_model_kwargs' to get model_kwargs.
        - If model_kwargs is not None, do not produce any effect

        Input
        - Self

        Output
        - None

        Side effect
        -self._metadata.set_model_kwargs is never call.
        """

    def _update_metadata_no_kwargs(self):
        """Test the GaussianCopula._update_metadata method with model
        without data previously introduced.

        The _update_metadata method is expected to:
        - Apply the method self._metadata.get_model_kwargs to get model_kwargs.
        - If model_kwargs is None, set values.

        Input
        - Self

        Output
        - None

        Side effect
        -self._metadata.set_model_kwargs is  call.
        """

    def test_get_parameters(self):
        """Test the GaussianCopula.get_parameters method.

        The GaussianCopula.get_parameters method is expected to:
        - Get parameters from the GaussianCopula instance and store it in a flatten dict.
        - Return the dictionary.
        Input:
        - Self
        Output:
        - Flattened parameter dictionary
        """

    def test_get_parameters_non_parametric(self):
        """Test the GaussianCopula.get_parameters method for non parametric values.

        The GaussianCopula.test_get_parameters_non_parametric method is expected to:
        - Get the variable  'copulas.univariate.ParametricType.NON_PARAMETRIC'.
        - If copulas.univariate.ParametricType.NON_PARAMETRIC
        is True, raise error message.

        Input:
        - Self

        Output:
        - Raise Error
        """

    def set_parameters(self):
        """Test the GaussianCopula.set_parameters method.

        The GaussianCopula.set_parameters method is expected to:
        - Transform a flattened dict into its original form with `unflatten_dict` method.
        - Rebuild the model params to recreate a Gaussian Multivariate instance with
          self._rebuild_gaussian_copula method.
        - Store the number of rows in the `self._num_rows` attribute.
        - Store the GaussianMultivariate instance in the self._model attribute.

        Input:
        - Copula flatten parameters

        Output:
        - None

        Side Effects:
        - GaussianMultivariate is called with `self._rebuild_gaussian_copula`.
        - `Parameters['num_rows']` value is stored as `self._num_rows`
        - GaussianMultivariate output is stored as `self._model`
        """

    def set_parameters_negative_max_rows(self):
        """Test the GaussianCopula.set_parameters method with a negative number of rows.

        The GaussianCopula.set_parameters method is expected to:
        - Transform a flattened dict into its original form with `unflatten_dict` method.
        - Rebuild the model params to recreate a Gaussian Multivariate instance with
          self._rebuild_gaussian_copula method.
        - Store the number of rows in the `self._num_rows` attribute, the expected value is 0.
        - Store the GaussianMultivariate instance in the self._model attribute.

        Input:
        - Copula flatten parameters

        Output:
        - None

        Side Effects:
        - GaussianMultivariate is called with `self._rebuild_gaussian_copula`.
        - `self._num_rows` value is 0.
        - GaussianMultivariate output is stored as self._model
        """
