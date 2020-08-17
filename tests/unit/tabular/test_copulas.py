from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from copulas.multivariate.gaussian import GaussianMultivariate

from sdv.tabular.copulas import GaussianCopula


class TestGaussianCopula:

    def test__get_distribution_none(self):
        """Test the ``_get_distribution`` passing None.

        If ``None`` is passed, the output should be the a PARAMETRIC
        Univariate.

        Input:
        - None

        Expected Output:
        - Instance of ``Univariate`` with:
            - univariate.PARAMETRIC == NON_PARAMETRIC
            - univariate.BOUNDED == UNBOUNDED
        """
        # Run
        out = GaussianCopula._get_distribution(None)
        # Asserts
        assert out is GaussianCopula._DISTRIBUTIONS['parametric']

    def test__get_distribution_str(self):
        """Test the ``_get_distribution method passing a known str.

        If the name of a known distribution is passed, return the corresponding
        instance.

        Input:
         - Name of a distribution (for example, ``gamma``)

        Expected Output:
        - Instance of the indicated distribution. (for example,
          ``copulas.univariate.gamma.GammaUnivariate``)
         """
        # Run
        out = GaussianCopula._get_distribution('bounded')
        # Assert
        assert out is GaussianCopula._DISTRIBUTIONS['bounded']

    def test__get_distribution_unknown(self):
        """Test the ``_get_distribution`` passing an unknown str.

        If an unknown str is passed, return it as it is.

        Input:
        - Unknown string

        Expected Output:
        - Input string
        """
        # Run
        Result = GaussianCopula._get_distribution('unknown')
        # Asserts
        assert Result == 'unknown'

    def test__get_distribution_dict(self):
        """ Test the ``_get_distribution`` passing a dict.

        If a dict is passed, a new dict with the same keys must be returned,
        where each value has been resolved to the corresponding instance.

        Input:
        - dict passing one value and one key of each type:
            - None
            - a str with the name of a known distribution
            - a str with an unknown name
        - For example: {'a': None, 'b': 'gamma', 'c': 'unknown'}

        Expected Output:
        - dict containing the corresponding instances.
        """
        # Run
        dictionary = {1: 'bounded', 2: None}
        out = GaussianCopula._get_distribution(dictionary)

        # Assert
        expected = {1: GaussianCopula._DISTRIBUTIONS['bounded'],
                    2: GaussianCopula._DISTRIBUTIONS['parametric']
                    }
        assert out == expected

    def _update_metadata_existing_model_kargs(self):
        """Test ``_update_metadata`` if metadata already has model_kwargs.

        If ``self._metadata`` already has ``model_kwargs`` in it, this
        method should do nothing.

        Setup:
        - self._metadata.get_model_kwargs that returns a kwargs dict

        Expected Output
        - None

        Side Effects
        - ``self._metadata.set_model_kwargs`` is not called.
        """

    def _update_metadata_no_model_kwargs(self):
        """Test ``_update_metadata`` if metadata has no model_kwargs.

        If ``self._metadata`` has no ``model_kwargs`` in it, this
        method should prepare the ``model_kwargs`` dict and call
        ``self._metadata.set_model_kwargs`` with it.

        Setup:
        - self._metadata.get_model_kwargs that returns None.

        Expected Output
        - None

        Side Effects
        - ``self._metadata.set_model_kwargs`` is called with the
          expected dict.
        """

    @patch('sdv.tabular.copulas.copulas.multivariate.GaussianMultivariate',
           spec_set=GaussianMultivariate)
    def test__fit(self, gm_mock):
        """Test the ``GaussianCopula._fit`` method.

        The ``_fit`` method is expected to:
        - Create a GaussianMultivriate object with the indicated distribution value
        (``self._distribution``)
        - Store the GaussianMultivariate instance in the `self._model` attribute.
        - Fit the GaussianMultivariate instance with the given table data, unmodified.
        - Call the `_update_metadata` method.

        Input:
        - pandas.DataFrame
        Expected Output:
        - None
        Side Effects:
        - GaussianMultivariate is called with ``self._distribution`` as input
        - GaussianMultivariate output is stored as ``self._model``
        - ``self._model.fit`` is called with the input dataframe
        - ``self._update_metadata`` is called without arguments
        """

        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        gaussian_copula._distribution = 'a_distribution'

        # Run
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

    def test__sample(self):
        """Test the ``GaussianCopula._sample`` method.

        The GaussianCopula._sample method is expected to:
        - call ``self._model.sample`` method passing the given num_rows.
        - Return the output from the ``self._model.sample call``.

        Input:
        - Integer
        Expected Output:
        - ``self._model.sample.return_value``
        Side Effects:
        - ``self._model.sample`` is called with the given integer as input
        """

    def test_get_parameters(self):
        """Test the ``get_parameters`` method when model is parametric.

        If all the distributions are parametric, ``get_parameters``
        should return a flattened version of the parameters returned
        by the ``GaussianMultivariate`` instance.

        Setup:
        - ``self._model`` will be set to a REAL GaussianMultivarite instance
          with the following properties:
          - Uses the following distributions:
              - GaussianUnivariate
              - Univariate(parametric=PARAMETRIC)
          - Is fitted with a two column dataframe where the column
            of the ``GaussianUnivariate`` is constant (to force
            ``scale==0``) and the other one is not constant (to
            force ``scale!=0``). The dataframe can contain only
            three rows:

                gm = GaussianMultivariate(distribution={
                    'a': GaussianMultivariate,
                    'b': Univariate(parametric=PARAMETRIC)
                })
                pd.DataFrame({
                    'a': [1, 1, 1],
                    'b': [1, 2, 3],
                })

        Output:
        - Flattened parameter dictionary with the right values in it:
            - triangular covariance matrix
            - ``np.log`` applied to the ``EPSILON`` value for the
              univariate that had ``scale==0``.
            - ``np.log`` applied to the other ``scale`` parameter.
        """

    def test_get_parameters_non_parametric(self):
        """Test the ``get_parameters`` method when model is parametric.

        If there is at least one distributions in the model that is not
        parametric, a NonParametricError should be raised.

        Setup:
        - ``self._model`` is set to a ``GaussianMultivariate`` that
          uses ``GaussianKDE`` as its ``distribution``.

        Side Effects:
        - A NonParametricError is raised.
        """

    def test__rebuild_gaussian_copula(self):
        """Test the ``GaussianCopula._rebuild_gaussian_copula`` method.

        The ``test__rebuild_gaussian_copula`` method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.

        Input:
        - Triangular covariance matrix

        Expected Output:
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

    def set_parameters(self):
        """Test the ``set_parameters`` method with positive num_rows.

        The ``GaussianCopula.set_parameters`` method is expected to:
        - Transform a flattened dict into its original form with
          the unflatten_dict function.
        - pass the unflattended dict to the ``self._rebuild_gaussian_copula``
          method.
        - Store the number of rows in the `self._num_rows` attribute.
        - Create a GaussianMultivariate instance from the params dict
          and store it in the 'self._model' attribute.

        Input:
        - flat parameters dict

        Output:
        - None

        Side Effects:
        - Call ``_rebuild_gaussian_copula`` with the unflatted dict.
        - ``self._num_rows`` gets the given value.
        - ``GaussianMultivariate`` is called
        - ``GaussianMultivariate`` return value is stored as `self._model`
        """

    def set_parameters_negative_max_rows(self):
        """Test the ``set_parameters`` method with negative num_rows.

        If the max rows value is negative, it is expected to be set
        to zero.

        The ``GaussianCopula.set_parameters`` method is expected to:
        - Transform a flattened dict into its original form with
          the unflatten_dict function.
        - pass the unflattended dict to the ``self._rebuild_gaussian_copula``
          method.
        - Store ``0`` in the `self._num_rows` attribute.
        - Create a GaussianMultivariate instance from the params dict
          and store it in the 'self._model' attribute.

        Input:
        - flat parameters dict

        Output:
        - None

        Side Effects:
        - Call ``_rebuild_gaussian_copula`` with the unflatted dict.
        - ``self._num_rows`` is set to ``0``.
        - ``GaussianMultivariate`` is called
        - ``GaussianMultivariate`` return value is stored as `self._model`
        """
