from unittest.mock import Mock, patch

import copulas
import numpy as np
import pandas as pd
import pytest
from copulas.multivariate.gaussian import GaussianMultivariate, Univariate
from copulas.univariate import GaussianKDE, GaussianUnivariate

from sdv.tabular.base import NonParametricError
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
        assert out.PARAMETRIC is GaussianCopula._DISTRIBUTIONS['parametric'].PARAMETRIC
        assert out.BOUNDED is GaussianCopula._DISTRIBUTIONS['parametric'].BOUNDED

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
        out = GaussianCopula._get_distribution('gamma')

        # Assert
        assert out is GaussianCopula._DISTRIBUTIONS['gamma']

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
        dictionary = {1: None, 2: 'gamma', 3: 'unknown'}
        out = GaussianCopula._get_distribution(dictionary)

        # Assert
        expected = {1: GaussianCopula._DISTRIBUTIONS['parametric'],
                    2: GaussianCopula._DISTRIBUTIONS['gamma'],
                    3: 'unknown'
                    }
        for key in out.keys():
            assert out[key] is expected[key]

    def test___init__(self):
        """Test ``__init__`` with empty input values.

        All the parameters of the class are inicialized with the default values.

        Expected Output
        - None

        Side Effects
        - ``Table.from_dict`` is not called.
        - ``table_metadata.get_model_kwargs`` is not called.
        - ``_distribution`` is set to a instance of the indicated distribution.
        """

    def test__init__metadata(self):
        """Test ``__init__`` with not empty input values for `table_metadata`.

        Load the values of `table_metadata`

        Input
        - dict

        Expected Output
        - None

        Side Effects
        - ``Table.from_dict`` is called.
        - ``table_metadata.get_model_kwargs`` is called.
        - ``_distribution`` is set to a instance of the indicated distribution.
        """

    def test__update_metadata_existing_model_kargs(self):
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
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)

        # Run
        out = GaussianCopula._update_metadata(gaussian_copula)

        # Asserts
        assert out is None
        assert not gaussian_copula._metadata.set_model_kwargs.called

    def test__update_metadata_no_model_kwargs(self):
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
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        gaussian_copula._metadata.get_model_kwargs.return_value = False
        distribution = Mock()
        gaussian_copula._categorical_transformer = distribution
        model_parameters = {
            'univariates': [{
                'scale': 1.0,
                'loc': 5,
                'type': 'GaussianUnivariate'
            }
            ],
            'columns': ['foo'],
            'distribution': 'GaussianUnivariate',
            'covariance': [[0.4, 0.17], [0.17, 0.07]]
        }
        gaussian_copula._model.to_dict.return_value = model_parameters

        # Run
        out = GaussianCopula._update_metadata(gaussian_copula)

        # Asserts
        # call = {
        #    'distribution': {'foo': 'GaussianUnivariate'},
        #    'categorical_transformer': distribution,
        # }
        assert out is None
        # assert gaussian_copula._metadata.\
        #    set_model_kwargs.assert_called_once_with('GaussianCopula', call)

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
        # Setup
        n_rows = 2
        gaussian_copula = Mock(spec_set=GaussianCopula)

        # Run
        GaussianCopula._sample(gaussian_copula, n_rows)

        # Asserts
        gaussian_copula._model.sample.assert_called_once_with(n_rows)

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
            of the ``GaussianMultivariate`` is constant (to force
            ``scale==0``) and the other one is not constant (to
            force ``scale!=0``). The dataframe can contain only
            three rows:

                gm = GaussianMultivariate(distribution={
                    'a': GaussianUnivariate,
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
        # Setup
        gm = GaussianCopula(distribution={
            'a': GaussianUnivariate(),
            'b': Univariate(parametric=copulas.univariate.ParametricType.PARAMETRIC)
        })
        data = pd.DataFrame({
            'a': [1, 1, 1],
            'b': [1, 2, 3],
        })
        gm.fit(data)

        # Run
        # out = gm.get_parameters()

        # Asserts
        # assert np.isclose(out['covariance__0__0'],1.5)
        # assert np.isclose(out['univariates__b__scale'],-0.2, atol = 0.003)

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
        # Setup
        gm = GaussianCopula(distribution=GaussianKDE())
        data = pd.DataFrame([1, 1, 1])
        gm.fit(data)
        # Run, Assert
        with pytest.raises(NonParametricError):
            gm.get_parameters()  # Out[9]: copulas.multivariate.gaussian.GaussianMultivariate

    def test__rebuild_covariance_matrix_positive_definite(self):
        """Test the ``_rebuild_covariance_matrix``
        method for a positive definide covariance matrix.

        The _rebuild_covariance_matrix method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.
        - Call ``make_positive_definite`` if input matrix is not positive definite,

        Input
        - numpy array, Symmetric positive definite matrix triangular format

        output
        - numpy array, Square matrix positive definite

        Side Effects:
        - ``make_positive_definite`` is not called.
        """
        # Run
        covariance = [[1], [0, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[1., 0.], [0., 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_covariance_matrix_not_positive_definite(self):
        """Test the ``_rebuild_covariance_matrix``
        method for a not positive definide covariance matrix.

        The _rebuild_covariance_matrix method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.
        - Call ``make_positive_definite`` if input matrix is not positive definite,

        Input
        - numpy array, Symmetric no positive definite matrix triangular format

        output
        - numpy array, Square matrix positive definite

        Side Effects:
        - ``make_positive_definite`` is called.
        """
        # Run
        covariance = [[1], [-1, 1]]
        result = GaussianCopula._rebuild_covariance_matrix(Mock(), covariance)

        # Asserts
        expected = np.array([[1, -1.0], [-1.0, 1.0]])
        np.testing.assert_almost_equal(result, expected)

    def test__rebuild_gaussian_copula(self):
        """Test the ``GaussianCopula._rebuild_gaussian_copula`` method.

        The ``test__rebuild_gaussian_copula`` method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.

        Input:
        - numpy array, Triangular covariance matrix

        Expected Output:
        - numpy array, Square covariance matrix
        """
        # Setup
        gaussian_copula = Mock(autospec=GaussianCopula)
        gaussian_copula._rebuild_covariance_matrix.return_value = [[0.4, 0.17], [0.17, 0.07]]
        gaussian_copula._distribution = {'foo': 'GaussianUnivariate'}

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
        result = GaussianCopula._rebuild_gaussian_copula(gaussian_copula, model_parameters)

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

    def test_set_parameters(self):
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

    def test_set_parameters_negative_max_rows(self):
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
