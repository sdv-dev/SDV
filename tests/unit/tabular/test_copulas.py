from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import scipy
from copulas.multivariate.gaussian import GaussianMultivariate
from copulas.univariate import GaussianKDE, GaussianUnivariate

from sdv.constraints import CustomConstraint
from sdv.tabular.base import NonParametricError
from sdv.tabular.copulas import GaussianCopula


class TestGaussianCopula:

    def test___init__no_metadata(self):
        """Test ``__init__`` without passing a table_metadata.

        In this case, the parent __init__ will be called and the metadata
        will be created based on the given arguments

        Input:
            - field_names
            - field_types
            - field_transformers
            - anonymize_fields
            - primary_key
            - constraints
            - field_distributions
            - default_distribution
            - categorical_transformer

        Side Effects
            - attributes are set to the right values
            - metadata is created with the right values
        """
        gc = GaussianCopula(
            field_names=['a_field'],
            field_types={
                'a_field': {
                    'type': 'categorical',
                }
            },
            field_transformers={'a_field': 'categorical'},
            anonymize_fields={'a_field': 'name'},
            primary_key=['a_field'],
            constraints=[CustomConstraint()],
            field_distributions={'a_field': 'gaussian'},
            default_distribution='bounded',
            categorical_transformer='categorical_fuzzy'
        )

        assert gc._field_distributions == {'a_field': GaussianUnivariate}
        assert gc._default_distribution == GaussianCopula._DISTRIBUTIONS['bounded']
        assert gc._categorical_transformer == 'categorical_fuzzy'
        assert gc._DTYPE_TRANSFORMERS == {'O': 'categorical_fuzzy'}

        metadata = gc._metadata.to_dict()
        assert metadata == {
            'fields': None,
            'constraints': [
                {
                    'constraint': 'sdv.constraints.tabular.CustomConstraint'
                }
            ],
            'model_kwargs': {
                'GaussianCopula': {
                    'field_distributions': {
                        'a_field': 'gaussian'
                    },
                    'default_distribution': 'bounded',
                    'categorical_transformer': 'categorical_fuzzy'
                }
            },
            'name': None,
            'primary_key': ['a_field'],
            'sequence_index': None,
            'entity_columns': [],
            'context_columns': []
        }

    def test___init__metadata_dict(self):
        """Test ``__init__`` passing a table_metadata dict.

        In this case, metadata will be loaded from the dict and passed
        to the parent.

        Input:
            - table_metadata
            - field_distributions
            - default_distribution
            - categorical_transformer

        Side Effects
            - attributes are set to the right values
            - metadata is created with the right values
        """
        table_metadata = {
            'name': 'test',
            'fields': {
                'a_field': {
                    'type': 'categorical'
                },
            },
            'model_kwargs': {
                'GaussianCopula': {
                    'field_distributions': {
                        'a_field': 'gaussian',
                    },
                    'categorical_transformer': 'categorical_fuzzy',
                }
            }
        }
        gc = GaussianCopula(
            default_distribution='bounded',
            table_metadata=table_metadata,
        )

        assert gc._field_distributions == {'a_field': GaussianUnivariate}
        assert gc._default_distribution == GaussianCopula._DISTRIBUTIONS['bounded']
        assert gc._categorical_transformer == 'categorical_fuzzy'
        assert gc._DTYPE_TRANSFORMERS == {'O': 'categorical_fuzzy'}

        metadata = gc._metadata.to_dict()
        assert metadata == {
            'name': 'test',
            'fields': {
                'a_field': {
                    'type': 'categorical'
                },
            },
            'constraints': [],
            'model_kwargs': {
                'GaussianCopula': {
                    'field_distributions': {
                        'a_field': 'gaussian'
                    },
                    'default_distribution': 'bounded',
                    'categorical_transformer': 'categorical_fuzzy'
                }
            },
            'primary_key': None,
            'sequence_index': None,
            'entity_columns': [],
            'context_columns': [],
        }

    def test__update_metadata(self):
        """Test ``_update_metadata`` if metadata has no model_kwargs.

        If ``self._metadata`` has no ``model_kwargs`` in it, this
        method should prepare the ``model_kwargs`` dict and call
        ``self._metadata.set_model_kwargs`` with it.

        Setup:
        - self._metadata.get_model_kwargs that returns None.
        - self.get_distributions that returns a distribution dict.

        Expected Output
        - None

        Side Effects
        - ``self._metadata.set_model_kwargs`` is called with the
          expected dict.
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        gaussian_copula._metadata.get_model_kwargs.return_value = dict()
        gaussian_copula._categorical_transformer = 'a_categorical_transformer_value'
        gaussian_copula._default_distribution = 'a_distribution'
        gaussian_copula.get_distributions.return_value = {
            'foo': 'copulas.univariate.gaussian.GaussianUnivariate'
        }

        # Run
        out = GaussianCopula._update_metadata(gaussian_copula)

        # Asserts
        assert out is None
        expected_kwargs = {
            'field_distributions': {'foo': 'copulas.univariate.gaussian.GaussianUnivariate'},
            'default_distribution': 'a_distribution',
            'categorical_transformer': 'a_categorical_transformer_value',
        }
        gaussian_copula._metadata.set_model_kwargs.assert_called_once_with(
            'GaussianCopula', expected_kwargs)

    @patch('sdv.tabular.copulas.copulas.multivariate.GaussianMultivariate',
           spec_set=GaussianMultivariate)
    def test__fit(self, gm_mock):
        """Test the ``GaussianCopula._fit`` method.

        The ``_fit`` method is expected to:
        - Call the _get_distribution method to build the distributions dict.
        - Set the output from _get_distribution method as self._distribution.
        - Create a GaussianMultivriate object with the self._distribution value.
        - Store the GaussianMultivariate instance in the self._model attribute.
        - Fit the GaussianMultivariate instance with the given table data, unmodified.
        - Call the _update_metadata method.

        Setup:
            - mock _get_distribution to return a distribution dict

        Input:
            - pandas.DataFrame

        Expected Output:
            - None

        Side Effects:
            - self._distribution is set to the output from _get_distribution
            - GaussianMultivariate is called with self._distribution as input
            - GaussianMultivariate output is stored as self._model
            - self._model.fit is called with the input dataframe
            - self._update_metadata is called without arguments
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        gaussian_copula._field_distributions = {'a': 'a_distribution'}

        # Run
        data = pd.DataFrame({
            'a': [1, 2, 3]
        })
        out = GaussianCopula._fit(gaussian_copula, data)

        # asserts
        assert out is None
        assert gaussian_copula._field_distributions == {'a': 'a_distribution'}
        gm_mock.assert_called_once_with(distribution={'a': 'a_distribution'})

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
        - call ``self._model.sample`` method passing the given num_rows and conditions.
        - Return the output from the ``self._model.sample call``.

        Input:
        - Integer + dict
        Expected Output:
        - ``self._model.sample.return_value``
        Side Effects:
        - ``self._model.sample`` is called with the given integer as input
        """
        # Setup
        gaussian_copula = Mock(spec_set=GaussianCopula)
        expected = pd.DataFrame([1, 2, 3])
        gaussian_copula._model.sample.return_value = expected

        # Run
        out = GaussianCopula._sample(gaussian_copula, 2, {'a': 'a'})

        # Asserts
        gaussian_copula._model.sample.assert_called_once_with(2, conditions={'a': 'a'})
        assert expected.equals(out)

    def test_get_parameters(self):
        """Test the ``get_parameters`` method when model is parametric.

        If all the distributions are parametric, ``_get_parameters``
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

    def test__get_parameters_non_parametric(self):
        """Test the ``_get_parameters`` method when model is parametric.

        If there is at least one distributions in the model that is not
        parametric, a NonParametricError should be raised.

        Setup:
        - ``self._model`` is set to a ``GaussianMultivariate`` that
          uses ``GaussianKDE`` as its ``distribution``.

        Side Effects:
        - A NonParametricError is raised.
        """
        # Setup
        gm = GaussianMultivariate(distribution=GaussianKDE())
        data = pd.DataFrame([1, 1, 1])
        gm.fit(data)
        gc = Mock()
        gc._model = gm

        # Run, Assert
        with pytest.raises(NonParametricError):
            GaussianCopula._get_parameters(gc)

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
        output = GaussianCopula._get_nearest_correlation_matrix(correlation_matrix)

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
        output = GaussianCopula._get_nearest_correlation_matrix(not_psd_matrix)

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
        correlation = GaussianCopula._rebuild_correlation_matrix(triangular_covariance)

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
        correlation = GaussianCopula._rebuild_correlation_matrix(triangular_covariance)

        # Assert
        expected = [
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 0.5],
            [1.0, 0.5, 1.0]
        ]
        assert expected == correlation

    def test__rebuild_gaussian_copula(self):
        """Test the ``GaussianCopula._rebuild_gaussian_copula`` method.

        The ``test__rebuild_gaussian_copula`` method is expected to:
        - Rebuild a square covariance matrix out of a triangular one.

        Input:
        - numpy array, Triangular correlation matrix

        Expected Output:
        - numpy array, Square correlation matrix
        """
        # Setup
        gaussian_copula = GaussianCopula()
        gaussian_copula._field_distributions = {
            'foo': 'GaussianUnivariate',
            'bar': 'GaussianUnivariate',
            'baz': 'GaussianUnivariate',
        }

        # Run
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
            'distribution': 'GaussianUnivariate',
        }
        result = GaussianCopula._rebuild_gaussian_copula(gaussian_copula, model_parameters)

        # Asserts
        expected = {
            'univariates': [
                {
                    'scale': 0.0,
                    'loc': 0.0,
                    'type': 'GaussianUnivariate'
                },
                {
                    'scale': 1.0,
                    'loc': 1.0,
                    'type': 'GaussianUnivariate'
                },
                {
                    'scale': 2.0,
                    'loc': 2.0,
                    'type': 'GaussianUnivariate'
                },
            ],
            'covariance': [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0]
            ],
            'distribution': 'GaussianUnivariate',
            'columns': ['foo', 'bar', 'baz'],
        }
        assert result == expected

    def test__set_parameters(self):
        """Test the ``_set_parameters`` method.

        The ``GaussianCopula._set_parameters`` method is expected to:
        - Transform a flattened dict into its original form with
          the unflatten_dict function.
        - pass the unflattended dict to the ``self._rebuild_gaussian_copula``
          method.
        - Create a GaussianMultivariate instance from the params dict
          and store it in the 'self._model' attribute.

        Input:
        - flat parameters dict

        Output:
        - None

        Side Effects:
        - Call ``_rebuild_gaussian_copula`` with the unflatted dict.
        - ``GaussianMultivariate`` is called
        - ``GaussianMultivariate`` return value is stored as `self._model`
        """
        # Setup
        gaussian_copula = Mock(autospec=GaussianCopula)
        returned = {
            'univariates': [
                {
                    'scale': 0.0,
                    'loc': 0.0,
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate'
                },
                {
                    'scale': 1.0,
                    'loc': 1.0,
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate'
                }
            ],
            'columns': ['foo', 'bar'],
            'num_rows': 3,
            'covariance': [[0.4, 0.17], [0.17, 0.07]]
        }
        gaussian_copula._rebuild_gaussian_copula.return_value = returned

        # Run
        flatten_parameters = {
            'univariates__foo__scale': 0.0,
            'univariates__foo__loc': 0.0,
            'univariates__bar__scale': 1.0,
            'univariates__bar__loc': 1.0,
            'covariance__0__0': 0.1,
            'covariance__1__0': 0.4,
            'covariance__1__1': 0.1,
            'num_rows': 3
        }
        GaussianCopula._set_parameters(gaussian_copula, flatten_parameters)

        # Asserts
        expected = {
            'covariance': [[0.1], [0.4, 0.1]],
            'num_rows': 3,
            'univariates': {
                'foo': {
                    'scale': 0.0,
                    'loc': 0.0,
                },
                'bar': {
                    'scale': 1.0,
                    'loc': 1.0,
                }
            }
        }
        gaussian_copula._rebuild_gaussian_copula.assert_called_once_with(expected)
        assert isinstance(gaussian_copula._model, GaussianMultivariate)

    def test__validate_distribution_none(self):
        """Test the ``_validate_distribution`` method if None is passed.

        If None is passed, it should just return None.

        Input:
        - None

        Output:
        - None
        """
        out = GaussianCopula._validate_distribution(None)

        assert out is None

    def test__validate_distribution_not_str(self):
        """Test the ``_validate_distribution`` method if something that is not an str is passed.

        If the input is not an str, it should be returned without modification.

        Input:
        - a dummy object

        Output:
        - the same dummy object
        """
        dummy = object()
        out = GaussianCopula._validate_distribution(dummy)

        assert out is dummy

    def test__validate_distribution_distribution_name(self):
        """Test the ``_validate_distribution`` method passing a valid distribution name.

        If the input is one keys from the ``_DISTRIBUTIONS`` dict, the value should be returned.

        Input:
        - A key from the ``_DISTRIBUTIONS`` dict.

        Output:
        - The corresponding value.
        """
        out = GaussianCopula._validate_distribution('gaussian_kde')

        assert out is GaussianKDE

    def test__validate_distribution_fqn(self):
        """Test the ``_validate_distribution`` method passing a FQN distribution name.

        If the input is an importable FQN of a Python object, return the input.

        Input:
        - A univariate distribution FQN.

        Output:
        - The corresponding class.
        """
        out = GaussianCopula._validate_distribution('copulas.univariate.GaussianUnivariate')

        assert out == 'copulas.univariate.GaussianUnivariate'
