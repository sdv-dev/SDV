"""Wrappers around copulas models."""

import inspect
import logging
import warnings
from copy import deepcopy

import copulas
import copulas.univariate
import numpy as np
import pandas as pd
import scipy
from copulas import multivariate
from rdt.transformers import OneHotEncoder

from sdv.errors import NonParametricError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import (
    flatten_dict,
    unflatten_dict,
    validate_numerical_distributions,
    warn_missing_numerical_distributions,
)

LOGGER = logging.getLogger(__name__)


class GaussianCopulaSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        metadata (sdv.metadata.Metadata):
            Single table metadata representing the data that this synthesizer will be used for.
            * sdv.metadata.SingleTableMetadata can be used but will be deprecated.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used. The distributions can be passed as either
            a ``copulas.univariate`` instance or as one of the following values:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.

        default_distribution (str):
            Copulas univariate distribution to use by default. Valid options are:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a Truncated Gaussian distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
             Defaults to ``beta``.
    """

    _DISTRIBUTIONS = {
        'norm': copulas.univariate.GaussianUnivariate,
        'beta': copulas.univariate.BetaUnivariate,
        'truncnorm': copulas.univariate.TruncatedGaussian,
        'gamma': copulas.univariate.GammaUnivariate,
        'uniform': copulas.univariate.UniformUnivariate,
        'gaussian_kde': copulas.univariate.GaussianKDE,
    }

    @classmethod
    def get_distribution_class(cls, distribution):
        """Return the corresponding distribution class from ``copulas.univariate``.

        Args:
            distribution (str):
                A string representing a copulas univariate distribution.

        Returns:
            copulas.univariate:
                A copulas univariate class that corresponds to the distribution.
        """
        if not isinstance(distribution, str) or distribution not in cls._DISTRIBUTIONS:
            error_message = f"Invalid distribution specification '{distribution}'."
            raise ValueError(error_message)

        return cls._DISTRIBUTIONS[distribution]

    def __init__(
        self,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=['en_US'],
        numerical_distributions=None,
        default_distribution=None,
    ):
        super().__init__(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )
        validate_numerical_distributions(
            numerical_distributions,
            self._get_table_metadata().columns,
        )

        self.default_distribution = default_distribution or 'beta'
        self._default_distribution = self.get_distribution_class(self.default_distribution)

        self._set_numerical_distributions(numerical_distributions)
        self._num_rows = None

    def _set_numerical_distributions(self, numerical_distributions):
        self.numerical_distributions = numerical_distributions or {}
        self._numerical_distributions = {
            field: self.get_distribution_class(distribution)
            for field, distribution in self.numerical_distributions.items()
        }

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        warn_missing_numerical_distributions(self.numerical_distributions, processed_data.columns)
        self._num_rows = self._learn_num_rows(processed_data)
        numerical_distributions = self._get_numerical_distributions(processed_data)
        self._model = self._initialize_model(numerical_distributions)
        self._fit_model(processed_data)

    def _learn_num_rows(self, processed_data):
        return len(processed_data)

    def _get_numerical_distributions(self, processed_data):
        numerical_distributions = deepcopy(self._numerical_distributions)
        for column in processed_data.columns:
            if column not in numerical_distributions:
                numerical_distributions[column] = self._numerical_distributions.get(
                    column, self._default_distribution
                )

        return numerical_distributions

    def _initialize_model(self, numerical_distributions):
        return multivariate.GaussianMultivariate(distribution=numerical_distributions)

    def _fit_model(self, processed_data):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='scipy')
            self._model.fit(processed_data)

    def _warn_quality_and_performance(self, column_name_to_transformer):
        """Raise warning if the quality/performance may be impacted.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        for column, transformer in column_name_to_transformer.items():
            if isinstance(transformer, OneHotEncoder):
                warnings.warn(
                    f"Using a OneHotEncoder transformer for column '{column}' "
                    'may slow down the preprocessing and modeling times.'
                )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._model.sample(num_rows, conditions=conditions)

    def _get_valid_columns_from_metadata(self, columns):
        valid_columns = []
        table_metadata = self._get_table_metadata()
        for column in columns:
            for valid_column in table_metadata.columns:
                if column.startswith(valid_column):
                    valid_columns.append(column)
                    break

        return valid_columns

    def get_learned_distributions(self):
        """Get the marginal distributions used by the ``GaussianCopula``.

        Return a dictionary mapping the column names with the distribution name and the learned
        parameters for those.

        Returns:
            dict:
                Dictionary containing the distributions used or detected for each column and the
                learned parameters for those.
        """
        if not self._fitted:
            raise ValueError(
                "Distributions have not been learned yet. Please fit your model first using 'fit'."
            )

        if not hasattr(self._model, 'to_dict') or not self._model.to_dict():
            return {}

        parameters = self._model.to_dict()
        columns = parameters['columns']
        univariates = deepcopy(parameters['univariates'])
        learned_distributions = {}
        valid_columns = self._get_valid_columns_from_metadata(columns)
        distribution_names = {v.__name__: k for k, v in self._DISTRIBUTIONS.items()}
        for column, learned_params in zip(columns, univariates):
            if column in valid_columns:
                distribution_name = learned_params.pop('type').split('.')[-1]
                distribution = distribution_names[distribution_name]
                learned_distributions[column] = {
                    'distribution': distribution,
                    'learned_parameters': learned_params,
                }

        return learned_distributions

    def _get_parameters(self):
        """Get copula model parameters.

        Compute model ``correlation`` and ``distribution.std``
        before it returns the flatten dict.

        Returns:
            dict:
                Copula parameters.

        Raises:
            NonParametricError:
                If a non-parametric distribution has been used.
        """
        for univariate in self._model.univariates:
            univariate_type = type(univariate)
            if univariate_type is copulas.univariate.Univariate:
                univariate = univariate._instance

            if univariate.PARAMETRIC == copulas.univariate.ParametricType.NON_PARAMETRIC:
                raise NonParametricError('This GaussianCopula uses non parametric distributions')

        params = self._model.to_dict()

        correlation = []
        for index, row in enumerate(params['correlation'][1:]):
            correlation.append(row[: index + 1])

        params['correlation'] = correlation
        params['univariates'] = dict(zip(params.pop('columns'), params['univariates']))
        params['num_rows'] = self._num_rows

        return flatten_dict(params)

    @staticmethod
    def _get_nearest_correlation_matrix(matrix):
        """Find the nearest correlation matrix.

        If the given matrix is not Positive Semi-definite, which means
        that any of its eigenvalues is negative, find the nearest PSD matrix
        by setting the negative eigenvalues to 0 and rebuilding the matrix
        from the same eigenvectors and the modified eigenvalues.

        After this, the matrix will be PSD but may not have 1s in the diagonal,
        so the diagonal is replaced by 1s and then the PSD condition of the
        matrix is validated again, repeating the process until the built matrix
        contains 1s in all the diagonal and is PSD.

        After 10 iterations, the last step is skipped and the current PSD matrix
        is returned even if it does not have all 1s in the diagonal.

        Insipired by: https://stackoverflow.com/a/63131250
        """
        eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
        negative = eigenvalues < 0
        identity = np.identity(len(matrix))

        iterations = 0
        while np.any(negative):
            eigenvalues[negative] = 0
            matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            if iterations >= 10:
                break

            matrix = matrix - matrix * identity + identity

            max_value = np.abs(np.abs(matrix).max())
            if max_value > 1:
                matrix /= max_value

            eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
            negative = eigenvalues < 0
            iterations += 1

        return matrix

    @classmethod
    def _rebuild_correlation_matrix(cls, triangular_correlation):
        """Rebuild a valid correlation matrix from its lower half triangle.

        The input of this function is a list of lists of floats of size 1, 2, 3...n-1:

           [[c_{2,1}], [c_{3,1}, c_{3,2}], ..., [c_{n,1},...,c_{n,n-1}]]

        Corresponding to the values from the lower half of the original correlation matrix,
        **excluding** the diagonal.

        The output is the complete correlation matrix reconstructed using the given values
        and scaled to the :math:`[-1, 1]` range if necessary.

        Args:
            triangle_correlation (list[list[float]]):
                A list that contains lists of floats of size 1, 2, 3... up to ``n-1``,
                where ``n`` is the size of the target correlation matrix.

        Returns:
            numpy.ndarray:
                rebuilt correlation matrix.
        """
        zero = [0.0]
        size = len(triangular_correlation) + 1
        left = np.zeros((size, size))
        right = np.zeros((size, size))
        for idx, values in enumerate(triangular_correlation):
            values = values + zero * (size - idx - 1)
            left[idx + 1, :] = values
            right[:, idx + 1] = values

        correlation = left + right
        max_value = np.abs(correlation).max()
        if max_value > 1:
            correlation /= max_value

        correlation += np.identity(size)

        return cls._get_nearest_correlation_matrix(correlation).tolist()

    def _rebuild_gaussian_copula(self, model_parameters, default_params=None):
        """Rebuild the model params to recreate a Gaussian Multivariate instance.

        Args:
            model_parameters (dict):
                Sampled and reestructured model parameters.
            default_parameters (dict):
                Fall back parameters if sampled params are invalid.

        Returns:
            dict:
                Model parameters ready to recreate the model.
        """
        if default_params is None:
            default_params = {}

        columns = []
        univariates = []
        for column, univariate in model_parameters['univariates'].items():
            columns.append(column)
            if column in self._numerical_distributions:
                univariate_type = self._numerical_distributions[column]
            else:
                univariate_type = self.get_distribution_class(self.default_distribution)

            univariate['type'] = univariate_type
            model = univariate_type.MODEL_CLASS
            if hasattr(model, '_argcheck'):
                to_check = {
                    parameter: univariate[parameter]
                    for parameter in inspect.signature(model._argcheck).parameters.keys()
                    if parameter in univariate
                }
                if not model._argcheck(**to_check):
                    if column in default_params.get('univariates', []):
                        LOGGER.info(
                            f"Invalid parameters sampled for column '{column}', "
                            'using default parameters.'
                        )
                        univariate = default_params['univariates'][column]
                        univariate['type'] = univariate_type
                    else:
                        LOGGER.debug(f"Column '{column}' has invalid parameters.")
            else:
                LOGGER.debug(f"Univariate for col '{column}' does not have _argcheck method.")

            if 'scale' in univariate:
                univariate['scale'] = max(0, univariate['scale'])

            univariates.append(univariate)

        model_parameters['univariates'] = univariates
        model_parameters['columns'] = columns

        correlation = model_parameters.get('correlation')
        if correlation:
            model_parameters['correlation'] = self._rebuild_correlation_matrix(correlation)
        else:
            model_parameters['correlation'] = [[1.0]]

        return model_parameters

    def _get_likelihood(self, table_rows):
        return self._model.probability_density(table_rows)

    def _set_parameters(self, parameters, default_params=None):
        """Set copula model parameters.

        Args:
            params [dict]:
                Copula flatten parameters.
            default_params [list]:
                Flattened list of parameters to fall back to if `params` are invalid.

        """
        if default_params is not None:
            default_params = unflatten_dict(default_params)
        else:
            default_params = {}

        parameters = unflatten_dict(parameters)
        if 'num_rows' in parameters:
            num_rows = parameters.pop('num_rows')
            self._num_rows = 0 if pd.isna(num_rows) else max(0, int(round(num_rows)))

        if parameters:
            parameters = self._rebuild_gaussian_copula(parameters, default_params)
            self._model = multivariate.GaussianMultivariate.from_dict(parameters)
