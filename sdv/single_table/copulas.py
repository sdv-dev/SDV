"""Wrappers around copulas models."""
import warnings
from copy import deepcopy

import copulas
import copulas.multivariate
import copulas.univariate
import numpy as np
import scipy
from rdt.transformers import OneHotEncoder

from sdv.errors import NonParametricError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import flatten_dict, unflatten_dict


class GaussianCopulaSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
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

        default_distribution (copulas.univariate.Univariate or str):
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

    _model = None

    @classmethod
    def _validate_distribution(cls, distribution):
        if not isinstance(distribution, str) or distribution not in cls._DISTRIBUTIONS:
            error_message = f"Invalid distribution specification '{distribution}'."
            raise ValueError(error_message)

        return cls._DISTRIBUTIONS[distribution]

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True,
                 numerical_distributions=None, default_distribution=None):
        super().__init__(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        if numerical_distributions and not isinstance(numerical_distributions, dict):
            raise TypeError('numerical_distributions can only be None or a dict instance')

        self.default_distribution = default_distribution or 'beta'
        self.numerical_distributions = numerical_distributions or {}

        self._default_distribution = self._validate_distribution(self.default_distribution)
        self._numerical_distributions = {
            field: self._validate_distribution(distribution)
            for field, distribution in (numerical_distributions or {}).items()
        }
        self._num_rows = None

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        self._num_rows = len(processed_data)
        numerical_distributions = deepcopy(self._numerical_distributions)

        for column in processed_data.columns:
            if column not in numerical_distributions:
                column_name = column.replace('.value', '')
                numerical_distributions[column] = self._numerical_distributions.get(
                    column_name, self._default_distribution)

        self._model = copulas.multivariate.GaussianMultivariate(
            distribution=numerical_distributions
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='scipy')
            self._model.fit(processed_data)

    def _warn_for_update_transformers(self, column_name_to_transformer):
        """Raise warnings for update_transformers.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        for column, transformer in column_name_to_transformer.items():
            sdtype = self.metadata._columns[column]['sdtype']
            if sdtype == 'categorical' and isinstance(transformer, OneHotEncoder):
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

    def _get_parameters(self):
        """Get copula model parameters.

        Compute model ``covariance`` and ``distribution.std``
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

        covariance = []
        for index, row in enumerate(params['covariance'][1:]):
            covariance.append(row[:index + 1])

        params['covariance'] = covariance
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

    def _rebuild_gaussian_copula(self, model_parameters):
        """Rebuild the model params to recreate a Gaussian Multivariate instance.

        Args:
            model_parameters (dict):
                Sampled and reestructured model parameters.

        Returns:
            dict:
                Model parameters ready to recreate the model.
        """
        columns = []
        univariates = []
        for column, univariate in model_parameters['univariates'].items():
            columns.append(column)
            univariate['type'] = self._numerical_distributions.get(column, 'beta')
            if 'scale' in univariate:
                univariate['scale'] = max(0, univariate['scale'])

            univariates.append(univariate)

        model_parameters['univariates'] = univariates
        model_parameters['columns'] = columns

        covariance = model_parameters.get('covariance')
        if covariance:
            model_parameters['covariance'] = self._rebuild_correlation_matrix(covariance)
        else:
            model_parameters['covariance'] = [[1.0]]

        return model_parameters

    def _set_parameters(self, parameters):
        """Set copula model parameters.

        Args:
            dict:
                Copula flatten parameters.
        """
        parameters = unflatten_dict(parameters)
        parameters = self._rebuild_gaussian_copula(parameters)

        self._model = copulas.multivariate.GaussianMultivariate.from_dict(parameters)
