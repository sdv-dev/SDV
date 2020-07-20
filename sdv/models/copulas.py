"""Wrappers around copulas models."""

import numpy as np
from copulas import EPSILON
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate

from sdv.models.base import SDVModel
from sdv.tabular.utils import (
    check_matrix_symmetric_positive_definite, flatten_dict, impute, make_positive_definite,
    square_matrix, unflatten_dict)


class GaussianCopula(SDVModel):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        distribution (copulas.univariate.Univariate or str):
            Copulas univariate distribution to use.

    Example:
        The example below shows simple usage case where a ``GaussianMultivariate``
        is being created and its ``fit`` and ``sample`` methods are being called.

        >>> model = GaussianMultivariate()
        >>> model.fit(pd.DataFrame({'a_field': list(range(10))}))
        >>> model.sample(5)
            a_field
        0  4.796559
        1  7.395329
        2  7.400417
        3  2.794212
        4  1.925887
    """

    DISTRIBUTION = GaussianUnivariate
    distribution = None
    model = None

    def __init__(self, distribution=None):
        self.distribution = distribution or self.DISTRIBUTION

    def fit(self, table_data):
        """Fit the model to the table.

        Impute the table data before fit the model.

        Args:
            table_data (pandas.DataFrame):
                Data to be fitted.
        """
        table_data = impute(table_data)
        self.model = GaussianMultivariate(distribution=self.distribution)
        self.model.fit(table_data)

    def sample(self, num_samples):
        """Sample ``num_samples`` rows from the model.

        Args:
            num_samples (int):
                Amount of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled data with the number of rows specified in ``num_samples``.
        """
        return self.model.sample(num_samples)

    def get_parameters(self):
        """Get copula model parameters.

        Compute model ``covariance`` and ``distribution.std``
        before it returns the flatten dict.

        Returns:
            dict:
                Copula flatten parameters.
        """
        values = list()
        triangle = np.tril(self.model.covariance)

        for index, row in enumerate(triangle.tolist()):
            values.append(row[:index + 1])

        self.model.covariance = np.array(values)
        params = self.model.to_dict()
        univariates = dict()
        for name, univariate in zip(params.pop('columns'), params['univariates']):
            univariates[name] = univariate
            if 'scale' in univariate:
                scale = univariate['scale']
                if scale == 0:
                    scale = EPSILON

                univariate['scale'] = np.log(scale)

        params['univariates'] = univariates

        return flatten_dict(params)

    def _prepare_sampled_covariance(self, covariance):
        """Prepare a covariance matrix.

        Args:
            covariance (list):
                covariance after unflattening model parameters.

        Result:
            list[list]:
                symmetric Positive semi-definite matrix.
        """
        covariance = np.array(square_matrix(covariance))
        covariance = (covariance + covariance.T - (np.identity(covariance.shape[0]) * covariance))

        if not check_matrix_symmetric_positive_definite(covariance):
            covariance = make_positive_definite(covariance)

        return covariance.tolist()

    def _unflatten_gaussian_copula(self, model_parameters):
        """Prepare unflattened model params to recreate Gaussian Multivariate instance.

        The preparations consist basically in:

            - Transform sampled negative standard deviations from distributions into positive
              numbers

            - Ensure the covariance matrix is a valid symmetric positive-semidefinite matrix.

            - Add string parameters kept inside the class (as they can't be modelled),
              like ``distribution_type``.

        Args:
            model_parameters (dict):
                Sampled and reestructured model parameters.

        Returns:
            dict:
                Model parameters ready to recreate the model.
        """
        univariate_kwargs = {
            'type': model_parameters['distribution']
        }

        columns = list()
        univariates = list()
        for column, univariate in model_parameters['univariates'].items():
            columns.append(column)
            univariate.update(univariate_kwargs)
            univariate['scale'] = np.exp(univariate['scale'])
            univariates.append(univariate)

        model_parameters['univariates'] = univariates
        model_parameters['columns'] = columns

        covariance = model_parameters.get('covariance')
        model_parameters['covariance'] = self._prepare_sampled_covariance(covariance)

        return model_parameters

    def set_parameters(self, parameters):
        """Set copula model parameters.

        Add additional keys after unflatte the parameters
        in order to set expected parameters for the copula.

        Args:
            dict:
                Copula flatten parameters.
        """
        parameters = unflatten_dict(parameters)
        parameters.setdefault('fitted', True)
        parameters.setdefault('distribution', self.distribution)

        parameters = self._unflatten_gaussian_copula(parameters)

        self.model = GaussianMultivariate.from_dict(parameters)
