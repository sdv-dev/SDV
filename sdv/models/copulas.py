import numpy as np

from copulas.multivariate import GaussianMultivariate as CopulasGaussianMultivariate
from sdv.models.base import SDVModel
from sdv.models.utils import (
    check_matrix_symmetric_positive_definite, flatten_dict, impute, make_positive_definite,
    square_matrix, unflatten_dict)


class GaussianMultivariate(SDVModel):

    distribution = None
    model = None

    def __init__(self, distribution):
        self.distribution = distribution

    def fit(self, table_data):
        table_data = impute(table_data)
        self.model = CopulasGaussianMultivariate(distribution=self.distribution)
        self.model.fit(table_data)

    def sample(self, num_samples):
        return self.model.sample(num_samples)

    def get_parameters(self):
        values = list()
        triangle = np.tril(self.model.covariance)

        for index, row in enumerate(triangle.tolist()):
            values.append(row[:index + 1])

        self.model.covariance = np.array(values)
        for distribution in self.model.distribs.values():
            if distribution.std is not None:
                distribution.std = np.log(distribution.std)

        return flatten_dict(self.model.to_dict())

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

        distribution_kwargs = {
            'fitted': True,
            'type': model_parameters['distribution']
        }

        distribs = model_parameters['distribs']
        for distribution in distribs.values():
            distribution.update(distribution_kwargs)
            distribution['std'] = np.exp(distribution['std'])

        covariance = model_parameters['covariance']
        model_parameters['covariance'] = self._prepare_sampled_covariance(covariance)

        return model_parameters

    def set_parameters(self, parameters):
        parameters = unflatten_dict(parameters)

        if parameters.get('fitted') is None:
            parameters['fitted'] = True

        if parameters.get('distribution') is None:
            parameters['distribution'] = self.distribution

        parameters = self._unflatten_gaussian_copula(parameters)

        for param in parameters['distribs'].values():
            if param.get('type') is None:
                param['type'] = self.distribution

            if param.get('fitted') is None:
                param['fitted'] = True

        self.model = CopulasGaussianMultivariate.from_dict(parameters)
