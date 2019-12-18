from copulas.multivariate import GaussianMultivariate as CopulasGaussianMultivariate

from sdv.models.base import SDVModel
from sdv.models.utils import flatten_dict, unflatten_dict, unflatten_gaussian_copula


class GaussianMultivariate(SDVModel):

    distribution = None
    model = None

    def __init__(self, distribution):
        self.distribution = distribution

    def fit(self, table_data):
        self.model = CopulasGaussianMultivariate(distribution=self.distribution)
        self.model.fit(table_data)

    def sample(self, num_samples):
        return self.model.sample(num_samples)

    def get_parameters(self):
        return flatten_dict(self.model.to_dict())

    def set_parameters(self, parameters):
        unflatten_parameters = unflatten_dict(parameters)

        if unflatten_parameters.get('fitted') is None:
            unflatten_parameters['fitted'] = True

        if unflatten_parameters.get('distribution') is None:
            unflatten_parameters['distribution'] = self.distribution

        unflatten_parameters = unflatten_gaussian_copula(unflatten_parameters)

        for param in unflatten_parameters['distribs'].values():
            if param.get('type') is None:
                param['type'] = self.distribution

            if param.get('fitted') is None:
                param['fitted'] = True

        self.model = CopulasGaussianMultivariate.from_dict(unflatten_parameters)
