from copulas.multivariate import GaussianMultivariate as CopulasGaussianMultivariate

from sdv.models.base import SDVModel
from sdv.models.utils import flatten_dict, unflatten_dict


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
        self.model = CopulasGaussianMultivariate.from_dict(unflatten_parameters)
