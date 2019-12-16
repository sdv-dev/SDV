from copulas.multivariate import GaussianMultivariate

from sdv.models.base import SDVModel
# from sdv.models.utils import flatten_dict, unflatten_dict


class Gaussian(SDVModel):

    covariance = None
    distribs = None
    distribution = None
    means = None
    model = None

    def __init__(self, distribution):
        self.distribution = distribution

    def fit(self, table_data):
        self.model = GaussianMultivariate(distribution=self.distribution)
        self.model.fit(table_data)

        self.covariance = self.model.covariance
        self.distribs = self.model.distribs
        self.means = self.model.means

    def sample(self, num_samples):
        return self.model.sample(num_samples)

    def get_parameters(self):
        # return flatten_dict(self.model.get_parameters())
        raise NotImplementedError

    def set_parameters(self, parameters):
        # self.model.set_parameters(unflatten_dict(parameters))
        raise NotImplementedError

    def to_dict(self):
        return self.model.to_dict()

    @classmethod
    def from_dict(cls, copula_dict):
        model = GaussianMultivariate.from_dict(copula_dict)

        instance = cls(model.distribution)
        instance.covariance = model.covariance
        instance.distribs = model.distribs
        instance.means = model.means
        instance.model = model

        return instance
