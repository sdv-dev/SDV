import numpy as np
import copulas

from sdv.tabular.base import BaseTabularModel
from sdv.tabular.utils import (
    check_matrix_symmetric_positive_definite, flatten_dict, make_positive_definite,
    square_matrix, unflatten_dict)


class GaussianCopula(BaseTabularModel):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        distribution (copulas.univariate.Univariate or str):
            Copulas univariate distribution to use.
    """

    DISTRIBUTION = copulas.univariate.GaussianUnivariate
    _distribution = None
    _model = None

    HYPERPARAMETERS = {
        'distribution': {
            'type': 'str or copulas.univariate.Univariate',
            'default': 'copulas.univariate.Univariate',
            'choices': [
                'copulas.univariate.Univariate',
                'copulas.univariate.GaussianUnivariate',
                'copulas.univariate.GammaUnivariate',
                'copulas.univariate.BetaUnivariate',
                'copulas.univariate.StudentTUnivariate',
                'copulas.univariate.GaussianKDE',
                'copulas.univariate.TruncatedGaussian',
            ]
        }
    }

    def __init__(self, distribution=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribution = distribution or self.DISTRIBUTION

    def _update_metadata(self):
        parameters = self._model.to_dict()
        univariates = parameters['univariates']
        columns = parameters['columns']

        fields = self._metadata.get_fields()
        for field_name, univariate in zip(columns, univariates):
            field_meta = fields[field_name]
            field_meta['distribution'] = univariate['type']

    def _fit(self, data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be fitted.
        """
        self._model = copulas.multivariate.GaussianMultivariate(distribution=self._distribution)
        self._model.fit(data)
        # self._update_metadata()

    def _sample(self, size):
        """Sample ``size`` rows from the model.

        Args:
            size (int):
                Amount of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._model.sample(size)

    def get_parameters(self):
        """Get copula model parameters.

        Compute model ``covariance`` and ``distribution.std``
        before it returns the flatten dict.

        Returns:
            dict:
                Copula flatten parameters.
        """
        values = list()
        triangle = np.tril(self._model.covariance)

        for index, row in enumerate(triangle.tolist()):
            values.append(row[:index + 1])

        self._model.covariance = np.array(values)
        params = self._model.to_dict()
        univariates = dict()
        for name, univariate in zip(params.pop('columns'), params['univariates']):
            univariates[name] = univariate
            if 'scale' in univariate:
                scale = univariate['scale']
                if scale == 0:
                    scale = copulas.EPSILON

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
            if 'scale' in univariate:
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
        parameters.setdefault('distribution', self.distribution)

        parameters = self._unflatten_gaussian_copula(parameters)

        self._model = copulas.multivariate.GaussianMultivariate.from_dict(parameters)
