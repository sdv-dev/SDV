import copulas
import numpy as np
import rdt

from sdv.tabular.base import BaseTabularModel
from sdv.tabular.utils import (
    check_matrix_symmetric_positive_definite, flatten_dict, make_positive_definite, square_matrix,
    unflatten_dict)


class GaussianCopula(BaseTabularModel):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        distribution (copulas.univariate.Univariate or str):
            Copulas univariate distribution to use.
        categorical_transformer (str):
            Type of transformer to use for the categorical variables, to choose
            from ``one_hot_encoding``, ``label_encoding``, ``categorical`` and
            ``categorical_fuzzy``.
    """

    DEFAULT_DISTRIBUTION = copulas.univariate.Univariate
    _distribution = None
    _categorical_transformer = None
    _model = None

    HYPERPARAMETERS = {
        'distribution': {
            'type': 'str or copulas.univariate.Univariate',
            'default': 'copulas.univariate.Univariate',
            'description': 'Univariate distribution to use to model each column',
            'choices': [
                'copulas.univariate.Univariate',
                'copulas.univariate.GaussianUnivariate',
                'copulas.univariate.GammaUnivariate',
                'copulas.univariate.BetaUnivariate',
                'copulas.univariate.StudentTUnivariate',
                'copulas.univariate.GaussianKDE',
                'copulas.univariate.TruncatedGaussian',
            ]
        },
        'categorical_transformer': {
            'type': 'str',
            'default': 'categoircal_fuzzy',
            'description': 'Type of transformer to use for the categorical variables',
            'choices': [
                'categorical',
                'categorical_fuzzy',
                'one_hot_encoding',
                'label_encoding'
            ]
        }
    }
    DEFAULT_TRANSFORMER = 'one_hot_encoding'
    CATEGORICAL_TRANSFORMERS = {
        'categorical': rdt.transformers.CategoricalTransformer(fuzzy=False),
        'categorical_fuzzy': rdt.transformers.CategoricalTransformer(fuzzy=True),
        'one_hot_encoding': rdt.transformers.OneHotEncodingTransformer,
        'label_encoding': rdt.transformers.LabelEncodingTransformer,
    }
    TRANSFORMER_TEMPLATES = {
        'O': rdt.transformers.OneHotEncodingTransformer
    }

    def __init__(self, distribution=None, categorical_transformer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._metadata is not None and 'model_kwargs' in self._metadata._metadata:
            model_kwargs = self._metadata._metadata['model_kwargs']
            if distribution is None:
                distribution = model_kwargs['distribution']

            if categorical_transformer is None:
                categorical_transformer = model_kwargs['categorical_transformer']

            self._size = model_kwargs['_size']

        self._distribution = distribution or self.DEFAULT_DISTRIBUTION

        categorical_transformer = categorical_transformer or self.DEFAULT_TRANSFORMER
        self._categorical_transformer = categorical_transformer

        self.TRANSFORMER_TEMPLATES['O'] = self.CATEGORICAL_TRANSFORMERS[categorical_transformer]

    def _update_metadata(self):
        parameters = self._model.to_dict()
        univariates = parameters['univariates']
        columns = parameters['columns']

        distributions = {}
        for column, univariate in zip(columns, univariates):
            distributions[column] = univariate['type']

        self._metadata._metadata['model_kwargs'] = {
            'distribution': distributions,
            'categorical_transformer': self._categorical_transformer,
            '_size': self._size
        }

    def _fit(self, data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be fitted.
        """
        self._model = copulas.multivariate.GaussianMultivariate(distribution=self._distribution)
        self._model.fit(data)
        self._update_metadata()

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

    def get_parameters(self, flatten=False):
        """Get copula model parameters.

        Compute model ``covariance`` and ``distribution.std``
        before it returns the flatten dict.

        Args:
            flatten (bool):
                Whether to flatten the parameters or not before
                returning them.

        Returns:
            dict:
                Copula parameters.
        """
        if not flatten:
            return self._model.to_dict()

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

    def set_parameters(self, parameters, unflatten=False):
        """Set copula model parameters.

        Add additional keys after unflatte the parameters
        in order to set expected parameters for the copula.

        Args:
            dict:
                Copula flatten parameters.
            unflatten (bool):
                Whether the parameters need to be unflattened or not.
        """
        if unflatten:
            parameters = unflatten_dict(parameters)
            parameters.setdefault('distribution', self._distribution)

            parameters = self._unflatten_gaussian_copula(parameters)

        self._model = copulas.multivariate.GaussianMultivariate.from_dict(parameters)
