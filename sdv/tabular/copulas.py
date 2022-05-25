"""Wrappers around copulas models."""

import logging
import warnings

import copulas
import copulas.multivariate
import copulas.univariate
import numpy as np
import scipy

from sdv.metadata import Table
from sdv.tabular.base import BaseTabularModel, NonParametricError
from sdv.tabular.utils import flatten_dict, unflatten_dict

LOGGER = logging.getLogger(__name__)


class GaussianCopula(BaseTabularModel):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        field_names (list[str]):
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types (dict[str, dict]):
            Dictinary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        field_transformers (dict[str, str]):
            Dictinary specifying which transformers to use for each field.
            Available transformers are:

                * ``integer``: Uses a ``NumericalTransformer`` of dtype ``int``.
                * ``float``: Uses a ``NumericalTransformer`` of dtype ``float``.
                * ``categorical``: Uses a ``CategoricalTransformer`` without gaussian noise.
                * ``categorical_fuzzy``: Uses a ``CategoricalTransformer`` adding gaussian noise.
                * ``one_hot_encoding``: Uses a ``OneHotEncodingTransformer``.
                * ``label_encoding``: Uses a ``LabelEncodingTransformer``.
                * ``boolean``: Uses a ``BooleanTransformer``.
                * ``datetime``: Uses a ``DatetimeTransformer``.

        anonymize_fields (dict[str, str]):
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key (str):
            Name of the field which is the primary key of the table.
        constraints (list[Constraint, dict]):
            List of Constraint objects or dicts.
        table_metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
            If given alongside any other metadata-related arguments, an
            exception will be raised.
            If not given at all, it will be built using the other
            arguments or learned from the data.
        field_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used. The distributions can be passed as either
            a ``copulas.univariate`` instance or as one of the following values:

                * ``gaussian``: Use a Gaussian distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``beta``: Use a Beta distribution.
                * ``student_t``: Use a Student T distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
                * ``truncated_gaussian``: Use a Truncated Gaussian distribution.

        default_distribution (copulas.univariate.Univariate or str):
            Copulas univariate distribution to use by default. To choose from the list
            of possible ``field_distribution`` values.
            Defaults to ``truncated_gaussian``.
        categorical_transformer (str):
            Type of transformer to use for the categorical variables, which must be one of the
            following values:

                * ``one_hot_encoding``: Apply a OneHotEncodingTransformer to the
                  categorical column, which replaces the  column with one boolean
                  column for each possible category, indicating whether each row
                  had that value or not.
                * ``label_encoding``: Apply a LabelEncodingTransformer, which
                  replaces the value of each category with an integer value that
                  acts as its *label*.
                * ``categorical``: Apply CategoricalTransformer, which replaces
                  each categorical value with a float number in the `[0, 1]` range
                  which is inversely proportional to the frequency of that category.
                * ``categorical_fuzzy``: Apply a CategoricalTransformer with the
                  ``fuzzy`` argument set to ``True``, which makes it add gaussian
                  noise around each value.
            Defaults to ``categorical_fuzzy``.
        rounding (int, str or None):
            Define rounding scheme for ``NumericalTransformer``. If set to an int, values
            will be rounded to that number of decimal places. If ``None``, values will not
            be rounded. If set to ``'auto'``, the transformer will round to the maximum number
            of decimal places detected in the fitted data. Defaults to ``'auto'``.
        min_value (int, str or None):
            Specify the minimum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be greater than or equal to it. If the string ``'auto'``
            is given, the minimum will be the minimum value seen in the fitted data. If ``None``
            is given, there won't be a minimum. Defaults to ``'auto'``.
        max_value (int, str or None):
            Specify the maximum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be less than or equal to it. If the string ``'auto'``
            is given, the maximum will be the maximum value seen in the fitted data. If ``None``
            is given, there won't be a maximum. Defaults to ``'auto'``.
    """

    _field_distributions = None
    _default_distribution = None
    _categorical_transformer = None
    _model = None

    _DISTRIBUTIONS = {
        'gaussian': copulas.univariate.GaussianUnivariate,
        'gamma': copulas.univariate.GammaUnivariate,
        'beta': copulas.univariate.BetaUnivariate,
        'student_t': copulas.univariate.StudentTUnivariate,
        'gaussian_kde': copulas.univariate.GaussianKDE,
        'truncated_gaussian': copulas.univariate.TruncatedGaussian,
    }
    _DEFAULT_DISTRIBUTION = _DISTRIBUTIONS['truncated_gaussian']
    _DEFAULT_TRANSFORMER = 'categorical_fuzzy'

    @classmethod
    def _validate_distribution(cls, distribution):
        if not isinstance(distribution, str):
            return distribution
        if distribution in cls._DISTRIBUTIONS:
            return cls._DISTRIBUTIONS[distribution]

        try:
            copulas.get_instance(distribution)
            return distribution
        except (ValueError, ImportError):
            error_message = 'Invalid distribution specification {}'.format(distribution)
            raise ValueError(error_message) from None

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 field_distributions=None, default_distribution=None,
                 categorical_transformer=None, rounding='auto', min_value='auto',
                 max_value='auto'):

        if isinstance(table_metadata, dict):
            table_metadata = Table.from_dict(table_metadata)

        if table_metadata:
            model_kwargs = table_metadata.get_model_kwargs(self.__class__.__name__)
            if model_kwargs:
                if field_distributions is None:
                    field_distributions = model_kwargs['field_distributions']

                if default_distribution is None:
                    default_distribution = model_kwargs['default_distribution']

                if categorical_transformer is None:
                    categorical_transformer = model_kwargs['categorical_transformer']

        if field_distributions and not isinstance(field_distributions, dict):
            raise TypeError('field_distributions can only be None or a dict instance')

        self._field_distributions = {
            field: self._validate_distribution(distribution)
            for field, distribution in (field_distributions or {}).items()
        }
        self._default_distribution = (
            self._validate_distribution(default_distribution) or self._DEFAULT_DISTRIBUTION
        )

        self._categorical_transformer = categorical_transformer or self._DEFAULT_TRANSFORMER
        self._DTYPE_TRANSFORMERS = {'O': self._categorical_transformer}

        super().__init__(
            field_names=field_names,
            field_types=field_types,
            field_transformers=field_transformers,
            anonymize_fields=anonymize_fields,
            primary_key=primary_key,
            constraints=constraints,
            table_metadata=table_metadata,
            rounding=rounding,
            max_value=max_value,
            min_value=min_value
        )

        self._metadata.set_model_kwargs(self.__class__.__name__, {
            'field_distributions': field_distributions,
            'default_distribution': default_distribution,
            'categorical_transformer': categorical_transformer,
        })

    def get_distributions(self):
        """Get the marginal distributions used by this copula.

        Returns:
            dict:
                Dictionary containing the distributions used or detected
                for each column.
        """
        parameters = self._model.to_dict()
        univariates = parameters['univariates']
        columns = parameters['columns']

        distributions = {}
        for column, univariate in zip(columns, univariates):
            distributions[column] = univariate['type']

        return distributions

    def _update_metadata(self):
        """Add arguments needed to reproduce this model to the Metadata.

        Additional arguments include:
            - Distribution found for each column
            - categorical_transformer
        """
        class_name = self.__class__.__name__
        distributions = self.get_distributions()
        self._metadata.set_model_kwargs(class_name, {
            'field_distributions': distributions,
            'default_distribution': self._default_distribution,
            'categorical_transformer': self._categorical_transformer,
        })

    def _fit(self, table_data):
        """Fit the model to the table.

        Args:
            table_data (pandas.DataFrame):
                Data to be fitted.
        """
        for column in table_data.columns:
            if column not in self._field_distributions:
                # Check if the column is a derived column.
                column_name = column.replace('.value', '')
                self._field_distributions[column] = self._field_distributions.get(
                    column_name, self._default_distribution)

        self._model = copulas.multivariate.GaussianMultivariate(
            distribution=self._field_distributions)

        LOGGER.debug('Fitting %s to table %s; shape: %s', self._model.__class__.__name__,
                     self._metadata.name, table_data.shape)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='scipy')
            self._model.fit(table_data)

        self._update_metadata()

    def sample_conditions(self, conditions, batch_size=None, randomize_samples=True,
                          output_file_path=None):
        """Sample rows from this table with the given conditions.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of sdv.sampling.Condition objects, which specify the column
                values in a condition, along with the number of rows for that
                condition.
            batch_size (int or None):
                The batch size to sample. Defaults to `num_rows`, if None.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to
                a temporary file, if None.

        Returns:
            pandas.DataFrame:
                Sampled data.

        Raises:
            ConstraintsNotMetError:
                If the conditions are not valid for the given constraints.
            ValueError:
                If any of the following happens:
                    * any of the conditions' columns are not valid.
                    * no rows could be generated.
        """
        return self._sample_conditions(
            conditions, 100, batch_size, randomize_samples, output_file_path)

    def sample_remaining_columns(self, known_columns, batch_size=None, randomize_samples=True,
                                 output_file_path=None):
        """Sample rows from this table.

        Args:
            known_columns (pandas.DataFrame):
                A pandas.DataFrame with the columns that are already known. The output
                is a DataFrame such that each row in the output is sampled
                conditionally on the corresponding row in the input.
            batch_size (int or None):
                The batch size to sample. Defaults to `num_rows`, if None.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to
                a temporary file, if None.

        Returns:
            pandas.DataFrame:
                Sampled data.

        Raises:
            ConstraintsNotMetError:
                If the conditions are not valid for the given constraints.
            ValueError:
                If any of the following happens:
                    * any of the conditions' columns are not valid.
                    * no rows could be generated.
        """
        return self._sample_remaining_columns(
            known_columns, 100, batch_size, randomize_samples, output_file_path)

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates `num_rows` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._model.sample(num_rows, conditions=conditions)

    def _set_random_state(self, random_state):
        """Set the random state of the model's random number generator.

        Args:
            random_state (int, np.random.RandomState, or None):
                Seed or RandomState to use.
        """
        self._model.set_random_state(random_state)

    def get_likelihood(self, table_data):
        """Get the likelihood of each row belonging to this table."""
        transformed = self._metadata.transform(table_data)
        return self._model.probability_density(transformed)

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
                raise NonParametricError("This GaussianCopula uses non parametric distributions")

        params = self._model.to_dict()

        covariance = list()
        for index, row in enumerate(params['covariance'][1:]):
            covariance.append(row[:index + 1])

        params['covariance'] = covariance
        params['univariates'] = dict(zip(params.pop('columns'), params['univariates']))

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
    def _rebuild_correlation_matrix(cls, triangular_covariance):
        """Rebuild a valid correlation matrix from its lower half triangle.

        The input of this function is a list of lists of floats of size 1, 2, 3...n-1:

           [[c_{2,1}], [c_{3,1}, c_{3,2}], ..., [c_{n,1},...,c_{n,n-1}]]

        Corresponding to the values from the lower half of the original correlation matrix,
        **excluding** the diagonal.

        The output is the complete correlation matrix reconstructed using the given values
        and scaled to the :math:`[-1, 1]` range if necessary.

        Args:
            triangle_covariange (list[list[float]]):
                A list that contains lists of floats of size 1, 2, 3... up to ``n-1``,
                where ``n`` is the size of the target covariance matrix.

        Returns:
            numpy.ndarray:
                rebuilt correlation matrix.
        """
        zero = [0.0]
        size = len(triangular_covariance) + 1
        left = np.zeros((size, size))
        right = np.zeros((size, size))
        for idx, values in enumerate(triangular_covariance):
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
        columns = list()
        univariates = list()
        for column, univariate in model_parameters['univariates'].items():
            columns.append(column)
            univariate['type'] = self._field_distributions[column]
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
