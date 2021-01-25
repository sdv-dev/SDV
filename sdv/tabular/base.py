"""Base Class for tabular models."""

import logging
import pickle

import pandas as pd

from sdv.metadata import Table

LOGGER = logging.getLogger(__name__)


class NonParametricError(Exception):
    """Exception to indicate that a model is not parametric."""


class BaseTabularModel:
    """Base class for all the tabular models.

    The ``BaseTabularModel`` class defines the common API that all the
    TabularModels need to implement, as well as common functionality.

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
    """

    _DTYPE_TRANSFORMERS = None

    _metadata = None

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None):
        if table_metadata is None:
            self._metadata = Table(
                field_names=field_names,
                primary_key=primary_key,
                field_types=field_types,
                field_transformers=field_transformers,
                anonymize_fields=anonymize_fields,
                constraints=constraints,
                dtype_transformers=self._DTYPE_TRANSFORMERS,
            )
            self._metadata_fitted = False
        else:
            for arg in (field_names, primary_key, field_types, anonymize_fields, constraints):
                if arg:
                    raise ValueError(
                        'If table_metadata is given {} must be None'.format(arg.__name__))

            if isinstance(table_metadata, dict):
                table_metadata = Table.from_dict(table_metadata)

            table_metadata._dtype_transformers.update(self._DTYPE_TRANSFORMERS)

            self._metadata = table_metadata
            self._metadata_fitted = table_metadata.fitted

    def fit(self, data):
        """Fit this model to the data.

        If the table metadata has not been given, learn it from the data.

        Args:
            data (pandas.DataFrame or str):
                Data to fit the model to. It can be passed as a
                ``pandas.DataFrame`` or as an ``str``.
                If an ``str`` is passed, it is assumed to be
                the path to a CSV file which can be loaded using
                ``pandas.read_csv``.
        """
        LOGGER.debug('Fitting %s to table %s; shape: %s', self.__class__.__name__,
                     self._metadata.name, data.shape)
        if not self._metadata_fitted:
            self._metadata.fit(data)

        self._num_rows = len(data)

        LOGGER.debug('Transforming table %s; shape: %s', self._metadata.name, data.shape)
        transformed = self._metadata.transform(data)

        if self._metadata.get_dtypes(ids=False):
            LOGGER.debug(
                'Fitting %s model to table %s', self.__class__.__name__, self._metadata.name)
            self._fit(transformed)

    def get_metadata(self):
        """Get metadata about the table.

        This will return an ``sdv.metadata.Table`` object containing
        the information about the data that this model has learned.

        This Table metadata will contain some common information,
        such as field names and data types, as well as additional
        information that each Sub-class might add, such as the
        observed data field distributions and their parameters.

        Returns:
            sdv.metadata.Table:
                Table metadata.
        """
        return self._metadata

    def _sample_rows(self, num_to_sample):
        if self._metadata.get_dtypes(ids=False):
            return self._sample(num_to_sample)
        else:
            return pd.DataFrame(index=range(num_to_sample))

    def sample(self, num_rows=None, max_retries=100):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
            max_retries (int):
                Number of times to retry sampling discarded rows.
                Defaults to 100.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        num_rows = num_rows or self._num_rows
        num_to_sample = num_rows
        sampled = self._sample_rows(num_to_sample)
        sampled = self._metadata.reverse_transform(sampled)
        sampled = self._metadata.filter_valid(sampled)
        num_valid = len(sampled)

        counter = 0
        total_sampled = num_to_sample
        while num_valid < num_rows:
            counter += 1
            if counter >= max_retries:
                raise ValueError('Could not get enough valid rows within %s trials', max_retries)

            remaining = num_rows - num_valid
            valid_ratio = num_valid / total_sampled
            num_to_sample = int(counter * remaining / (valid_ratio if valid_ratio != 0 else 1))

            LOGGER.info('%s valid rows remaining. Resampling %s rows', remaining, num_to_sample)
            resampled = self._sample_rows(num_to_sample)
            resampled = self._metadata.reverse_transform(resampled)

            sampled = sampled.append(resampled)
            sampled = self._metadata.filter_valid(sampled)
            num_valid = len(sampled)

        return sampled.head(num_rows)

    def _get_parameters(self):
        raise NonParametricError()

    def get_parameters(self):
        """Get the parameters learned from the data.

        The result is a flat dict (single level) which contains
        all the necessary parameters to be able to reproduce
        this model.

        Subclasses which are not parametric, such as DeepLearning
        based models, raise a NonParametricError indicating that
        this method is not supported for their implementation.

        Returns:
            parameters (dict):
                flat dict (single level) which contains all the
                necessary parameters to be able to reproduce
                this model.

        Raises:
            NonParametricError:
                If the model is not parametric or cannot be described
                using a simple dictionary.
        """
        if self._metadata.get_dtypes(ids=False):
            parameters = self._get_parameters()
        else:
            parameters = {}

        parameters['num_rows'] = self._num_rows
        return parameters

    def _set_parameters(self, parameters):
        raise NonParametricError()

    def set_parameters(self, parameters):
        """Regenerate a previously learned model from its parameters.

        Subclasses which are not parametric, such as DeepLearning
        based models, raise a NonParametricError indicating that
        this method is not supported for their implementation.

        Args:
            dict:
                Model parameters.

        Raises:
            NonParametricError:
                If the model is not parametric or cannot be described
                using a simple dictionary.
        """
        num_rows = parameters.pop('num_rows')
        self._num_rows = 0 if pd.isnull(num_rows) else max(0, int(round(num_rows)))

        if self._metadata.get_dtypes(ids=False):
            self._set_parameters(parameters)

    def save(self, path):
        """Save this model instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a TabularModel instance from a given path.

        Args:
            path (str):
                Path from which to load the instance.

        Returns:
            TabularModel:
                The loaded tabular model.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
