"""Base Class for timeseries models."""

import copy
import logging
import pickle
import uuid

import pandas as pd
import rdt

from sdv.metadata import Table
from sdv.tabular.copulas import GaussianCopula
from sdv.utils import get_package_versions, throw_version_mismatch_warning

LOGGER = logging.getLogger(__name__)


class BaseTimeseriesModel:
    """Base class for timeseries models.

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
        anonymize_fields (dict[str, str]):
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key (str):
            Name of the field which is the primary key of the table.
        entity_columns (list[str]):
            Names of the columns which identify different time series
            sequences. These will be used to group the data in separated
            training examples.
        context_columns (list[str]):
            The columns in the dataframe which are constant within each
            group/entity. These columns will be provided at sampling time
            (i.e. the samples will be conditioned on the context variables).
        segment_size (int, pd.Timedelta or str):
            If specified, cut each training sequence in several segments of
            the indicated size. The size can either can passed as an integer
            value, which will interpreted as the number of data points to
            put on each segment, or as a pd.Timedelta (or equivalent str
            representation), which will be interpreted as the segment length
            in time. Timedelta segment sizes can only be used with sequence
            indexes of type datetime.
        sequence_index (str):
            Name of the column that acts as the order index of each
            sequence. The sequence index column can be of any type that can
            be sorted, such as integer values or datetimes.
        context_model (str or sdv.tabular.BaseTabularModel):
            Model to use to sample the context rows. It can be passed as a
            a string, which must be one of the following:

            * `gaussian_copula` (default): Use a GaussianCopula model.

            Alternatively, a preconfigured Tabular model instance can be
            passed.

        table_metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
            If given alongside any other metadata-related arguments, an
            exception will be raised.
            If not given at all, it will be built using the other
            arguments or learned from the data.
    """

    _DTYPE_TRANSFORMERS = {
        'i': None,
        'f': None,
        'M': rdt.transformers.DatetimeTransformer(strip_constant=True),
        'b': None,
        'O': None,
    }
    _CONTEXT_MODELS = {
        'gaussian_copula': (GaussianCopula, {'categorical_transformer': 'categorical_fuzzy'})
    }

    _metadata = None

    def __init__(self, field_names=None, field_types=None, anonymize_fields=None,
                 primary_key=None, entity_columns=None, context_columns=None,
                 sequence_index=None, segment_size=None, context_model=None,
                 table_metadata=None):
        if table_metadata is None:
            self._metadata = Table(
                field_names=field_names,
                primary_key=primary_key,
                field_types=field_types,
                anonymize_fields=anonymize_fields,
                dtype_transformers=self._DTYPE_TRANSFORMERS,
                sequence_index=sequence_index,
                entity_columns=entity_columns,
                context_columns=context_columns,
            )
            self._metadata_fitted = False
        else:
            null_args = (
                field_names,
                primary_key,
                field_types,
                anonymize_fields,
                sequence_index,
                entity_columns,
                context_columns
            )
            for arg in null_args:
                if arg:
                    raise ValueError(
                        'If table_metadata is given {} must be None'.format(arg.__name__))

            if isinstance(table_metadata, dict):
                table_metadata = Table.from_dict(
                    table_metadata,
                    dtype_transformers=self._DTYPE_TRANSFORMERS,
                )

            self._metadata = table_metadata
            self._metadata_fitted = table_metadata.fitted

        # Validate arguments
        if segment_size is not None and not isinstance(segment_size, int):
            if sequence_index is None:
                raise TypeError(
                    '`segment_size` must be of type `int` if '
                    'no `sequence_index` is given.'
                )

            segment_size = pd.to_timedelta(segment_size)

        self._context_columns = self._metadata._context_columns
        self._entity_columns = self._metadata._entity_columns
        self._sequence_index = self._metadata._sequence_index
        self._segment_size = segment_size

        context_model = context_model or 'gaussian_copula'
        if isinstance(context_model, str):
            context_model = self._CONTEXT_MODELS[context_model]

        self._context_model_template = context_model

    def _fit(self, timeseries_data):
        raise NotImplementedError()

    def _fit_context_model(self, transformed):
        template = self._context_model_template
        default_kwargs = {
            'primary_key': self._entity_columns,
            'field_types': {
                name: meta
                for name, meta in self._metadata.get_fields().items()
                if name in self._entity_columns
            }
        }
        if isinstance(template, tuple):
            context_model_class, context_model_kwargs = copy.deepcopy(template)
            if 'primary_key' not in context_model_kwargs:
                context_model_kwargs['primary_key'] = self._entity_columns
                for keyword, argument in default_kwargs.items():
                    if keyword not in context_model_kwargs:
                        context_model_kwargs[keyword] = argument

            self._context_model = context_model_class(**context_model_kwargs)
        elif isinstance(template, type):
            self._context_model = template(**default_kwargs)
        else:
            self._context_model = copy.deepcopy(template)

        LOGGER.debug('Fitting context model %s', self._context_model.__class__.__name__)
        if self._context_columns:
            context = transformed[self._entity_columns + self._context_columns]
        else:
            context = transformed[self._entity_columns].copy()
            # Add constant column to allow modeling
            context[str(uuid.uuid4())] = 0

        context = context.groupby(self._entity_columns).first().reset_index()
        self._context_model.fit(context)

    def fit(self, timeseries_data):
        """Fit this model to the data.

        Args:
            timseries_data (pandas.DataFrame):
                pandas.DataFrame containing both the sequences,
                the entity columns and the context columns.
        """
        LOGGER.debug('Fitting %s to table %s; shape: %s', self.__class__.__name__,
                     self._metadata.name, timeseries_data.shape)
        if not self._metadata_fitted:
            self._metadata.fit(timeseries_data)

        LOGGER.debug('Transforming table %s; shape: %s',
                     self._metadata.name, timeseries_data.shape)
        transformed = self._metadata.transform(timeseries_data)

        for column in self._entity_columns:
            transformed[column] = timeseries_data[column]

        if self._entity_columns:
            self._fit_context_model(transformed)

        LOGGER.debug('Fitting %s model to table %s', self.__class__.__name__, self._metadata.name)
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

    def _sample(self, context=None, sequence_length=None):
        raise NotImplementedError()

    def sample(self, num_sequences=None, context=None, sequence_length=None):
        """Sample new sequences.

        Args:
            num_sequences (int):
                Number of sequences to sample. If context is
                passed, this is ignored. If not given, the
                same number of sequences as in the original
                timeseries_data is sampled.
            context (pandas.DataFrame):
                Context values to use when generating the sequences.
                If not passed, the context values will be sampled
                using the specified tabular model.
            sequence_length (int):
                If passed, sample sequences of this length. If not
                given, the sequence length will be sampled from
                the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same
                format as that he training data had.
        """
        if not self._entity_columns:
            if context is not None:
                raise TypeError('If there are no entity_columns, context must be None')

            context = pd.DataFrame(index=range(num_sequences or 1))
        elif context is None:
            context = self._context_model.sample(num_sequences, output_file_path='disable')
            for column in self._entity_columns or []:
                if column not in context:
                    context[column] = range(len(context))

        sampled = self._sample(context, sequence_length)
        return self._metadata.reverse_transform(sampled)

    def save(self, path):
        """Save this model instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        self._package_versions = get_package_versions(getattr(self, '_model', None))

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
            model = pickle.load(f)
            throw_version_mismatch_warning(getattr(model, '_package_versions', None))

            return model
