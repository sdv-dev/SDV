"""PAR Synthesizer class."""

import inspect
import logging
import uuid

import numpy as np
from deepecho import PARModel
from deepecho.sequences import assemble_sequences

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table.base import BaseSynthesizer
from sdv.utils import cast_to_iterable

LOGGER = logging.getLogger(__name__)


class PARSynthesizer(BaseSynthesizer):
    """Synthesizer for sequential data.

    This synthesizer uses the ``deepecho.models.par.PARModel`` class as the core model.
    Additionally, it uses a separate synthesizer to model and sample the context columns
    to be passed into PAR.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        context_columns (list[str]):
            A list of strings, representing the columns that do not vary in a sequence.
        segment_size (int):
            If specified, cut each training sequence in several segments of
            the indicated size. The size can be passed as an integer
            value, which will interpreted as the number of data points to
            put on each segment.
        epochs (int):
            The number of epochs to train for. Defaults to 128.
        sample_size (int):
            The number of times to sample (before choosing and
            returning the sample which maximizes the likelihood).
            Defaults to 1.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        verbose (bool):
            Whether to print progress to console or not.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'numerical': None,
        'boolean': None
    }

    def _get_context_metadata(self):
        context_columns_dict = {}
        context_columns = self.context_columns.copy() if self.context_columns else []
        if self._sequence_key:
            context_columns += self._sequence_key

        for column in context_columns:
            context_columns_dict[column] = self.metadata._columns[column]

        context_metadata_dict = {'columns': context_columns_dict}
        return SingleTableMetadata._load_from_dict(context_metadata_dict)

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=False,
                 context_columns=None, segment_size=None, epochs=128, sample_size=1, cuda=True,
                 verbose=False):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        sequence_key = self.metadata._sequence_key
        self._sequence_key = list(cast_to_iterable(sequence_key)) if sequence_key else None
        self._sequence_index = self.metadata._sequence_index
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.context_columns = context_columns
        self.segment_size = segment_size
        self._model_kwargs = {
            'epochs': epochs,
            'sample_size': sample_size,
            'cuda': cuda,
            'verbose': verbose,
        }
        context_metadata = self._get_context_metadata()
        self._context_synthesizer = GaussianCopulaSynthesizer(
            metadata=context_metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding
        )

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        for parameter_name, value in self._model_kwargs.items():
            instantiated_parameters[parameter_name] = value

        return instantiated_parameters

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        For PAR, none of the sequence keys are transformed.
        """
        sequence_key_transformers = {sequence_key: None for sequence_key in self._sequence_key}
        if self._data_processor._hyper_transformer.field_transformers == {}:
            self.auto_assign_transformers(data)

        self.update_transformers(sequence_key_transformers)
        return super().preprocess(data)

    def _fit_context_model(self, transformed):
        LOGGER.debug(f'Fitting context synthesizer {self._context_synthesizer.__class__.__name__}')
        if self.context_columns:
            context = transformed[self._sequence_key + self.context_columns]
        else:
            context = transformed[self._sequence_key].copy()
            # Add constant column to allow modeling
            context[str(uuid.uuid4())] = 0

        context = context.groupby(self._sequence_key).first().reset_index()
        self._context_synthesizer.fit(context)

    def _build_model(self):
        return PARModel(**self._model_kwargs)

    def _transform_sequence_index(self, sequences):
        sequence_index_idx = self._data_columns.index(self._sequence_index)
        for sequence in sequences:
            data = sequence['data']
            sequence_index = data[sequence_index_idx]
            diffs = np.diff(sequence_index).tolist()
            data[sequence_index_idx] = diffs[0:1] + diffs
            data.append(sequence_index[0:1] * len(sequence_index))

    def _fit_sequence_columns(self, timeseries_data):
        self._model = self._build_model()

        if self._sequence_index:
            timeseries_data = timeseries_data.rename(columns={
                self._sequence_index + '.value': self._sequence_index
            })

        self._output_columns = list(timeseries_data.columns)
        self._data_columns = [
            column
            for column in timeseries_data.columns
            if column not in self._sequence_key + self.context_columns
        ]

        sequences = assemble_sequences(
            timeseries_data,
            self._sequence_key,
            self.context_columns,
            self.segment_size,
            self._sequence_index,
            drop_sequence_index=False
        )

        data_types = []
        context_types = []
        for field in self._output_columns:
            dtype = timeseries_data[field].dtype
            kind = dtype.kind
            if kind in ('i', 'f'):
                data_type = 'continuous'
            elif kind in ('O', 'b'):
                data_type = 'categorical'
            else:
                raise ValueError(f'Unsupported dtype {dtype}')

            if field in self._data_columns:
                data_types.append(data_type)
            elif field in self.context_columns:
                context_types.append(data_type)

        if self._sequence_index:
            self._transform_sequence_index(sequences)
            data_types.append('continuous')

        # Validate and fit
        self._model.fit_sequences(sequences, context_types, data_types)

    def _fit(self, processed_data):
        """Fit this model to the data.

        Args:
            processed_data (pandas.DataFrame):
                pandas.DataFrame containing both the sequences,
                the entity columns and the context columns.
        """
        if self._sequence_key:
            self._fit_context_model(processed_data)

        LOGGER.debug(f'Fitting {self.__class__.__name__} model to table')
        self._fit_sequence_columns(processed_data)
