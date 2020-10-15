import numpy as np
import pandas as pd
import tqdm
from deepecho import PARModel
from deepecho.sequences import assemble_sequences
from rdt.transformers import DatetimeTransformer

from sdv.timeseries.base import BaseTimeseriesModel


class DeepEchoModel(BaseTimeseriesModel):
    """Base class for all the SDV Time series models based on DeepEcho."""

    _MODEL_CLASS = None
    _MODEL_KWARGS = None

    _DTYPE_TRANSFORMERS = {
        'O': None,
        'M': DatetimeTransformer(strip_constant=True),
    }
    _DATA_TYPES = {
        'numerical': 'continuous',
        'categorical': 'categorical',
        'boolean': 'categorical',
        'datetime': 'datetime',
    }

    def _build_model(self):
        return self._MODEL_CLASS(**self._MODEL_KWARGS)

    def _transform_sequence_index(self, sequences):
        sequence_index_idx = self._data_columns.index(self._sequence_index)
        for sequence in sequences:
            data = sequence['data']
            sequence_index = data[sequence_index_idx]
            diffs = np.diff(sequence_index).tolist()
            data[sequence_index_idx] = diffs[0:1] + diffs
            data.append(sequence_index[0:1] * len(sequence_index))

    def _fit(self, timeseries_data):
        self._model = self._build_model()

        self._output_columns = list(timeseries_data.columns)
        self._data_columns = [
            column
            for column in timeseries_data.columns
            if column not in self._entity_columns + self._context_columns
        ]

        sequences = assemble_sequences(
            timeseries_data,
            self._entity_columns,
            self._context_columns,
            self._segment_size,
            self._sequence_index,
            drop_sequence_index=False
        )

        data_types = list()
        context_types = list()
        for field, meta in self._metadata.get_fields().items():
            data_type = self._DATA_TYPES.get(meta['type'])
            if data_type:
                if field == self._sequence_index:
                    data_types.append('continuous')
                elif field in self._data_columns:
                    data_types.append(data_type)
                elif field in self._context_columns:
                    context_types.append(data_type)

        if self._sequence_index:
            self._transform_sequence_index(sequences)
            data_types.append('continuous')

        # Validate and fit
        self._model.fit_sequences(sequences, context_types, data_types)

    def _sample(self, context=None, sequence_length=None):
        """Sample new sequences.

        Args:
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
        # Set the entity_columns as index to properly iterate over them
        if self._entity_columns:
            context = context.set_index(self._entity_columns)

        iterator = context.iterrows()
        if self._verbose:
            iterator = tqdm.tqdm(iterator, total=len(context))

        output = list()
        for entity_values, context_values in iterator:
            context_values = context_values.tolist()
            sequence = self._model.sample_sequence(context_values, sequence_length)
            if self._sequence_index:
                sequence_index_idx = self._data_columns.index(self._sequence_index)
                diffs = sequence[sequence_index_idx]
                start = sequence.pop(-1)
                sequence[sequence_index_idx] = np.cumsum(diffs) - diffs[0] + start

            # Reformat as a DataFrame
            group = pd.DataFrame(
                dict(zip(self._data_columns, sequence)),
                columns=self._data_columns
            )
            group[self._entity_columns] = entity_values
            for column, value in zip(self._context_columns, context_values):
                if column == self._sequence_index:
                    sequence_index = group[column]
                    group[column] = sequence_index.cumsum() - sequence_index.iloc[0] + value
                else:
                    group[column] = value

            output.append(group)

        output = pd.concat(output)
        return output[self._output_columns].reset_index(drop=True)


class PAR(DeepEchoModel):

    _MODEL_CLASS = PARModel

    def __init__(self, field_names=None, field_types=None, anonymize_fields=None,
                 primary_key=None, entity_columns=None, context_columns=None,
                 sequence_index=None, segment_size=None, context_model=None,
                 table_metadata=None, epochs=128, max_seq_len=100, sample_size=1,
                 cuda=True, verbose=True):
        super().__init__(
            field_names=field_names,
            field_types=field_types,
            anonymize_fields=anonymize_fields,
            primary_key=primary_key,
            entity_columns=entity_columns,
            context_columns=context_columns,
            sequence_index=sequence_index,
            context_model=context_model,
            table_metadata=table_metadata,
        )

        self._MODEL_KWARGS = {
            'epochs': epochs,
            'max_seq_len': max_seq_len,
            'sample_size': sample_size,
            'cuda': cuda,
            'verbose': verbose,
        }
        self._verbose = verbose
