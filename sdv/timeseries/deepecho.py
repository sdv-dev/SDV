"""Timeseries models based on the DeepEcho library.

This modulde defines a base DeepEchoModel class which prepares the
data in the format that the DeepEcho models expect.
This class is also responsible for transforming the sequence index
in a way that can be learned by the models and for transforming it
back to the original data format after sampling.

Currently implemented models are:
    - PAR: Based on the deepecho.models.par.PARModel
"""

import numpy as np
import pandas as pd
import tqdm
from deepecho import PARModel
from deepecho.sequences import assemble_sequences

from sdv.timeseries.base import BaseTimeseriesModel


class DeepEchoModel(BaseTimeseriesModel):
    """Base class for all the SDV Time series models based on DeepEcho."""

    _MODEL_CLASS = None
    _model_kwargs = None
    _verbose = False

    def _build_model(self):
        return self._MODEL_CLASS(**self._model_kwargs)

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

        if self._sequence_index:
            timeseries_data = timeseries_data.rename(columns={
                self._sequence_index + '.value': self._sequence_index
            })

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

        iterator = tqdm.tqdm(context.iterrows(), disable=not self._verbose, total=len(context))

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
        output = output[self._output_columns].reset_index(drop=True)
        if self._sequence_index:
            output = output.rename(columns={
                self._sequence_index: self._sequence_index + '.value'
            })

        return output


class PAR(DeepEchoModel):
    """DeepEcho model based on the deepecho.models.par.PARModel class.

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

    _MODEL_CLASS = PARModel

    def __init__(self, field_names=None, field_types=None, anonymize_fields=None,
                 primary_key=None, entity_columns=None, context_columns=None,
                 sequence_index=None, segment_size=None, context_model=None,
                 table_metadata=None, epochs=128, sample_size=1, cuda=True, verbose=False):
        super().__init__(
            field_names=field_names,
            field_types=field_types,
            anonymize_fields=anonymize_fields,
            primary_key=primary_key,
            entity_columns=entity_columns,
            context_columns=context_columns,
            sequence_index=sequence_index,
            segment_size=segment_size,
            context_model=context_model,
            table_metadata=table_metadata,
        )

        self._model_kwargs = {
            'epochs': epochs,
            'sample_size': sample_size,
            'cuda': cuda,
            'verbose': verbose,
        }
        self._verbose = verbose
