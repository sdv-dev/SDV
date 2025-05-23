"""PAR Synthesizer class."""

import inspect
import logging
import uuid
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import tqdm
from deepecho import PARModel
from deepecho.sequences import assemble_sequences
from rdt.transformers import FloatFormatter

from sdv._utils import _cast_to_iterable, _groupby_list
from sdv.cag import ProgrammableConstraint
from sdv.cag._utils import _validate_constraints_single_table
from sdv.errors import SamplingError, SynthesizerInputError
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata import Metadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table.base import BaseSynthesizer
from sdv.single_table.ctgan import LossValuesMixin

LOGGER = logging.getLogger(__name__)


class PARSynthesizer(LossValuesMixin, BaseSynthesizer):
    """Synthesizer for sequential data.

    This synthesizer uses the ``deepecho.models.par.PARModel`` class as the core model.
    Additionally, it uses a separate synthesizer to model and sample the context columns
    to be passed into PAR.

    Args:
        metadata (sdv.metadata.Metadata):
            Metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
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

    _model_sdtype_transformers = {'categorical': None, 'numerical': None, 'boolean': None}

    def _get_context_metadata(self):
        context_metadata_dict = {}
        context_columns_dict = {}
        context_columns = self.context_columns.copy() if self.context_columns else []
        if self._sequence_key:
            context_columns += self._sequence_key
            context_metadata_dict['primary_key'] = self._sequence_key[0]

        for column, column_metadata in self._extra_context_columns.items():
            context_columns_dict[column] = column_metadata

        table_metadata = self.metadata.tables[self._table_name]
        for column in context_columns:
            context_columns_dict[column] = table_metadata.columns[column]

        context_metadata_dict['columns'] = self._update_context_column_dict(context_columns_dict)
        return SingleTableMetadata.load_from_dict(context_metadata_dict)

    def _update_context_column_dict(self, context_columns_dict):
        """Update context column dictionary based on available transformers.

        Args:
            context_columns_dict (dict):
                Dictionary of context columns.

        Returns:
            dict:
                Updated context column metadata.
        """
        default_transformers_by_sdtype = deepcopy(self._data_processor._transformers_by_sdtype)
        table_metadata = self.metadata.tables[self._table_name]
        for column in self.context_columns:
            column_metadata = table_metadata.columns[column]
            if default_transformers_by_sdtype.get(column_metadata['sdtype']):
                context_columns_dict[column] = {'sdtype': 'numerical'}

        return context_columns_dict

    def _get_context_columns_for_processing(self):
        columns_to_be_processed = []
        default_transformers_by_sdtype = deepcopy(self._data_processor._transformers_by_sdtype)
        table_metadata = self._get_table_metadata()
        for column in self.context_columns:
            if default_transformers_by_sdtype.get(table_metadata.columns[column]['sdtype']):
                columns_to_be_processed.append(column)

        return columns_to_be_processed

    def __init__(
        self,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=['en_US'],
        context_columns=None,
        segment_size=None,
        epochs=128,
        sample_size=1,
        cuda=True,
        verbose=False,
    ):
        if type(metadata) is Metadata and len(metadata.tables) > 1:
            raise InvalidMetadataError('PARSynthesizer can only be used with a single table.')

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        # Cast the sequence key to an iterable. While the metadata only supports a single sequence
        # key, but PAR is set up to handle composite sequence keys
        sequence_key = self._get_table_metadata().sequence_key
        self._sequence_key = list(_cast_to_iterable(sequence_key)) if sequence_key else None
        if not self._sequence_key:
            raise SynthesizerInputError(
                'The PARSythesizer is designed for multi-sequence data, identifiable through a '
                'sequence key. Your metadata does not include a sequence key.'
            )

        self._sequence_index = self._get_table_metadata().sequence_index
        self.context_columns = context_columns or []
        self._validate_sequence_key_and_context_columns()
        self._extra_context_columns = {}
        self.extended_columns = {}
        self.segment_size = segment_size
        self._model_kwargs = {
            'epochs': epochs,
            'sample_size': sample_size,
            'cuda': cuda,
            'verbose': verbose,
        }
        context_metadata = self._get_context_metadata()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=".*The 'SingleTableMetadata' is deprecated.*")
            self._context_synthesizer = GaussianCopulaSynthesizer(
                metadata=context_metadata,
                enforce_min_max_values=enforce_min_max_values,
                enforce_rounding=enforce_rounding,
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

    def add_constraints(self, constraints):
        """Add constraints to the synthesizer.

        For PARSynthesizers only allow a list of constraints that follow these rules:

        1) All constraints must be either for all contextual columns or non-contextual column.
           No mixing constraints that cover both contextual and non-contextual columns
        2) No overlapping constraints (there are no constraints that act on the same column)
        3) No custom constraints

        Args:
            constraints (list):
                List of constraints described as dictionaries in the following format:
                    * ``constraint_class``: Name of the constraint to apply.
                    * ``constraint_parameters``: A dictionary with the constraint parameters.
        """
        _validate_constraints_single_table(constraints, self._fitted)
        metadata_columns = self.get_metadata('modified').get_column_names()
        transformer_by_sdtype = self._data_processor._transformers_by_sdtype
        context_set = set(self.context_columns)
        constraint_cols = []
        for constraint in constraints:
            if isinstance(constraint, ProgrammableConstraint):
                raise SynthesizerInputError(
                    'The PARSynthesizer cannot accommodate with programmable constraints.'
                )

            columns = []
            init_signature = inspect.signature(constraint.__class__.__init__)
            init_param_names = [p for p in init_signature.parameters if p != 'self']
            for param in init_param_names:
                value = getattr(constraint, param, None)
                if value is None:
                    value = getattr(constraint, f'_{param}', None)

                if isinstance(value, str) and value in metadata_columns:
                    columns.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str) and item in metadata_columns:
                            columns.append(item)

            for col in columns:
                if col in constraint_cols:
                    raise SynthesizerInputError(
                        'The PARSynthesizer cannot accommodate multiple constraints '
                        'that overlap on the same columns.'
                    )
                constraint_cols.append(col)

        all_context = all(col in context_set for col in constraint_cols)
        no_context = all(col not in context_set for col in constraint_cols)
        if all_context or no_context:
            super().add_constraints(constraints)
            self._data_processor._transformers_by_sdtype = transformer_by_sdtype
            if all_context:
                new_columns = self.metadata.get_column_names()
                extra_columns = set(new_columns).difference(set(metadata_columns))
                self.context_columns = list(
                    extra_columns.union(context_set.intersection(set(new_columns)))
                )
                context_metadata = self._get_context_metadata()
                self._context_synthesizer.metadata = context_metadata
        else:
            raise SynthesizerInputError(
                'The PARSynthesizer cannot accommodate constraints '
                'with a mix of context and non-context columns.'
            )

    def load_custom_constraint_classes(self, filepath, class_names):
        """Error that tells the user custom constraints can't be used in the ``PARSynthesizer``."""
        raise SynthesizerInputError('The PARSynthesizer cannot accommodate custom constraints.')

    def _validate_sequence_key_and_context_columns(self):
        """Check that the sequence key is not present in the context colums.

        Args:
            sequence_key (list[str]):
                A list of column that identify which row(s) belong to which sequences.
            context_columns (list[str]):
                A list of strings, representing the columns that do not vary in a sequence.
        """
        if set(self._sequence_key).intersection(set(self.context_columns)):
            raise SynthesizerInputError(
                f'The sequence key {self._sequence_key} cannot be a context column. '
                'To proceed, please remove the sequence key from the context_columns parameter.'
            )

    def _validate_context_columns(self, data):
        errors = []
        if self.context_columns:
            for sequence_key_value, data_values in data.groupby(_groupby_list(self._sequence_key)):
                for context_column in self.context_columns:
                    if len(data_values[context_column].unique()) > 1:
                        errors.append(
                            (
                                f"Context column '{context_column}' is changing inside sequence "
                                f'({self._sequence_key}={sequence_key_value}).'
                            )
                        )

        return errors

    def _validate(self, data):
        data = self._validate_transform_constraints(data)
        return self._validate_context_columns(data)

    def _transform_sequence_index(self, data):
        sequence_index = data[self._sequence_key + [self._sequence_index]]
        sequence_index_context = sequence_index.groupby(self._sequence_key).agg('first')
        sequence_index_context = sequence_index_context.rename(
            columns={self._sequence_index: f'{self._sequence_index}.context'}
        )
        if all(sequence_index[self._sequence_key].nunique() == 1):
            sequence_index_sequence = sequence_index[[self._sequence_index]].diff().bfill()
        else:
            sequence_index_sequence = (
                sequence_index.groupby(self._sequence_key)
                .apply(lambda x: x[self._sequence_index].diff().bfill())
                .droplevel(1)
                .reset_index()
            )

        if all(sequence_index_sequence[self._sequence_index].isna()):
            fill_value = 0
        else:
            fill_value = min(sequence_index_sequence[self._sequence_index].dropna())
        sequence_index_sequence = sequence_index_sequence.fillna(fill_value)

        data[self._sequence_index] = sequence_index_sequence[self._sequence_index].to_numpy()
        data = data.merge(sequence_index_context, left_on=self._sequence_key, right_index=True)

        self.extended_columns[self._sequence_index] = FloatFormatter(enforce_min_max_values=True)
        self.extended_columns[self._sequence_index].fit(
            sequence_index_sequence, self._sequence_index
        )
        self._extra_context_columns[f'{self._sequence_index}.context'] = {'sdtype': 'numerical'}

        return data

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (dict):
                Mapping of table name to pandas.DataFrame.

        Raises:
            InvalidDataError:
                If a table of the data is not present in the metadata.
        """
        self._data_processor.prepare_for_fitting(data)
        # Ensure that sequence index does not get auto assigned with enforce_min_max_values
        if self._sequence_index:
            sequence_index_transformer = self.get_transformers()[self._sequence_index]
            if sequence_index_transformer and getattr(
                sequence_index_transformer, 'enforce_min_max_values', False
            ):
                sequence_index_transformer.enforce_min_max_values = False

    def _preprocess(self, data):
        """Transform the raw data to numerical space.

        For PAR, none of the sequence keys are transformed.

        Args:
            data (pandas.DataFrame):
                The raw data to be transformed.

        Returns:
            pandas.DataFrame:
                The preprocessed data.
        """
        self._extra_context_columns = {}
        sequence_key_transformers = {sequence_key: None for sequence_key in self._sequence_key}
        if not self._data_processor._prepared_for_fitting:
            self.auto_assign_transformers(data)

        self.update_transformers(sequence_key_transformers)
        preprocessed = super()._preprocess(data)

        if self._sequence_index:
            preprocessed = self._transform_sequence_index(preprocessed)

        return preprocessed

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.

        Raises:
            ValueError:
                Raise when the transformer of a context column is passed.
        """
        if set(column_name_to_transformer).intersection(set(self.context_columns)):
            raise SynthesizerInputError(
                'Transformers for context columns are not allowed to be updated.'
            )

        super().update_transformers(column_name_to_transformer)

    def _fit_context_model(self, transformed):
        LOGGER.debug(f'Fitting context synthesizer {self._context_synthesizer.__class__.__name__}')
        context_metadata: SingleTableMetadata = self._get_context_metadata()
        if self.context_columns or self._extra_context_columns:
            context_cols = (
                self._sequence_key + self.context_columns + list(self._extra_context_columns.keys())
            )
            context = transformed[context_cols]
        else:
            context = transformed[self._sequence_key].copy()
            # Add constant column to allow modeling
            constant_column = str(uuid.uuid4())
            context[constant_column] = 0
            context_metadata.add_column(constant_column, sdtype='numerical')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=".*The 'SingleTableMetadata' is deprecated.*")
            self._context_synthesizer = GaussianCopulaSynthesizer(
                context_metadata,
                enforce_min_max_values=self._context_synthesizer.enforce_min_max_values,
                enforce_rounding=self._context_synthesizer.enforce_rounding,
            )
        context = context.groupby(self._sequence_key).first().reset_index()
        self._context_synthesizer.fit(context)

    def _fit_sequence_columns(self, timeseries_data):
        self._model = PARModel(**self._model_kwargs)

        self._output_columns = list(timeseries_data.columns)
        self._data_columns = [
            column
            for column in timeseries_data.columns
            if column
            not in (
                self._sequence_key + self.context_columns + list(self._extra_context_columns.keys())
            )
        ]

        sequences = assemble_sequences(
            timeseries_data,
            self._sequence_key,
            self.context_columns + list(self._extra_context_columns.keys()),
            self.segment_size,
            self._sequence_index,
            drop_sequence_index=False,
        )
        data_types = []
        context_types = []
        for field in self._output_columns:
            dtype = timeseries_data[field].dtype
            kind = dtype.kind
            if kind in ('i', 'f'):
                data_type = 'continuous'
                sdtype = (
                    self.metadata.tables[self._table_name]
                    .columns.get(field, {})
                    .get('sdtype', None)
                )
                if sdtype == 'categorical':  # Check if metadata overrides this data type
                    transformer = self.get_transformers().get(field)
                    if not transformer or transformer.get_output_sdtypes().get(field) != 'float':
                        data_type = 'categorical'

            elif kind in ('O', 'b'):
                data_type = 'categorical'

            else:
                raise ValueError(f'Unsupported dtype {dtype}')

            if field in self._data_columns:
                data_types.append(data_type)
            elif field in self.context_columns or field in self._extra_context_columns.keys():
                context_types.append(data_type)

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

    def _sample_from_par(self, context, sequence_length=None):
        """Sample new sequences.

        Args:
            context (pandas.DataFrame):
                Context values to use when generating the sequences.
            sequence_length (int):
                Length of each sequence to sample. If not
                given, the sequence length will be sampled from
                the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same
                format as that he training data had.
        """
        # Set the sequence_key as index to properly iterate over them
        if self._sequence_key:
            context = context.set_index(self._sequence_key)
            # reorder context columns
            context_columns = self.context_columns + list(self._extra_context_columns.keys())
            context = context[context_columns]

        should_disable = not self._model_kwargs['verbose']
        iterator = tqdm.tqdm(context.iterrows(), disable=should_disable, total=len(context))

        output = []
        for sequence_key_values, context_values in iterator:
            context_values = context_values.tolist()
            sequence = self._model.sample_sequence(context_values, sequence_length)
            if self._sequence_index:
                sequence_index_idx = self._data_columns.index(self._sequence_index)
                diffs = sequence[sequence_index_idx]
                float_formatter = self.extended_columns[self._sequence_index]
                diffs = float_formatter.reverse_transform(
                    pd.DataFrame({self._sequence_index: diffs})
                )[self._sequence_index].to_numpy()
                start_index = context_columns.index(f'{self._sequence_index}.context')
                start = context_values[start_index]
                sequence[sequence_index_idx] = np.cumsum(diffs) - diffs[0] + start

            # Reformat as a DataFrame
            sequence_df = pd.DataFrame(
                dict(zip(self._data_columns, sequence)), columns=self._data_columns
            )
            sequence_df[self._sequence_key] = sequence_key_values
            context_columns = self.context_columns + list(self._extra_context_columns.keys())
            for column, value in zip(context_columns, context_values):
                sequence_df[column] = value

            output.append(sequence_df)

        output = pd.concat([df.dropna(axis=1, how='all') for df in output])
        output = output[self._output_columns].reset_index(drop=True)

        return output

    def _sample(self, context_columns, sequence_length=None):
        sampled = self._sample_from_par(context_columns, sequence_length)
        sampled = self._data_processor.reverse_transform(sampled)
        return self.reverse_transform_constraints(sampled)

    def sample(self, num_sequences, sequence_length=None):
        """Sample new sequences.

        Args:
            num_sequences (int):
                Number of sequences to sample.
            sequence_length (int):
                If passed, sample sequences of this length. If ``None``, the sequence length will
                be sampled from the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences in the same format as the fitted data.
        """
        if self._sequence_key:
            context_columns = self._context_synthesizer._sample_with_progress_bar(
                num_sequences, output_file_path='disable', show_progress_bar=False
            )

        else:
            context_columns = pd.DataFrame(index=range(num_sequences or 1))

        for column in self._sequence_key or []:
            if column not in context_columns:
                context_columns[column] = range(len(context_columns))

        return self._sample(context_columns, sequence_length)

    def sample_sequential_columns(self, context_columns, sequence_length=None):
        """Sample the sequential columns based ont he provided context columns.

        Args:
            context_columns (pandas.DataFrame):
                Context values to use when generating the sequences.
            sequence_length (int):
                If passed, sample sequences of this length. If ``None``, the sequence length will
                be sampled from the model.

        Returns:
            pandas.DataFrame:
                Table containing the sampled sequences based on the provided context columns.
        """
        if not self.context_columns:
            raise SamplingError(
                "This synthesizer does not have any context columns. Please use 'sample()' "
                'to sample new sequences.'
            )

        condition_columns = list(
            set.intersection(
                set(context_columns.columns), set(self._context_synthesizer._model.columns)
            )
        )
        context_columns = self._process_context_columns(context_columns)

        condition_columns = context_columns[condition_columns].to_dict('records')
        synthesizer_conditions = [Condition(conditions) for conditions in condition_columns]
        context = self._context_synthesizer.sample_from_conditions(synthesizer_conditions)
        context.update(context_columns)
        return self._sample(context, sequence_length)

    def _process_context_columns(self, context_columns):
        """Process context columns by applying appropriate transformations.

        Args:
            context_columns (pandas.DataFrame):
                Context values containing potential columns for transformation.

        Returns:
            context_columns (pandas.DataFrame):
                Updated context columns with transformed values.
        """
        columns_to_be_processed = self._get_context_columns_for_processing()

        if columns_to_be_processed:
            context_columns[columns_to_be_processed] = self._data_processor.transform(
                context_columns[columns_to_be_processed]
            )

        return context_columns
