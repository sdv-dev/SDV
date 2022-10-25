"""Base Synthesizer class."""

import functools
import inspect
import logging
import math
import os
import uuid
import warnings
from collections import defaultdict

import copulas
import numpy as np
import pandas as pd
import tqdm

from sdv.data_processing.data_processor import DataProcessor
from sdv.errors import ConstraintsNotMetError, InvalidPreprocessingError
from sdv.single_table.errors import InvalidDataError
from sdv.single_table.utils import check_num_rows, handle_sampling_error, validate_file_path
from sdv.utils import is_boolean_type, is_datetime_type, is_numerical_type

LOGGER = logging.getLogger(__name__)
COND_IDX = str(uuid.uuid4())
FIXED_RNG_SEED = 73251
TMP_FILE_NAME = '.sample.csv.temp'
DISABLE_TMP_FILE = 'disable'


class BaseSynthesizer:
    """Base class for all ``Synthesizers``.

    The ``BaseSynthesizer`` class defines the common API that all the
    ``Synthesizers`` need to implement, as well as common functionality.

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
    """

    _model_sdtype_transformers = None

    def _update_default_transformers(self):
        if self._model_sdtype_transformers is not None:
            for sdtype, transformer in self._model_sdtype_transformers.items():
                self._data_processor._update_transformers_by_sdtypes(sdtype, transformer)

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True):
        self.metadata = metadata
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self._data_processor = DataProcessor(metadata)
        self._update_default_transformers()
        self._fitted = False

    def _validate_metadata_matches_data(self, columns):
        errors = []
        metadata_columns = self.metadata._columns or []
        missing_data_columns = set(columns).difference(metadata_columns)
        if missing_data_columns:
            errors.append(
                f'The columns {sorted(missing_data_columns)} are not present in the metadata.')

        missing_metadata_columns = set(metadata_columns).difference(columns)
        if missing_metadata_columns:
            errors.append(
                f'The metadata columns {sorted(missing_metadata_columns)} '
                'are not present in the data.'
            )

        if errors:
            raise InvalidDataError(errors)

    def _get_primary_and_alternate_keys(self):
        keys = set(self.metadata._alternate_keys)
        if self.metadata._primary_key:
            keys.update({self.metadata._primary_key})

        return keys

    def _get_set_of_sequence_keys(self):
        if isinstance(self.metadata._sequence_key, tuple):
            return set(self.metadata._sequence_key)

        if isinstance(self.metadata._sequence_key, str):
            return {self.metadata._sequence_key}

        return set()

    def _validate_keys_dont_have_missing_values(self, data):
        errors = []
        keys = self._get_primary_and_alternate_keys()
        keys.update(self._get_set_of_sequence_keys())
        for key in sorted(keys):
            if pd.isna(data[key]).any():
                errors.append(f"Key column '{key}' contains missing values.")

        return errors

    @staticmethod
    def _format_invalid_values_string(invalid_values):
        invalid_values = sorted(invalid_values, key=lambda x: str(x))
        if len(invalid_values) > 3:
            return invalid_values[:3] + [f'+ {len(invalid_values) - 3} more']

        return invalid_values

    def _validate_key_values_are_unique(self, data):
        errors = []
        keys = self._get_primary_and_alternate_keys()
        for key in sorted(keys):
            repeated_values = set(data[key][data[key].duplicated()])
            if repeated_values:
                repeated_values = self._format_invalid_values_string(repeated_values)
                errors.append(f"Key column '{key}' contains repeating values: {repeated_values}")

        return errors

    def _validate_context_columns(self, data):
        # NOTE: move method to PARSynthesizer when it has been implemented
        errors = []
        context_column_names = self._data_processor._model_kwargs.get('context_columns')
        if context_column_names:
            sequence_key_names = sorted(self._get_set_of_sequence_keys())
            for sequence_key, group in data.groupby(sequence_key_names):
                if len(group.groupby(context_column_names)) != 1:
                    invalid_context_cols = {}
                    for context_column_name in context_column_names:
                        values = set(group[context_column_name])
                        if len(values) != 1:
                            invalid_context_cols[context_column_name] = values

                    errors.append(
                        f'Context column(s) {invalid_context_cols} are changing '
                        f'inside the sequence keys ({sequence_key_names}: {sequence_key}).'
                    )

        return errors

    def _validate_sdtype(self, sdtype, column, validation):
        valid = column.apply(validation)
        invalid_values = set(column[~valid])
        if invalid_values:
            invalid_values = self._format_invalid_values_string(invalid_values)
            return [f"Invalid values found for {sdtype} column '{column.name}': {invalid_values}."]

        return []

    def _validate_column(self, column):
        """Validate values of the column satisfy its sdtype properties."""
        errors = []
        sdtype = self.metadata._columns[column.name]['sdtype']

        # boolean values must be True/False, None or missing values
        # int/str are not allowed
        if sdtype == 'boolean':
            errors += self._validate_sdtype(sdtype, column, is_boolean_type)

        # numerical values must be int/float, None or missing values
        # str/bool are not allowed
        if sdtype == 'numerical':
            errors += self._validate_sdtype(sdtype, column, is_numerical_type)

        # datetime values must be castable to datetime, None or missing values
        if sdtype == 'datetime':
            errors += self._validate_sdtype(
                sdtype, column, lambda x: pd.isna(x) | is_datetime_type(x))

        return errors

    def validate(self, data):
        """Validate data.

        Args:
            data (pd.DataFrame):
                The data to validate.

        Raises:
            ValueError:
                Raised when data is not of type pd.DataFrame.
            InvalidDataError:
                Raised if:
                    * data columns don't match metadata
                    * keys have missing values
                    * primary or alternate keys are not unique
                    * context columns vary for a sequence key
                    * values of a column don't satisfy their sdtype
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be a DataFrame, not a {type(data)}.')

        # Both metadata and data must have the same set of columns
        self._validate_metadata_matches_data(data.columns)

        errors = []
        # Primary, sequence and alternate keys can't have missing values
        errors += self._validate_keys_dont_have_missing_values(data)

        # Primary and alternate key values must be unique
        errors += self._validate_key_values_are_unique(data)

        # Context column values must be the same for each tuple of sequence keys
        errors += self._validate_context_columns(data)

        # Every column must satisfy the properties of their sdtypes
        for column in data:
            errors += self._validate_column(data[column])

        if errors:
            raise InvalidDataError(errors)

    def _validate_transformers(self, column_name_to_transformer):
        keys = self._get_primary_and_alternate_keys() | self._get_set_of_sequence_keys()
        for column, transformer in column_name_to_transformer.items():
            if column in keys and not transformer.is_generator():
                raise InvalidPreprocessingError(
                    f"Column '{column}' is a key. It cannot be preprocessed using "
                    f"the '{type(transformer).__name__}' transformer."
                )

            # If columns were set, the transformer was fitted
            if transformer.columns:
                raise InvalidPreprocessingError(
                    f"Transformer for column '{column}' has already been fit on data.")

    def _warn_for_update_transformers(self, column_name_to_transformer):
        """Raise warnings for update_transformers.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        for column in column_name_to_transformer:
            sdtype = self.metadata._columns[column]['sdtype']
            if sdtype in {'categorical', 'boolean'}:
                warnings.warn(
                    f"Replacing the default transformer for column '{column}' "
                    'might impact the quality of your synthetic data.'
                )

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        self._validate_transformers(column_name_to_transformer)
        self._warn_for_update_transformers(column_name_to_transformer)
        self._data_processor.update_transformers(column_name_to_transformer)

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer."""
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        return instantiated_parameters

    def get_metadata(self):
        """Return the ``SingleTableMetadata`` for this synthesizer."""
        return self.metadata

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (pandas.DataFrame):
                The raw data (before any transformations) that will be used to fit the model.
        """
        constrained = self._data_processor._fit_transform_constraints(data)
        columns_created_by_constraints = set(constrained.columns) - set(data.columns)
        self._data_processor._fit_numerical_formatters(data)
        config = self._data_processor._create_config(constrained, columns_created_by_constraints)
        self._data_processor._hyper_transformer.set_config(config)

    def get_transformers(self):
        """Get a dictionary mapping of ``column_name``  and ``rdt.transformers``.

        A dictionary representing the column names and the transformers that will be used
        to transform the data.

        Returns:
            dict:
                A dictionary mapping with column names and transformers.
        """
        field_transformers = {}
        if self._data_processor._hyper_transformer is not None:
            field_transformers = self._data_processor._hyper_transformer.field_transformers

        if field_transformers == {}:
            raise ValueError(
                "No transformers were returned in 'get_transformers'. "
                "Use 'auto_assign_transformers' or 'fit' to create them."
            )

        return field_transformers

    def preprocess(self, data):
        """Transform the raw data to numerical space."""
        self.validate(data)
        if self._fitted:
            warnings.warn(
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )

        self._data_processor.fit(data)
        return self._data_processor.transform(data)

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        raise NotImplementedError()

    def fit_processed_data(self, processed_data):
        """Fit this model to the transformed data.

        Args:
            processed_data (pandas.DataFrame):
                The transformed data used to fit the model to.
        """
        # Reset fit status
        self._fitted = False
        self._fit(processed_data)
        self._fitted = True

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (pandas.DataFrame):
                The raw data (before any transformations) to fit the model to.
        """
        processed_data = self.preprocess(data)
        self.fit_processed_data(processed_data)

    def _set_random_state(self, random_state):
        """Set the random state of the model's random number generator.

        Args:
            random_state (int, tuple[np.random.RandomState, torch.Generator], or None):
                Seed or tuple of random states to use.
        """
        self._model.set_random_state(random_state)

    @staticmethod
    def _filter_conditions(sampled, conditions, float_rtol):
        """Filter the sampled rows that match the conditions.

        If condition columns are float values, consider a match anything that
        is closer than the given ``float_rtol`` and then make the value exact.

        Args:
            sampled (pandas.DataFrame):
                The sampled rows, reverse transformed.
            conditions (dict):
                The dictionary of conditioning values.
            float_rtol (float):
                Maximum tolerance when considering a float match.

        Returns:
            pandas.DataFrame:
                Rows from the sampled data that match the conditions.
        """
        for column, value in conditions.items():
            column_values = sampled[column]
            if column_values.dtype.kind == 'f':
                distance = value * float_rtol
                sampled = sampled[np.abs(column_values - value) <= distance]
                sampled[column] = value
            else:
                sampled = sampled[column_values == value]

        return sampled

    def _sample_rows(self, num_rows, conditions=None, transformed_conditions=None,
                     float_rtol=0.1, previous_rows=None):
        """Sample rows with the given conditions.

        Input conditions is taken both in the raw input format, which will be used
        for filtering during the reject-sampling loop, and already transformed
        to the model format, which will be passed down to the model if it supports
        conditional sampling natively.

        If condition columns are float values, consider a match anything that
        is closer than the given ``float_rtol`` and then make the value exact.

        If the model does not have any data columns, the result of this call
        is a dataframe of the requested length with no columns in it.

        If there are no columns other than the ``primary_key``, this will proceed to sample
        only the ``primary_key`` using the ``DataProcessor``.

        Args:
            num_rows (int):
                Number of rows to sample.
            conditions (dict):
                The dictionary of conditioning values in the original format.
            transformed_conditions (dict):
                The dictionary of conditioning values transformed to the model format.
            float_rtol (float):
                Maximum tolerance when considering a float match.
            previous_rows (pandas.DataFrame):
                Valid rows sampled in the previous iterations.

        Returns:
            tuple:
                * pandas.DataFrame:
                    Rows from the sampled data that match the conditions.
                * int:
                    Number of rows that are considered valid.
        """
        if self._data_processor.get_sdtypes(primary_keys=False):
            if conditions is None:
                sampled = self._sample(num_rows)
            else:
                try:
                    sampled = self._sample(num_rows, transformed_conditions)
                except NotImplementedError:
                    sampled = self._sample(num_rows)

            sampled = self._data_processor.reverse_transform(sampled)

            if previous_rows is not None:
                sampled = pd.concat([previous_rows, sampled], ignore_index=True)

            sampled = self._data_processor.filter_valid(sampled)

            if conditions is not None:
                sampled = self._filter_conditions(sampled, conditions, float_rtol)

            num_valid = len(sampled)

            return sampled, num_valid

        else:
            sampled = pd.DataFrame(index=range(num_rows))
            sampled = self._data_processor.reverse_transform(sampled)
            return sampled, num_rows

    def _sample_batch(self, batch_size, max_tries=100,
                      conditions=None, transformed_conditions=None, float_rtol=0.01,
                      progress_bar=None, output_file_path=None):
        """Sample a batch of rows with the given conditions.

        This will enter a reject-sampling loop in which rows will be sampled until
        all of them are valid and match the requested conditions. If ``max_tries``
        is exceeded, it will return as many rows as it has sampled, which may be less
        than the target number of rows.

        Input conditions is taken both in the raw input format, which will be used
        for filtering during the reject-sampling loop, and already transformed
        to the model format, which will be passed down to the model if it supports
        conditional sampling natively.

        If condition columns are float values, consider a match anything that is
        relatively closer than the given ``float_rtol`` and then make the value exact.

        If the model does not have any data columns, the result of this call
        is a dataframe of the requested length with no columns in it.

        Args:
            batch_size (int):
                Number of rows to sample for this batch. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
            max_tries (int):
                Number of times to retry sampling until the batch size is met.
                Defaults to 100.
            conditions (dict):
                The dictionary of conditioning values in the original input format.
            transformed_conditions (dict):
                The dictionary of conditioning values transformed to the model format.
            float_rtol (float):
                Maximum tolerance when considering a float match.
            progress_bar (tqdm.tqdm or None):
                The progress bar to update when sampling. If None, a new tqdm progress
                bar will be created.
            output_file_path (str or None):
                The file to periodically write sampled rows to. If None, does not write
                rows anywhere.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        num_rows_to_sample = batch_size

        counter = 0
        num_valid = 0
        prev_num_valid = None
        remaining = batch_size
        sampled = pd.DataFrame()

        while num_valid < batch_size and counter < max_tries:
            prev_num_valid = num_valid
            sampled, num_valid = self._sample_rows(
                num_rows_to_sample,
                conditions,
                transformed_conditions,
                float_rtol,
                sampled
            )

            num_new_valid_rows = num_valid - prev_num_valid
            num_increase = min(num_new_valid_rows, remaining)
            num_sampled = min(len(sampled), batch_size)
            if num_increase > 0:
                if output_file_path:
                    append_kwargs = {'mode': 'a', 'header': False}
                    append_kwargs = append_kwargs if os.path.getsize(output_file_path) > 0 else {}
                    sampled.head(num_sampled).tail(num_increase).to_csv(
                        output_file_path,
                        index=False,
                        **append_kwargs,
                    )

                if progress_bar is not None:
                    progress_bar.update(num_increase)

            remaining = batch_size - num_valid
            valid_rate = max(num_new_valid_rows, 1) / max(num_rows_to_sample, 1)
            num_rows_to_sample = min(10 * batch_size, int(remaining / valid_rate))

            if remaining > 0:
                LOGGER.info(
                    f'{remaining} valid rows remaining. Resampling {num_rows_to_sample} rows')

            counter += 1

        return sampled.head(min(len(sampled), batch_size))

    @staticmethod
    def _make_condition_dfs(conditions):
        """Transform ``conditions`` into a list of dataframes.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of ``sdv.sampling.Condition``, where each ``Condition`` object
                represents a desired column value mapping and the number of rows
                to generate for that condition.

        Returns:
            list[pandas.DataFrame]:
                A list of ``conditions`` as dataframes.
        """
        condition_dataframes = defaultdict(list)
        for condition in conditions:
            column_values = condition.get_column_values()
            condition_dataframes[tuple(column_values.keys())].append(
                pd.DataFrame(column_values, index=range(condition.get_num_rows())))

        return [
            pd.concat(condition_list, ignore_index=True)
            for condition_list in condition_dataframes.values()
        ]

    def _sample_in_batches(self, num_rows, batch_size, max_tries_per_batch, conditions=None,
                           transformed_conditions=None, float_rtol=0.01, progress_bar=None,
                           output_file_path=None):
        sampled = []
        batch_size = batch_size if num_rows > batch_size else num_rows
        for step in range(math.ceil(num_rows / batch_size)):
            sampled_rows = self._sample_batch(
                batch_size=batch_size,
                max_tries=max_tries_per_batch,
                conditions=conditions,
                transformed_conditions=transformed_conditions,
                float_rtol=float_rtol,
                progress_bar=progress_bar,
                output_file_path=output_file_path,
            )
            sampled.append(sampled_rows)

        sampled = pd.concat(sampled, ignore_index=True) if len(sampled) > 0 else pd.DataFrame()
        return sampled.head(num_rows)

    def _conditionally_sample_rows(self, dataframe, condition, transformed_condition,
                                   max_tries_per_batch=None, batch_size=None, float_rtol=0.01,
                                   graceful_reject_sampling=True, progress_bar=None,
                                   output_file_path=None):
        batch_size = batch_size or len(dataframe)
        sampled_rows = self._sample_in_batches(
            num_rows=len(dataframe),
            batch_size=batch_size,
            max_tries_per_batch=max_tries_per_batch,
            conditions=condition,
            transformed_conditions=transformed_condition,
            float_rtol=float_rtol,
            progress_bar=progress_bar,
            output_file_path=output_file_path
        )

        if len(sampled_rows) > 0:
            sampled_rows[COND_IDX] = dataframe[COND_IDX].to_numpy()[:len(sampled_rows)]

        elif not graceful_reject_sampling:
            user_msg = (
                'Unable to sample any rows for the given conditions '
                f"'{transformed_condition}'. "
            )
            if hasattr(self, '_model') and isinstance(
                    self._model, copulas.multivariate.GaussianMultivariate):
                user_msg = user_msg + (
                    'This may be because the provided values are out-of-bounds in the '
                    'current model. \nPlease try again with a different set of values.'
                )
            else:
                user_msg = user_msg + (
                    f"Try increasing 'max_tries_per_batch' (currently: {max_tries_per_batch}) "
                    f"or increasing 'batch_size' (currently: {batch_size}). Note that "
                    'increasing these values will also increase the sampling time.'
                )

            raise ValueError(user_msg)

        return sampled_rows

    def _randomize_samples(self, randomize_samples):
        """Randomize the samples according to user input.

        If ``randomize_samples`` is false, fix the seed that the random number generator
        uses in the underlying models.

        Args:
            randomize_samples (bool):
                Whether or not to randomize the generated samples.
        """
        if self._model is None:
            return

        if randomize_samples:
            self._set_random_state(None)
        else:
            self._set_random_state(FIXED_RNG_SEED)

    def _sample_with_progress_bar(self, num_rows, randomize_samples=True, max_tries_per_batch=100,
                                  batch_size=None, output_file_path=None, conditions=None,
                                  show_progress_bar=True):
        if conditions is not None:
            raise TypeError('This method does not support the conditions parameter. '
                            "Please create 'sdv.sampling.Condition' objects and pass them "
                            "into the 'sample_conditions' method. "
                            'See User Guide or API for more details.')

        if num_rows is None:
            raise ValueError('You must specify the number of rows to sample (e.g. num_rows=100).')

        sampled = pd.DataFrame()
        if num_rows == 0:
            return sampled

        self._randomize_samples(randomize_samples)
        output_file_path = validate_file_path(output_file_path)
        batch_size = min(batch_size, num_rows) if batch_size else num_rows

        try:
            with tqdm.tqdm(total=num_rows, disable=not show_progress_bar) as progress_bar:
                progress_bar.set_description('Sampling rows')
                sampled = self._sample_in_batches(
                    num_rows=num_rows,
                    batch_size=batch_size,
                    max_tries_per_batch=max_tries_per_batch,
                    progress_bar=progress_bar,
                    output_file_path=output_file_path
                )

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path == TMP_FILE_NAME, output_file_path, error)

        else:
            if output_file_path == TMP_FILE_NAME and os.path.exists(output_file_path):
                os.remove(output_file_path)

        return sampled

    def sample(self, num_rows, randomize_samples=True, max_tries_per_batch=100, batch_size=None,
               output_file_path=None, conditions=None):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. This parameter is required.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int or None):
                The batch size to sample. Defaults to ``num_rows``, if None.
            output_file_path (str or None):
                The file to periodically write sampled rows to. If None, does not
                write rows anywhere.
            conditions:
                Deprecated argument. Use the ``sample_conditions`` method with
                ``sdv.sampling.Condition`` objects instead.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        has_constraints = bool(self.get_metadata()._constraints)
        has_batches = batch_size is not None and batch_size != num_rows
        show_progress_bar = has_constraints or has_batches

        return self._sample_with_progress_bar(
            num_rows,
            randomize_samples,
            max_tries_per_batch,
            batch_size,
            output_file_path,
            conditions,
            show_progress_bar=show_progress_bar
        )

    def _validate_conditions(self, conditions):
        """Validate the user-passed conditions."""
        for column in conditions.columns:
            if column not in self._data_processor.get_fields():
                raise ValueError(f"Unexpected column name '{column}'. "
                                 f'Use a column name that was present in the original data.')

    def _sample_with_conditions(self, conditions, max_tries_per_batch, batch_size,
                                progress_bar=None, output_file_path=None):
        """Sample rows with conditions.

        Args:
            conditions (pandas.DataFrame):
                A DataFrame representing the conditions to be sampled.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int):
                The batch size to use for each sampling call.
            progress_bar (tqdm.tqdm or None):
                The progress bar to update.
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
        condition_columns = list(conditions.columns)
        conditions.index.name = COND_IDX
        conditions = conditions.reset_index()
        grouped_conditions = conditions.groupby(condition_columns)

        # sample
        all_sampled_rows = []

        for group, dataframe in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            condition = dict(zip(condition_columns, group))
            condition_df = dataframe.iloc[0].to_frame().T
            try:
                transformed_condition = self._data_processor.transform(
                    condition_df,
                    is_condition=True
                )
            except ConstraintsNotMetError as error:
                raise ConstraintsNotMetError(
                    'Provided conditions are not valid for the given constraints.'
                ) from error

            transformed_conditions = pd.concat(
                [transformed_condition] * len(dataframe),
                ignore_index=True
            )
            transformed_columns = list(transformed_conditions.columns)
            if not transformed_conditions.empty:
                transformed_conditions.index = dataframe.index
                transformed_conditions[COND_IDX] = dataframe[COND_IDX]

            if len(transformed_columns) == 0:
                sampled_rows = self._conditionally_sample_rows(
                    dataframe=dataframe,
                    condition=condition,
                    transformed_condition=None,
                    max_tries_per_batch=max_tries_per_batch,
                    batch_size=batch_size,
                    progress_bar=progress_bar,
                    output_file_path=output_file_path,
                )
                all_sampled_rows.append(sampled_rows)
            else:
                transformed_groups = transformed_conditions.groupby(transformed_columns)
                for transformed_group, transformed_dataframe in transformed_groups:
                    if not isinstance(transformed_group, tuple):
                        transformed_group = [transformed_group]

                    transformed_condition = dict(zip(transformed_columns, transformed_group))
                    sampled_rows = self._conditionally_sample_rows(
                        dataframe=transformed_dataframe,
                        condition=condition,
                        transformed_condition=transformed_condition,
                        max_tries_per_batch=max_tries_per_batch,
                        batch_size=batch_size,
                        progress_bar=progress_bar,
                        output_file_path=output_file_path,
                    )
                    all_sampled_rows.append(sampled_rows)

        all_sampled_rows = pd.concat(all_sampled_rows)
        if len(all_sampled_rows) == 0:
            return all_sampled_rows

        all_sampled_rows = all_sampled_rows.set_index(COND_IDX)
        all_sampled_rows.index.name = conditions.index.name
        all_sampled_rows = all_sampled_rows.sort_index()

        return all_sampled_rows

    def _sample_conditions(self, conditions, max_tries_per_batch, batch_size, randomize_samples,
                           output_file_path):
        """Sample rows from this table with the given conditions."""
        output_file_path = validate_file_path(output_file_path)

        num_rows = functools.reduce(
            lambda num_rows, condition: condition.get_num_rows() + num_rows, conditions, 0)

        conditions = self._make_condition_dfs(conditions)
        for condition_dataframe in conditions:
            self._validate_conditions(condition_dataframe)

        self._randomize_samples(randomize_samples)

        sampled = pd.DataFrame()
        try:
            with tqdm.tqdm(total=num_rows) as progress_bar:
                progress_bar.set_description('Sampling conditions')
                for condition_dataframe in conditions:
                    sampled_for_condition = self._sample_with_conditions(
                        condition_dataframe,
                        max_tries_per_batch,
                        batch_size,
                        progress_bar,
                        output_file_path,
                    )
                    sampled = pd.concat([sampled, sampled_for_condition], ignore_index=True)

            is_reject_sampling = (hasattr(self, '_model') and not isinstance(
                self._model, copulas.multivariate.GaussianMultivariate))
            check_num_rows(
                num_rows=len(sampled),
                expected_num_rows=num_rows,
                is_reject_sampling=is_reject_sampling,
                max_tries_per_batch=max_tries_per_batch
            )

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path == TMP_FILE_NAME, output_file_path, error)

        else:
            if output_file_path == TMP_FILE_NAME and os.path.exists(output_file_path):
                os.remove(output_file_path)

        return sampled

    def sample_conditions(self, conditions, max_tries_per_batch=100, batch_size=None,
                          randomize_samples=True, output_file_path=None):
        """Sample rows from this table with the given conditions.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of sdv.sampling.Condition objects, which specify the column
                values in a condition, along with the number of rows for that
                condition.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int):
                The batch size to use per sampling call.
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
            conditions, max_tries_per_batch, batch_size, randomize_samples, output_file_path)

    def _sample_remaining_columns(self, known_columns, max_tries_per_batch, batch_size,
                                  randomize_samples, output_file_path):
        """Sample the remaining columns of a given DataFrame."""
        output_file_path = validate_file_path(output_file_path)

        self._randomize_samples(randomize_samples)

        known_columns = known_columns.copy()
        self._validate_conditions(known_columns)
        sampled = pd.DataFrame()
        try:
            with tqdm.tqdm(total=len(known_columns)) as progress_bar:
                progress_bar.set_description('Sampling remaining columns')
                sampled = self._sample_with_conditions(
                    known_columns, max_tries_per_batch, batch_size, progress_bar, output_file_path)

            is_reject_sampling = (hasattr(self, '_model') and not isinstance(
                self._model, copulas.multivariate.GaussianMultivariate))

            check_num_rows(
                num_rows=len(sampled),
                expected_num_rows=len(known_columns),
                is_reject_sampling=is_reject_sampling,
                max_tries_per_batch=max_tries_per_batch
            )

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path == TMP_FILE_NAME, output_file_path, error)

        else:
            if output_file_path == TMP_FILE_NAME and os.path.exists(output_file_path):
                os.remove(output_file_path)

        return sampled

    def sample_remaining_columns(self, known_columns, max_tries_per_batch=100, batch_size=None,
                                 randomize_samples=True, output_file_path=None):
        """Sample remaining rows from already known columns.

        Args:
            known_columns (pandas.DataFrame):
                A pandas.DataFrame with the columns that are already known. The output
                is a DataFrame such that each row in the output is sampled
                conditionally on the corresponding row in the input.
            max_tries_per_batch (int):
                Number of times to retry sampling until the batch size is met. Defaults to 100.
            batch_size (int):
                The batch size to use per sampling call.
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
            known_columns,
            max_tries_per_batch,
            batch_size,
            randomize_samples,
            output_file_path
        )
