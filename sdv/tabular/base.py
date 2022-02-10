"""Base Class for tabular models."""

import functools
import logging
import math
import pickle
import uuid
from collections import defaultdict
from warnings import warn

import numpy as np
import pandas as pd
from tqdm import tqdm

from sdv.errors import ConstraintsNotMetError
from sdv.metadata import Table

LOGGER = logging.getLogger(__name__)
COND_IDX = str(uuid.uuid4())
FIXED_SEED = 21321


def validate_sample_args(function):
    """Validate the args and kwargs passed to sample.

    Set more informative error messages for specific unsupported arguments.

    Args:
        function (Callable):
            The function to wrap around.
    """
    def wrapper(*args, **kwargs):
        if 'condition' in kwargs or 'conditions' in kwargs:
            raise TypeError('This method does not support the conditions parameter. '
                            'Please create `sdv.sampling.Condition` objects and pass them '
                            'into the `sample_conditions` method. '
                            'See User Guide or API for more details.')

        return function(*args, **kwargs)

    return wrapper


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

    _DTYPE_TRANSFORMERS = None

    _metadata = None

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 rounding='auto', min_value='auto', max_value='auto'):
        if table_metadata is None:
            self._metadata = Table(
                field_names=field_names,
                primary_key=primary_key,
                field_types=field_types,
                field_transformers=field_transformers,
                anonymize_fields=anonymize_fields,
                constraints=constraints,
                dtype_transformers=self._DTYPE_TRANSFORMERS,
                rounding=rounding,
                min_value=min_value,
                max_value=max_value
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
        if isinstance(data, pd.DataFrame):
            data = data.reset_index(drop=True)

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
                sampled = sampled[np.abs(column_values - value) < distance]
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
        if self._metadata.get_dtypes(ids=False):
            if conditions is None:
                sampled = self._sample(num_rows)
            else:
                try:
                    sampled = self._sample(num_rows, transformed_conditions)
                except NotImplementedError:
                    sampled = self._sample(num_rows)

            sampled = self._metadata.reverse_transform(sampled)

            if previous_rows is not None:
                sampled = previous_rows.append(sampled, ignore_index=True)

            sampled = self._metadata.filter_valid(sampled)

            if conditions is not None:
                sampled = self._filter_conditions(sampled, conditions, float_rtol)

            num_valid = len(sampled)

            return sampled, num_valid

        else:
            sampled = pd.DataFrame(index=range(num_rows))
            sampled = self._metadata.reverse_transform(sampled)
            return sampled, num_rows

    def _sample_batch(self, num_rows=None, max_tries=100, batch_size_per_try=None,
                      conditions=None, transformed_conditions=None, float_rtol=0.01,
                      progress_bar=None):
        """Sample a batch of rows with the given conditions.

        This will enter a reject-sampling loop in which rows will be sampled until
        all of them are valid and match the requested conditions. If `max_tries`
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
            num_rows (int):
                Number of rows to sample. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
            max_tries (int):
                Number of times to try sampling discarded rows.
                Defaults to 100.
            batch_size_per_try (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            conditions (dict):
                The dictionary of conditioning values in the original input format.
            transformed_conditions (dict):
                The dictionary of conditioning values transformed to the model format.
            float_rtol (float):
                Maximum tolerance when considering a float match.
            progress_bar (tqdm.tqdm or None):
                The progress bar to update when sampling. If None, a new tqdm progress
                bar will be created.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if not batch_size_per_try:
            batch_size_per_try = num_rows * 10

        if not progress_bar:
            progress_bar = tqdm(total=num_rows)

        counter = 0
        num_valid = 0
        prev_num_valid = None
        remaining = num_rows
        sampled = pd.DataFrame()

        while num_valid < num_rows:
            if counter >= max_tries:
                break

            prev_num_valid = num_valid
            sampled, num_valid = self._sample_rows(
                batch_size_per_try, conditions, transformed_conditions, float_rtol, sampled,
            )

            if num_valid > 0:
                progress_bar.update(min(num_valid - prev_num_valid, remaining))

            remaining = num_rows - num_valid
            if remaining > 0:
                LOGGER.info(
                    f'{remaining} valid rows remaining. Resampling {batch_size_per_try} rows')

            counter += 1

        return sampled.head(min(len(sampled), num_rows))

    def _make_condition_dfs(self, conditions):
        """Transform `conditions` into a list of dataframes.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of `sdv.sampling.Condition`, where each `Condition` object
                represents a desired column value mapping and the number of rows
                to generate for that condition.

        Returns:
            list[pandas.DataFrame]:
                A list of `conditions` as dataframes.
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

    def _conditionally_sample_rows(self, dataframe, condition, transformed_condition,
                                   max_tries=None, batch_size_per_try=None, float_rtol=0.01,
                                   graceful_reject_sampling=True, progress_bar=None):
        num_rows = len(dataframe)
        sampled_rows = self._sample_batch(
            num_rows,
            max_tries,
            batch_size_per_try,
            condition,
            transformed_condition,
            float_rtol,
            progress_bar,
        )
        num_sampled_rows = len(sampled_rows)

        if num_sampled_rows < num_rows:
            # Didn't get enough rows.
            if len(sampled_rows) == 0:
                error = 'No valid rows could be generated with the given conditions.'
                raise ValueError(error)

            elif not graceful_reject_sampling:
                error = f'Could not get enough valid rows within {max_tries} trials.'
                raise ValueError(error)

            else:
                warn(f'Only able to sample {len(sampled_rows)} rows for the given '
                     f'conditions. To sample more rows, try increasing `max_tries '
                     f'(currently: {max_tries}) or increasing `batch_size_per_try` '
                     f'(currently: {batch_size_per_try}. Note that increasing these values '
                     f'will also increase the sampling time.')

        if len(sampled_rows) > 0:
            sampled_rows[COND_IDX] = dataframe[COND_IDX].values[:len(sampled_rows)]

        return sampled_rows

    @validate_sample_args
    def sample(self, num_rows, randomize_samples=True, batch_size=None):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. This parameter is required.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            batch_size (int or None):
                The batch size to sample. Defaults to `num_rows`, if None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if num_rows is None:
            raise ValueError('You must specify the number of rows to sample (e.g. num_rows=100).')

        batch_size = min(batch_size, num_rows) if batch_size else num_rows

        sampled = []
        with tqdm(total=num_rows) as progress_bar:
            progress_bar.set_description(
                f'Sampling {num_rows} rows of data in batches of size {batch_size}')
            for step in range(math.ceil(num_rows / batch_size)):
                sampled_rows = self._sample_batch(
                    batch_size, batch_size_per_try=batch_size, progress_bar=progress_bar)
                sampled.append(sampled_rows)

        return pd.concat(sampled, ignore_index=True)

    def _sample_with_conditions(self, conditions, max_tries, batch_size_per_try,
                                progress_bar=None):
        """Sample rows with conditions.

        Args:
            conditions (pandas.DataFrame):
                A DataFrame representing the conditions to be sampled.
            max_tries (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size_per_try (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            progress_bar (tqdm.tqdm or None):
                The progress bar to update.

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
        for column in conditions.columns:
            if column not in self._metadata.get_fields():
                raise ValueError(f'Unexpected column name `{column}`. '
                                 f'Use a column name that was present in the original data.')

        try:
            transformed_conditions = self._metadata.transform(conditions, on_missing_column='drop')
        except ConstraintsNotMetError as cnme:
            cnme.message = 'Provided conditions are not valid for the given constraints'
            raise

        condition_columns = list(conditions.columns)
        transformed_columns = list(transformed_conditions.columns)
        conditions.index.name = COND_IDX
        conditions.reset_index(inplace=True)
        transformed_conditions.index.name = COND_IDX
        transformed_conditions.reset_index(inplace=True)
        grouped_conditions = conditions.groupby(condition_columns)

        # sample
        all_sampled_rows = list()

        for group, dataframe in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            condition_indices = dataframe[COND_IDX]
            condition = dict(zip(condition_columns, group))
            if len(transformed_columns) == 0:
                sampled_rows = self._conditionally_sample_rows(
                    dataframe,
                    condition,
                    None,
                    max_tries,
                    batch_size_per_try,
                    progress_bar=progress_bar,
                )
                all_sampled_rows.append(sampled_rows)
            else:
                transformed_conditions_in_group = transformed_conditions.loc[condition_indices]
                transformed_groups = transformed_conditions_in_group.groupby(transformed_columns)
                for transformed_group, transformed_dataframe in transformed_groups:
                    if not isinstance(transformed_group, tuple):
                        transformed_group = [transformed_group]

                    transformed_condition = dict(zip(transformed_columns, transformed_group))
                    sampled_rows = self._conditionally_sample_rows(
                        transformed_dataframe,
                        condition,
                        transformed_condition,
                        max_tries,
                        batch_size_per_try,
                        progress_bar=progress_bar,
                    )
                    all_sampled_rows.append(sampled_rows)

        all_sampled_rows = pd.concat(all_sampled_rows)
        all_sampled_rows = all_sampled_rows.set_index(COND_IDX)
        all_sampled_rows.index.name = conditions.index.name
        all_sampled_rows = all_sampled_rows.sort_index()
        all_sampled_rows = self._metadata.make_ids_unique(all_sampled_rows)

        return all_sampled_rows

    def sample_conditions(self, conditions, max_tries=100, batch_size_per_try=None,
                          randomize_samples=True):
        """Sample rows from this table with the given conditions.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of sdv.sampling.Condition objects, which specify the column
                values in a condition, along with the number of rows for that
                condition.
            max_tries (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size_per_try (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.

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
        num_rows = functools.reduce(
            lambda num_rows, condition: condition.get_num_rows() + num_rows, conditions, 0)
        conditions = self._make_condition_dfs(conditions)

        with tqdm(total=num_rows) as progress_bar:
            sampled = pd.DataFrame()
            for condition_dataframe in conditions:
                sampled_for_condition = self._sample_with_conditions(
                    condition_dataframe, max_tries, batch_size_per_try, progress_bar)
                sampled = pd.concat([sampled, sampled_for_condition], ignore_index=True)

        return sampled

    def sample_remaining_columns(self, known_columns, max_tries=100, batch_size_per_try=None,
                                 randomize_samples=True):
        """Sample rows from this table.

        Args:
            known_columns (pandas.DataFrame):
                A pandas.DataFrame with the columns that are already known. The output
                is a DataFrame such that each row in the output is sampled
                conditionally on the corresponding row in the input.
            max_tries (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size_per_try (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.

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
        with tqdm(total=len(known_columns)) as progress_bar:
            sampled = self._sample_with_conditions(
                known_columns, max_tries, batch_size_per_try, progress_bar)

        return sampled

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
