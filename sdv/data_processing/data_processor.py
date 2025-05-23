"""Single table data processing."""

import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd
import rdt
from pandas.api.types import is_float_dtype, is_integer_dtype
from pandas.errors import IntCastingNaNError
from rdt.transformers import AnonymizedFaker, get_default_transformers
from rdt.transformers.pii.anonymization import get_anonymized_transformer

from sdv._utils import _get_transformer_init_kwargs
from sdv.constraints import Constraint
from sdv.constraints.base import get_subclasses
from sdv.data_processing.datetime_formatter import DatetimeFormatter
from sdv.data_processing.errors import InvalidConstraintsError, NotFittedError
from sdv.data_processing.numerical_formatter import NumericalFormatter
from sdv.data_processing.utils import load_module_from_path
from sdv.errors import SynthesizerInputError
from sdv.metadata.single_table import SingleTableMetadata

LOGGER = logging.getLogger(__name__)


class DataProcessor:
    """Single table data processor.

    This class handles all pre and post processing that is done to a single table to get it ready
    for modeling and finalize sampling. These processes include formatting, transformations,
    anonymization and constraint handling.

    Args:
        metadata (metadata.SingleTableMetadata):
            The single table metadata instance that will be used to apply constraints and
            transformations to the data.
        enforce_rounding (bool):
            Define rounding scheme for FloatFormatter. If True, the data returned by
            reverse_transform will be rounded to that place. Defaults to True.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by reverse_transform of the numerical
            transformer, FloatFormatter, to the min and max values seen during fit.
            Defaults to True.
        model_kwargs (dict):
            Dictionary specifying the kwargs that need to be used in each tabular
            model when working on this table. This dictionary contains as keys the name of the
            TabularModel class and as values a dictionary containing the keyword arguments to use.
            This argument exists mostly to ensure that the models are fitted using the same
            arguments when the same DataProcessor is used to fit different model instances on
            different slices of the same table.
        table_name (str):
            Name of table this processor is for. Optional.
        locales (str or list):
            Default locales to use for AnonymizedFaker transformers. Defaults to ['en_US'].
    """

    _DTYPE_TO_SDTYPE = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }

    def _update_numerical_transformer(self, enforce_rounding, enforce_min_max_values):
        custom_float_formatter = rdt.transformers.FloatFormatter(
            missing_value_replacement='mean',
            missing_value_generation='random',
            learn_rounding_scheme=enforce_rounding,
            enforce_min_max_values=enforce_min_max_values,
        )
        self._transformers_by_sdtype.update({'numerical': custom_float_formatter})

    def _detect_multi_column_transformers(self):
        """Detect if there are any multi column transformers in the metadata.

        Returns:
            dict:
                A dictionary mapping column names to the multi column transformer.
        """
        result = {}
        if self.metadata.column_relationships:
            for relationship in self.metadata._valid_column_relationships:
                column_names = tuple(relationship['column_names'])
                relationship_type = relationship['type']
                if relationship_type in self._COLUMN_RELATIONSHIP_TO_TRANSFORMER:
                    transformer_name = self._COLUMN_RELATIONSHIP_TO_TRANSFORMER[relationship_type]
                    module = getattr(rdt.transformers, relationship_type)
                    transformer_class = getattr(module, transformer_name)
                    try:
                        transformer_instance = transformer_class(locales=self._locales)
                    except TypeError:  # If the transformer doesn't accept locales
                        transformer_instance = transformer_class()

                    result[column_names] = transformer_instance

        return result

    def __init__(
        self,
        metadata,
        enforce_rounding=True,
        enforce_min_max_values=True,
        model_kwargs=None,
        table_name=None,
        locales=['en_US'],
    ):
        self.metadata = metadata
        self._enforce_rounding = enforce_rounding
        self._enforce_min_max_values = enforce_min_max_values
        self._model_kwargs = model_kwargs or {}
        self._locales = locales
        self._constraints_list = []
        self._constraints = []
        self._constraints_to_reverse = []
        self._custom_constraint_classes = {}
        self._COLUMN_RELATIONSHIP_TO_TRANSFORMER = {
            'address': 'RandomLocationGenerator',
            'gps': 'GPSNoiser',
        }

        self._transformers_by_sdtype = deepcopy(get_default_transformers())
        self._transformers_by_sdtype['id'] = rdt.transformers.RegexGenerator()
        del self._transformers_by_sdtype['text']
        self.grouped_columns_to_transformers = self._detect_multi_column_transformers()

        self._update_numerical_transformer(enforce_rounding, enforce_min_max_values)
        self._hyper_transformer = rdt.HyperTransformer()
        self.table_name = table_name or ''
        self._dtypes = None
        self.fitted = False
        self.formatters = {}
        self._primary_key = self.metadata.primary_key
        self._warned_overflow = False
        self._prepared_for_fitting = False
        self._keys = deepcopy(self.metadata.alternate_keys)
        if self._primary_key:
            self._keys.append(self._primary_key)

    def _get_grouped_columns(self):
        """Get the columns that are part of a multi column transformer.

        Returns:
            list:
                A list of columns that are part of a multi column transformer.
        """
        return [col for col_tuple in self.grouped_columns_to_transformers for col in col_tuple]

    def get_model_kwargs(self, model_name):
        """Return the required model kwargs for the indicated model.

        Args:
            model_name (str):
                Qualified Name of the model for which model kwargs
                are needed.

        Returns:
            dict:
                Keyword arguments to use on the indicated model.
        """
        return deepcopy(self._model_kwargs.get(model_name))

    def set_model_kwargs(self, model_name, model_kwargs):
        """Set the model kwargs used for the indicated model.

        Args:
            model_name (str):
                Qualified Name of the model for which the kwargs will be set.
            model_kwargs (dict):
                The key word arguments for the model.
        """
        self._model_kwargs[model_name] = model_kwargs

    def get_sdtypes(self, primary_keys=False):
        """Get a ``dict`` with the ``sdtypes`` for each column of the table.

        Args:
            primary_keys (bool):
                Whether or not to include the primary key fields. Defaults to ``False``.

        Returns:
            dict:
                Dictionary that contains the column names and ``sdtypes``.
        """
        sdtypes = {}
        for name, column_metadata in self.metadata.columns.items():
            sdtype = column_metadata['sdtype']

            if primary_keys or (name not in self._keys):
                sdtypes[name] = sdtype

        return sdtypes

    def _validate_custom_constraint_name(self, class_name):
        reserved_class_names = list(get_subclasses(Constraint))
        if class_name in reserved_class_names:
            error_message = (
                f"The name '{class_name}' is a reserved constraint name. "
                'Please use a different one for the custom constraint.'
            )
            raise InvalidConstraintsError(error_message)

    def _validate_custom_constraints(self, filepath, class_names, module):
        errors = []
        for class_name in class_names:
            try:
                self._validate_custom_constraint_name(class_name)
            except InvalidConstraintsError as err:
                errors += err.errors

            if not hasattr(module, class_name):
                errors.append(f"The constraint '{class_name}' is not defined in '{filepath}'.")

        if errors:
            raise InvalidConstraintsError(errors)

    def load_custom_constraint_classes(self, filepath, class_names):
        """Load a custom constraint class for the current synthesizer.

        Args:
            filepath (str):
                String representing the absolute or relative path to the python file where
                the custom constraints are declared.
            class_names (list):
                A list of custom constraint classes to be imported.
        """
        path = Path(filepath)
        module = load_module_from_path(path)
        self._validate_custom_constraints(filepath, class_names, module)
        for class_name in class_names:
            constraint_class = getattr(module, class_name)
            self._custom_constraint_classes[class_name] = constraint_class

    def _load_constraints(self):
        loaded_constraints = []
        default_constraints_classes = list(get_subclasses(Constraint))
        for constraint in self._constraints_list:
            if constraint['constraint_class'] in default_constraints_classes:
                loaded_constraints.append(Constraint.from_dict(constraint))

            else:
                constraint_class = self._custom_constraint_classes[constraint['constraint_class']]
                loaded_constraints.append(
                    constraint_class(**constraint.get('constraint_parameters', {}))
                )

        return loaded_constraints

    def _update_transformers_by_sdtypes(self, sdtype, transformer):
        self._transformers_by_sdtype[sdtype] = transformer

    @staticmethod
    def create_anonymized_transformer(sdtype, column_metadata, cardinality_rule, locales=['en_US']):
        """Create an instance of an ``AnonymizedFaker``.

        Read the extra keyword arguments from the ``column_metadata`` and use them to create
        an instance of an ``AnonymizedFaker`` transformer.

        Args:
            sdtype (str):
                Sematic data type or a ``Faker`` function name.
            column_metadata (dict):
                A dictionary representing the rest of the metadata for the given ``sdtype``.
            cardinality_rule (str):
                If ``'unique'`` enforce that every created value is unique.
                If ``'match'`` match the cardinality of the data seen during fit.
                If ``None`` do not consider cardinality.
                Defaults to ``None``.
            locales (str or list):
                Locale or list of locales to use for the AnonymizedFaker transfomer.
                Defaults to ['en_US'].

        Returns:
            Instance of ``rdt.transformers.pii.AnonymizedFaker``.
        """
        kwargs = {'locales': locales, 'cardinality_rule': cardinality_rule}
        for key, value in column_metadata.items():
            if key not in ['pii', 'sdtype']:
                kwargs[key] = value

        try:
            transformer = get_anonymized_transformer(sdtype, kwargs)
        except AttributeError as error:
            raise SynthesizerInputError(
                f"The sdtype '{sdtype}' is not compatible with any of the locales. To "
                "continue, try changing the locales or adding 'en_US' as a possible option."
            ) from error

        return transformer

    def create_regex_generator(self, column_name, sdtype, column_metadata, is_numeric):
        """Create a ``RegexGenerator`` for the ``id`` columns.

        Read the keyword arguments from the ``column_metadata`` and use them to create
        an instance of a ``RegexGenerator``. If ``regex_format`` is not present in the
        metadata a default ``[0-1a-z]{5}`` will be used for object like data and an increasing
        integer from ``0`` will be used for numerical data. Also if the column name is a primary
        key or alternate key this will enforce the values to be unique.

        Args:
            column_name (str):
                Name of the column.
            sdtype (str):
                Sematic data type or a ``Faker`` function name.
            column_metadata (dict):
                A dictionary representing the rest of the metadata for the given ``sdtype``.
            is_numeric (boolean):
                A boolean representing whether or not data type is numeric or not.

        Returns:
            transformer:
                Instance of ``rdt.transformers.text.RegexGenerator`` or
                ``rdt.transformers.pii.AnonymizedFaker`` with ``enforce_uniqueness`` set to
                ``True``.
        """
        default_regex_format = r'\d{30}' if is_numeric else '[0-1a-z]{5}'
        regex_format = column_metadata.get('regex_format', default_regex_format)
        transformer = rdt.transformers.RegexGenerator(
            regex_format=regex_format,
            cardinality_rule='unique',
            generation_order='scrambled',
        )

        return transformer

    def _get_transformer_instance(self, sdtype, column_metadata):
        transformer = self._transformers_by_sdtype[sdtype]
        if isinstance(transformer, AnonymizedFaker):
            is_lexify = transformer.function_name == 'lexify'
            is_baseprovider = transformer.provider_name == 'BaseProvider'
            if is_lexify and is_baseprovider:  # Default settings
                return self.create_anonymized_transformer(
                    sdtype, column_metadata, None, self._locales
                )

        kwargs = {
            key: value for key, value in column_metadata.items() if key not in ['pii', 'sdtype']
        }
        if sdtype == 'datetime':
            kwargs['enforce_min_max_values'] = self._enforce_min_max_values

        if kwargs and transformer is not None:
            transformer_class = transformer.__class__
            default_transformer_kwargs = _get_transformer_init_kwargs(transformer)
            return transformer_class(**{**default_transformer_kwargs, **kwargs})

        return deepcopy(transformer)

    def _create_config(self, data):
        sdtypes = {}
        transformers = {}

        columns_in_multi_col_transformer = self._get_grouped_columns()
        for column in set(data.columns):
            column_metadata = self.metadata.columns.get(column)
            sdtype = column_metadata.get('sdtype')

            if column in columns_in_multi_col_transformer:
                sdtypes[column] = sdtype
                continue

            pii = (
                sdtype not in self._transformers_by_sdtype
                and sdtype not in {'unknown', 'id'}
                and (column_metadata.get('pii', True))
            )

            if sdtype == 'id':
                function_name = 'bothify'
                column_dtype = data[column].dtype
                is_numeric = pd.api.types.is_numeric_dtype(column_dtype)
                if column_metadata.get('regex_format', False):
                    transformers[column] = self.create_regex_generator(
                        column, sdtype, column_metadata, is_numeric
                    )
                    sdtypes[column] = 'text'
                elif column in self._keys:
                    if is_numeric:
                        function_name = 'random_int'
                        column_dtype = str(column_dtype).lower()
                        function_kwargs = {'min': 0, 'max': 16777216}
                        if 'int8' in column_dtype:
                            function_kwargs['max'] = 127
                        elif 'int16' in column_dtype:
                            function_kwargs['max'] = 32767

                    else:
                        function_kwargs = {'text': 'sdv-id-??????'}

                    cardinality_rule = None
                    if column in self._keys:
                        cardinality_rule = 'unique'

                    transformers[column] = AnonymizedFaker(
                        provider_name=None,
                        function_name=function_name,
                        function_kwargs=function_kwargs,
                        cardinality_rule=cardinality_rule,
                    )

                    sdtypes[column] = 'pii' if column_metadata.get('pii') else 'text'
                else:
                    transformers[column] = self._get_transformer_instance(
                        'categorical', column_metadata
                    )
                    sdtypes[column] = 'id'

            elif sdtype == 'unknown':
                sdtypes[column] = 'pii'
                function_name = 'bothify'
                function_kwargs = {
                    'text': 'sdv-pii-?????',
                    'letters': '0123456789abcdefghijklmnopqrstuvwxyz',
                }
                if pd.api.types.is_numeric_dtype(data[column]):
                    max_digits = len(str(abs(max(data[column]))))
                    min_digits = len(str(abs(min(data[column]))))
                    text = ('!' * (max_digits - min_digits)) + '%' + ('#' * (min_digits - 1))
                    function_name = 'numerify'
                    function_kwargs = {
                        'text': text,
                    }
                transformers[column] = AnonymizedFaker(
                    function_name=function_name,
                )
                transformers[column].function_kwargs = function_kwargs

            elif pii:
                sdtypes[column] = 'pii'
                cardinality_rule = 'unique' if bool(column in self._keys) else None
                transformers[column] = self.create_anonymized_transformer(
                    sdtype, column_metadata, cardinality_rule, self._locales
                )

            elif sdtype in self._transformers_by_sdtype:
                sdtypes[column] = sdtype
                if column != self._primary_key:
                    transformers[column] = self._get_transformer_instance(sdtype, column_metadata)
                else:
                    transformers[column] = self.create_anonymized_transformer(
                        sdtype=sdtype,
                        column_metadata=column_metadata,
                        cardinality_rule='unique',
                        locales=self._locales,
                    )

            else:
                sdtypes[column] = 'categorical'
                transformers[column] = self._get_transformer_instance(
                    'categorical', column_metadata
                )

        for columns, transformer in self.grouped_columns_to_transformers.items():
            transformers[columns] = transformer

        config = {'transformers': transformers, 'sdtypes': sdtypes}
        return config

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        if self._hyper_transformer.field_transformers == {}:
            raise NotFittedError(
                'The DataProcessor must be prepared for fitting before the transformers can be '
                'updated.'
            )

        for column, transformer in column_name_to_transformer.items():
            if column in self._keys and not transformer.is_generator():
                raise SynthesizerInputError(
                    f"Invalid transformer '{transformer.__class__.__name__}' for a primary "
                    f"or alternate key '{column}'. Please use a generator transformer instead."
                )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='rdt.hyper_transformer')
            self._hyper_transformer.update_transformers(column_name_to_transformer)

        self.grouped_columns_to_transformers = {
            col_tuple: transformer
            for col_tuple, transformer in self._hyper_transformer.field_transformers.items()
            if isinstance(col_tuple, tuple)
        }

    def _fit_hyper_transformer(self, data):
        """Create and return a new ``rdt.HyperTransformer`` instance.

        First get the ``dtypes`` and then use them to build a transformer dictionary
        to be used by the ``HyperTransformer``.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            rdt.HyperTransformer
        """
        self._hyper_transformer.fit(data)

    def _fit_formatters(self, data):
        """Fit ``NumericalFormatter`` and ``DatetimeFormatter`` for each column in the data."""
        for column_name in data:
            column_metadata = self.metadata.columns.get(column_name)
            sdtype = column_metadata.get('sdtype')
            if sdtype == 'numerical' and column_name != self._primary_key:
                representation = column_metadata.get('computer_representation', 'Float')
                self.formatters[column_name] = NumericalFormatter(
                    enforce_rounding=self._enforce_rounding,
                    enforce_min_max_values=self._enforce_min_max_values,
                    computer_representation=representation,
                )
                self.formatters[column_name].learn_format(data[column_name])

            elif sdtype == 'datetime' and column_name != self._primary_key:
                datetime_format = column_metadata.get('datetime_format')
                self.formatters[column_name] = DatetimeFormatter(datetime_format=datetime_format)
                self.formatters[column_name].learn_format(data[column_name])

    def prepare_for_fitting(self, data):
        """Prepare the ``DataProcessor`` for fitting.

        This method will learn the ``dtypes`` of the data, fit the numerical formatters,
        fit and create the configuration for the ``rdt.HyperTransformer``.
        If the ``rdt.HyperTransformer`` has already been updated, this will not perform the
        actions again.

        Args:
            data (pandas.DataFrame):
                Table data to be learnt.
        """
        if not self._prepared_for_fitting:
            LOGGER.info(f'Fitting table {self.table_name} metadata')
            self._dtypes = data[list(data.columns)].dtypes

            self.formatters = {}
            LOGGER.info(f'Fitting formatters for table {self.table_name}')
            self._fit_formatters(data)

            config = self._hyper_transformer.get_config()
            if not config.get('sdtypes'):
                LOGGER.info(
                    (
                        'Setting the configuration for the ``HyperTransformer`` '
                        f'for table {self.table_name}'
                    )
                )
                config = self._create_config(data)
                self._hyper_transformer.set_config(config)

            self._prepared_for_fitting = True

    def fit(self, data):
        """Fit this metadata to the given data.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
        """
        if data.empty:
            raise ValueError('The fit dataframe is empty, synthesizer will not be fitted.')
        self._prepared_for_fitting = False
        self.prepare_for_fitting(data)
        LOGGER.info(f'Fitting HyperTransformer for table {self.table_name}')
        self._fit_hyper_transformer(data)
        self.fitted = True

    def reset_sampling(self):
        """Reset the sampling state for the anonymized columns and primary keys."""
        self._hyper_transformer.reset_randomization()

    def generate_keys(self, num_rows, reset_keys=False):
        """Generate the columns that are identified as ``keys``.

        Args:
            num_rows (int):
                Number of rows to be created. Must be an integer greater than 0.
            reset_keys (bool):
                Whether or not to reset the keys generators. Defaults to ``False``.

        Returns:
            pandas.DataFrame:
                A dataframe with the newly generated primary keys of the size ``num_rows``.
        """
        generated_keys = self._hyper_transformer.create_anonymized_columns(
            num_rows=num_rows,
            column_names=self._keys,
        )
        return generated_keys

    def transform(self, data, is_condition=False):
        """Transform the given data.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        data = data.copy()
        if not self.fitted:
            raise NotFittedError()

        columns = [
            column
            for column in self.get_sdtypes(primary_keys=not is_condition)
            if column in data.columns
        ]
        data = data[columns]
        LOGGER.debug(f'Transforming table {self.table_name}')
        if self._keys and not is_condition:
            data = data.set_index(self._primary_key, drop=False)

        try:
            transformed = self._hyper_transformer.transform_subset(data)
        except (rdt.errors.NotFittedError, rdt.errors.ConfigNotSetError):
            transformed = data

        return transformed

    def reverse_transform(self, data, reset_keys=False, conditions=None):
        """Reverse the transformed data to the original format.

        Args:
            data (pandas.DataFrame):
                Data to be reverse transformed.
            reset_keys (bool):
                Whether or not to reset the keys generators. Defaults to ``False``.
            conditions (dict, optional):
                Dictionary of conditional values to use for generated columns.

        Returns:
            pandas.DataFrame
        """
        if not self.fitted:
            raise NotFittedError()

        reversible_columns = [
            column for column in self._hyper_transformer._output_columns if column in data.columns
        ]

        reversed_data = data
        try:
            if not data.empty:
                reversed_data = self._hyper_transformer.reverse_transform_subset(
                    data[reversible_columns]
                )
        except rdt.errors.NotFittedError:
            LOGGER.info(f'HyperTransformer has not been fitted for table {self.table_name}')

        for transformer in self.grouped_columns_to_transformers.values():
            if not transformer.output_columns:
                reversed_data = transformer.reverse_transform(reversed_data)

        num_rows = len(reversed_data)
        sampled_columns = list(reversed_data.columns)
        missing_columns = [
            column
            for column in self.metadata.columns.keys() - set(sampled_columns + self._keys)
            if self._hyper_transformer.field_transformers.get(column)
        ]
        if missing_columns and num_rows:
            missing_conditions = {}
            if conditions:
                missing_conditions = {
                    col: condition
                    for col, condition in conditions.items()
                    if col in missing_columns
                }
                missing_columns = [col for col in missing_columns if col not in conditions]

            if missing_columns:
                anonymized_data = self._hyper_transformer.create_anonymized_columns(
                    num_rows=num_rows, column_names=missing_columns
                )
                sampled_columns.extend(missing_columns)
                reversed_data[anonymized_data.columns] = anonymized_data[anonymized_data.notna()]

            conditional_data = pd.DataFrame(missing_conditions, index=reversed_data.index)
            sampled_columns.extend(list(missing_conditions.keys()))
            reversed_data[conditional_data.columns] = conditional_data

        if self._keys and num_rows:
            generated_keys = self.generate_keys(num_rows, reset_keys)
            sampled_columns.extend(self._keys)
            reversed_data[generated_keys.columns] = generated_keys[generated_keys.notna()]

        for constraint in reversed(self._constraints_to_reverse):
            reversed_data = constraint.reverse_transform(reversed_data)

        # Add new columns generated by the constraint
        new_columns = list(set(reversed_data.columns) - set(sampled_columns))
        sampled_columns.extend(new_columns)

        # Sort the sampled columns in the order of the metadata.
        # Any extra columns not present in the metadata will be dropped.
        # In multitable there may be missing columns in the sample such as foreign keys
        # And alternate keys. Thats the reason of ensuring that the metadata column is within
        # The sampled columns.
        sampled_columns = [
            column for column in self.metadata.columns.keys() if column in sampled_columns
        ]
        for column_name in sampled_columns:
            column_data = reversed_data[column_name]

            dtype = self._dtypes[column_name]
            if is_integer_dtype(dtype) and is_float_dtype(column_data.dtype):
                column_data = column_data.round()

            reversed_data[column_name] = column_data[column_data.notna()]
            try:
                reversed_data[column_name] = reversed_data[column_name].astype(dtype)
            except (IntCastingNaNError, ValueError) as e:
                message = (
                    f"The real data in '{column_name}' was stored as '{dtype}' but the "
                    'synthetic data could not be cast back to this type. If this is a '
                    'problem, please check your input data and metadata settings.'
                )
                if isinstance(e, IntCastingNaNError):
                    LOGGER.debug(message)
                    continue

                # Handle the ValueError case
                column_metadata = self.metadata.columns.get(column_name)
                sdtype = column_metadata.get('sdtype')
                if sdtype not in self._DTYPE_TO_SDTYPE.values():
                    LOGGER.info(message)
                    if column_name in self.formatters:
                        self.formatters.pop(column_name)
                else:
                    raise ValueError(e)
            except OverflowError:
                if not self._warned_overflow:
                    warnings.warn(
                        f"The real data in '{self.table_name}' and column '{column_name}' was "
                        f"stored as '{dtype}' but the synthetic data overflowed when casting back "
                        'to this type. If this is a problem, please check your input data '
                        'and metadata settings.'
                    )
                self._warned_overflow = True

        # reformat columns using the formatters
        for column in sampled_columns:
            if column in self.formatters:
                data_to_format = reversed_data[column]
                reversed_data[column] = self.formatters[column].format_data(data_to_format)

        return reversed_data[sampled_columns]

    def filter_valid(self, data):
        """Filter the data using the constraints and return only the valid rows.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Table containing only the valid rows.
        """
        for constraint in self._constraints:
            data = constraint.filter_valid(data)

        return data

    def to_dict(self):
        """Get a dict representation of this DataProcessor.

        Returns:
            dict:
                Dict representation of this DataProcessor.
        """
        constraints_to_reverse = [cnt.to_dict() for cnt in self._constraints_to_reverse]
        return {
            'metadata': deepcopy(self.metadata.to_dict()),
            'constraints_list': self._constraints,
            'constraints_to_reverse': constraints_to_reverse,
            'model_kwargs': deepcopy(self._model_kwargs),
        }

    @classmethod
    def from_dict(cls, metadata_dict, enforce_rounding=True, enforce_min_max_values=True):
        """Load a DataProcessor from a metadata dict.

        Args:
            metadata_dict (dict):
                Dict metadata to load.
            enforce_rounding (bool):
                If passed, set the ``enforce_rounding`` on the new instance.
            enforce_min_max_values (bool):
                If passed, set the ``enforce_min_max_values`` on the new instance.
        """
        instance = cls(
            metadata=SingleTableMetadata.load_from_dict(metadata_dict['metadata']),
            enforce_rounding=enforce_rounding,
            enforce_min_max_values=enforce_min_max_values,
            model_kwargs=metadata_dict.get('model_kwargs'),
        )

        instance._constraints_to_reverse = [
            Constraint.from_dict(cnt) for cnt in metadata_dict.get('constraints_to_reverse', [])
        ]
        instance._constraints_list = metadata_dict.get('constraints_list', [])

        return instance

    def to_json(self, filepath):
        """Dump this DataProcessor into a JSON file.

        Args:
            filepath (str):
                Path of the JSON file where this metadata will be stored.
        """
        with open(filepath, 'w') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

    @classmethod
    def from_json(cls, filepath):
        """Load a DataProcessor from a JSON.

        Args:
            filepath (str):
                Path of the JSON file to load
        """
        with open(filepath, 'r') as in_file:
            return cls.from_dict(json.load(in_file))
