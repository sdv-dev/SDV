"""Single table data processing."""

import itertools
import json
import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd
import rdt

from sdv.constraints import Constraint
from sdv.constraints.base import get_subclasses
from sdv.constraints.errors import (
    AggregateConstraintsError, FunctionError, MissingConstraintColumnError)
from sdv.data_processing.errors import InvalidConstraintsError, NotFittedError
from sdv.data_processing.numerical_formatter import NumericalFormatter
from sdv.data_processing.utils import load_module_from_path
from sdv.metadata.anonymization import get_anonymized_transformer
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
    """

    _DEFAULT_TRANSFORMERS_BY_SDTYPE = {
        'numerical': rdt.transformers.FloatFormatter(
            learn_rounding_scheme=True,
            enforce_min_max_values=True,
            missing_value_replacement='mean',
            model_missing_values=False,
        ),
        'categorical': rdt.transformers.LabelEncoder(add_noise=True),
        'boolean': rdt.transformers.LabelEncoder(add_noise=True),
        'datetime': rdt.transformers.UnixTimestampEncoder(
            missing_value_replacement='mean',
            model_missing_values=False,
        ),
        'text': rdt.transformers.RegexGenerator()
    }
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
            model_missing_values=False,
            learn_rounding_scheme=enforce_rounding,
            enforce_min_max_values=enforce_min_max_values
        )
        self._transformers_by_sdtype.update({'numerical': custom_float_formatter})

    def __init__(self, metadata, enforce_rounding=True, enforce_min_max_values=True,
                 model_kwargs=None, table_name=None):
        self.metadata = metadata
        self._enforce_rounding = enforce_rounding
        self._enforce_min_max_values = enforce_min_max_values
        self._model_kwargs = model_kwargs or {}
        self._constraints_list = []
        self._constraints = []
        self._constraints_to_reverse = []
        self._custom_constraint_classes = {}
        self._transformers_by_sdtype = self._DEFAULT_TRANSFORMERS_BY_SDTYPE.copy()
        self._update_numerical_transformer(enforce_rounding, enforce_min_max_values)
        self._hyper_transformer = rdt.HyperTransformer()
        self.table_name = table_name
        self._dtypes = None
        self.fitted = False
        self.formatters = {}
        self._anonymized_columns = []
        self._primary_key = self.metadata._primary_key
        self._prepared_for_fitting = False
        self._keys = deepcopy(self.metadata._alternate_keys)
        self._keys_generators = {}
        if self._primary_key:
            self._keys.append(self._primary_key)

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
        for name, column_metadata in self.metadata._columns.items():
            sdtype = column_metadata['sdtype']

            if primary_keys or (name not in self._keys):
                sdtypes[name] = sdtype

        return sdtypes

    def _validate_custom_constraints(self, filepath, class_names):
        errors = []
        reserved_class_names = list(get_subclasses(Constraint))
        module = load_module_from_path(Path(filepath))
        for class_name in class_names:
            if class_name in reserved_class_names:
                errors.append((
                    f"The name '{class_name}' is a reserved constraint name. "
                    'Please use a different one for the custom constraint.'
                ))

            if not hasattr(module, class_name):
                errors.append(f"The constraint '{class_name}' is not defined in '{filepath}'.")

        if errors:
            raise InvalidConstraintsError(errors)

    def load_custom_constraint_classes(self, filepath, class_names):
        """Load a custom constraint class for the current model.

        Args:
            filepath (str):
                String representing the absolute or relative path to the python file where
                the custom constraints are declared.
            class_names (list):
                A list of custom constraint classes to be imported.
        """
        self._validate_custom_constraints(filepath, class_names)
        for class_name in class_names:
            self._custom_constraint_classes[class_name] = filepath

    def _validate_constraint_dict(self, constraint_dict):
        """Validate a constraint against the single table metadata.

        Args:
            constraint_dict (dict):
                A dictionary containing:
                    * ``constraint_class``: Name of the constraint to apply.
                    * ``constraint_parameters``: A dictionary with the constraint parameters.
        """
        constraint_class = constraint_dict['constraint_class']
        constraint_parameters = constraint_dict.get('constraint_parameters', {})
        try:
            if constraint_class in self._custom_constraint_classes:
                path = Path(self._custom_constraint_classes[constraint_class])
                module = load_module_from_path(path)
                constraint_class = getattr(module, constraint_class)

            else:
                constraint_class = Constraint._get_class_from_dict(constraint_class)

        except KeyError:
            raise InvalidConstraintsError(f"Invalid constraint class ('{constraint_class}').")

        constraint_class._validate_metadata(self.metadata, **constraint_parameters)

    def add_constraints(self, constraints):
        """Add constraints to the data processor.

        Args:
            constraints (list):
                List of constraints described as dictionaries in the following format:
                    * ``constraint_class``: Name of the constraint to apply.
                    * ``constraint_parameters``: A dictionary with the constraint parameters.
        """
        errors = []
        validated_constraints = []
        for constraint_dict in constraints:
            constraint_dict = deepcopy(constraint_dict)
            try:
                self._validate_constraint_dict(constraint_dict)
                validated_constraints.append(constraint_dict)
            except (AggregateConstraintsError, InvalidConstraintsError) as e:
                reformated_errors = '\n'.join(map(str, e.errors))
                errors.append(reformated_errors)

        if errors:
            raise InvalidConstraintsError(errors)

        self._constraints_list.extend(validated_constraints)

    def get_constraints(self):
        """Get a list of the current constraints that will be used.

        Returns:
            list:
                List of dictionaries describing the constraints for this data processor.
        """
        return deepcopy(self._constraints_list)

    def _fit_constraints(self, data):
        errors = []
        for constraint in self._constraints:
            try:
                constraint.fit(data)
            except Exception as e:
                errors.append(e)

        if errors:
            raise AggregateConstraintsError(errors)

    def _transform_constraints(self, data, is_condition=False):
        errors = []
        if not is_condition:
            self._constraints_to_reverse = []

        for constraint in self._constraints:
            try:
                data = constraint.transform(data)
                if not is_condition:
                    self._constraints_to_reverse.append(constraint)

            except (MissingConstraintColumnError, FunctionError) as error:
                if isinstance(error, MissingConstraintColumnError):
                    LOGGER.info(
                        f'{constraint.__class__.__name__} cannot be transformed because columns: '
                        f'{error.missing_columns} were not found. Using the reject sampling '
                        'approach instead.'
                    )
                else:
                    LOGGER.info(
                        f'Error transforming {constraint.__class__.__name__}. '
                        'Using the reject sampling approach instead.'
                    )
                if is_condition:
                    indices_to_drop = data.columns.isin(constraint.constraint_columns)
                    columns_to_drop = data.columns.where(indices_to_drop).dropna()
                    data = data.drop(columns_to_drop, axis=1)

            except Exception as error:
                errors.append(error)

        if errors:
            raise AggregateConstraintsError(errors)

        return data

    def _fit_transform_constraints(self, data):
        # Fit and validate all constraints first because `transform` might change columns
        # making the following constraints invalid
        self._fit_constraints(data)
        data = self._transform_constraints(data)

        return data

    def _update_transformers_by_sdtypes(self, sdtype, transformer):
        self._transformers_by_sdtype[sdtype] = transformer

    @staticmethod
    def create_anonymized_transformer(sdtype, column_metadata):
        """Create an instance of an ``AnonymizedFaker``.

        Read the extra keyword arguments from the ``column_metadata`` and use them to create
        an instance of an ``AnonymizedFaker`` transformer.

        Args:
            sdtype (str):
                Sematic data type or a ``Faker`` function name.
            column_metadata (dict):
                A dictionary representing the rest of the metadata for the given ``sdtype``.

        Returns:
            Instance of ``rdt.transformers.pii.AnonymizedFaker``.
        """
        kwargs = {}
        for key, value in column_metadata.items():
            if key not in ['pii', 'sdtype']:
                kwargs[key] = value

        return get_anonymized_transformer(sdtype, kwargs)

    def create_key_transformer(self, column_name, sdtype, column_metadata):
        """Create an instance for the primary key or alternate key transformer.

        Read the keyword arguments from the ``column_metadata`` and use them to create
        an instance of an ``RegexGenerator`` or ``AnonymizedFaker`` transformer with
        ``enforce_uniqueness`` set to ``True``.

        Args:
            sdtype (str):
                Sematic data type or a ``Faker`` function name.
            column_metadata (dict):
                A dictionary representing the rest of the metadata for the given ``sdtype``.

        Returns:
            transformer:
                Instance of ``rdt.transformers.text.RegexGenerator`` or
                ``rdt.transformers.pii.AnonymizedFaker`` with ``enforce_uniqueness`` set to
                ``True``.
        """
        if sdtype == 'numerical':
            self._keys_generators[column_name] = itertools.count()
            return None

        if sdtype == 'text':
            regex_format = column_metadata.get('regex_format', '[A-Za-z]{5}')
            transformer = rdt.transformers.RegexGenerator(
                regex_format=regex_format,
                enforce_uniqueness=True
            )

        else:
            kwargs = deepcopy(column_metadata)
            kwargs['enforce_uniqueness'] = True
            transformer = self.create_anonymized_transformer(sdtype, kwargs)

        return transformer

    def _create_config(self, data, columns_created_by_constraints):
        sdtypes = {}
        transformers = {}
        self._anonymized_columns = []

        for column in set(data.columns) - columns_created_by_constraints:
            column_metadata = self.metadata._columns.get(column)
            sdtype = column_metadata.get('sdtype')
            pii = column_metadata.get('pii', sdtype not in self._DEFAULT_TRANSFORMERS_BY_SDTYPE)
            sdtypes[column] = 'pii' if pii else sdtype

            if column in self._keys:
                transformers[column] = self.create_key_transformer(column, sdtype, column_metadata)

            elif pii:
                transformers[column] = self.create_anonymized_transformer(sdtype, column_metadata)
                self._anonymized_columns.append(column)

            elif sdtype in self._transformers_by_sdtype:
                transformers[column] = deepcopy(self._transformers_by_sdtype[sdtype])

            else:
                sdtypes[column] = 'categorical'
                transformers[column] = deepcopy(self._transformers_by_sdtype['categorical'])

        for column in columns_created_by_constraints:
            dtype_kind = data[column].dtype.kind
            if dtype_kind in ('i', 'f'):
                sdtypes[column] = 'numerical'
                transformers[column] = rdt.transformers.FloatFormatter(
                    missing_value_replacement='mean',
                    model_missing_values=False,
                )
            else:
                sdtype = self._DTYPE_TO_SDTYPE.get(dtype_kind, 'categorical')
                sdtypes[column] = sdtype
                transformers[column] = deepcopy(self._transformers_by_sdtype[sdtype])

        return {'transformers': transformers, 'sdtypes': sdtypes}

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

        self._hyper_transformer.update_transformers(column_name_to_transformer)

    def _fit_hyper_transformer(self, data):
        """Create and return a new ``rdt.HyperTransformer`` instance.

        First get the ``dtypes`` and then use them to build a transformer dictionary
        to be used by the ``HyperTransformer``.

        Args:
            data (pandas.DataFrame):
                Data to transform.
            columns_created_by_constraints (set):
                Names of columns that are not in the metadata but that should also
                be transformed. In most cases, these are the fields that were added
                by previous transformations which the data underwent.

        Returns:
            rdt.HyperTransformer
        """
        if not self._hyper_transformer._fitted:
            if not data.empty:
                self._hyper_transformer.fit(data)

    def _fit_numerical_formatters(self, data):
        """Fit a ``NumericalFormatter`` for each column in the data."""
        self.formatters = {}
        for column_name in data:
            column_metadata = self.metadata._columns.get(column_name)
            if column_metadata.get('sdtype') == 'numerical' and column_name != self._primary_key:
                representation = column_metadata.get('computer_representation', 'Float')
                self.formatters[column_name] = NumericalFormatter(
                    enforce_rounding=self._enforce_rounding,
                    enforce_min_max_values=self._enforce_min_max_values,
                    computer_representation=representation
                )
                self.formatters[column_name].learn_format(data[column_name])

    def prepare_for_fitting(self, data):
        """Prepare the ``DataProcessor`` for fitting.

        This method will learn the ``dtypes`` of the data, fit the numerical formatters,
        fit the constraints and create the configuration for the ``rdt.HyperTransformer``.
        If the ``rdt.HyperTransformer`` has already been updated, this will not perform the
        actions again.

        Args:
            data (pandas.DataFrame):
                Table data to be learnt.
        """
        if not self._prepared_for_fitting:
            LOGGER.info(f'Fitting table {self.table_name} metadata')
            self._dtypes = data[list(data.columns)].dtypes

            LOGGER.info(f'Fitting numerical formatters for table {self.table_name}')
            self._fit_numerical_formatters(data)

            LOGGER.info(f'Fitting constraints for table {self.table_name}')
            constrained = self._fit_transform_constraints(data)
            columns_created_by_constraints = set(constrained.columns) - set(data.columns)

            if not self._hyper_transformer.get_config().get('sdtypes'):
                LOGGER.info((
                    'Setting the configuration for the ``HyperTransformer`` '
                    f'for table {self.table_name}'
                ))
                config = self._create_config(constrained, columns_created_by_constraints)
                self._hyper_transformer.set_config(config)

            self._prepared_for_fitting = True

    def _load_constraints(self):
        loaded_constraints = []
        default_constraints_classes = list(get_subclasses(Constraint))
        for constraint in self._constraints_list:
            if constraint['constraint_class'] in default_constraints_classes:
                loaded_constraints.append(Constraint.from_dict(constraint))

            else:
                constraint_class = constraint['constraint_class']
                path = Path(self._custom_constraint_classes[constraint_class])
                module = load_module_from_path(path)
                constraint_class = getattr(module, constraint_class)
                loaded_constraints.append(
                    constraint_class(**constraint.get('constraint_parameters', {}))
                )

        return loaded_constraints

    def fit(self, data):
        """Fit this metadata to the given data.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
        """
        self._prepared_for_fitting = False
        self._constraints = self._load_constraints()
        self.prepare_for_fitting(data)
        constrained = self._transform_constraints(data)
        LOGGER.info(f'Fitting HyperTransformer for table {self.table_name}')
        self._fit_hyper_transformer(constrained)
        self.fitted = True

    def reset_sampling(self):
        """Reset the sampling state for the anonymized columns and primary keys."""
        # Resetting the transformers manually until fixed on RDT
        for transformer in self._hyper_transformer.field_transformers.values():
            if transformer is not None:
                transformer.reset_randomization()

        self._keys_generators = {
            key: itertools.count()
            for key in self._keys_generators
        }

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
        anonymized_keys = []
        dataframes = {}
        for key in self._keys:
            if self._hyper_transformer.field_transformers.get(key) is None:
                if reset_keys:
                    self._keys_generators[key] = itertools.count()

                dataframes[key] = pd.DataFrame({
                    key: [next(self._keys_generators[key]) for _ in range(num_rows)]
                })

            else:
                anonymized_keys.append(key)

        # Add ``reset_keys`` for RDT once the version is updated.
        if anonymized_keys:
            anonymized_dataframe = self._hyper_transformer.create_anonymized_columns(
                num_rows=num_rows,
                column_names=anonymized_keys,
            )
            if dataframes:
                return pd.concat(list(dataframes.values()) + [anonymized_dataframe], axis=1)

            return anonymized_dataframe

        return pd.concat(dataframes.values(), axis=1)

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

        # Filter columns that can be transformed
        columns = [
            column for column in self.get_sdtypes(primary_keys=not is_condition)
            if column in data.columns
        ]
        LOGGER.debug(f'Transforming constraints for table {self.table_name}')
        data = self._transform_constraints(data[columns], is_condition)

        LOGGER.debug(f'Transforming table {self.table_name}')
        if self._keys and not is_condition:
            keys_to_drop = []
            for key in self._keys:
                if key == self._primary_key:
                    drop_primary_key = bool(self._keys_generators.get(key))
                    data = data.set_index(self._primary_key, drop=drop_primary_key)

                elif self._keys_generators.get(key):
                    keys_to_drop.append(key)

            if keys_to_drop:
                data = data.drop(keys_to_drop, axis=1)

        try:
            transformed = self._hyper_transformer.transform_subset(data)
        except (rdt.errors.NotFittedError, rdt.errors.ConfigNotSetError):
            transformed = data

        return transformed

    def reverse_transform(self, data, reset_keys=False):
        """Reverse the transformed data to the original format.

        Args:
            data (pandas.DataFrame):
                Data to be reverse transformed.
            reset_keys (bool):
                Whether or not to reset the keys generators. Defaults to ``False``.

        Returns:
            pandas.DataFrame
        """
        if not self.fitted:
            raise NotFittedError()

        reversible_columns = [
            column
            for column in self._hyper_transformer._output_columns
            if column in data.columns
        ]

        reversed_data = data
        try:
            if not data.empty:
                reversed_data = self._hyper_transformer.reverse_transform_subset(
                    data[reversible_columns]
                )
        except rdt.errors.NotFittedError:
            LOGGER.info(f'HyperTransformer has not been fitted for table {self.table_name}')

        for constraint in reversed(self._constraints_to_reverse):
            reversed_data = constraint.reverse_transform(reversed_data)

        num_rows = len(reversed_data)
        sampled_columns = list(reversed_data.columns)
        if self._anonymized_columns:
            anonymized_data = self._hyper_transformer.create_anonymized_columns(
                num_rows=num_rows,
                column_names=self._anonymized_columns,
            )
            sampled_columns.extend(self._anonymized_columns)

        if self._keys:
            generated_keys = self.generate_keys(num_rows, reset_keys)
            sampled_columns.extend(self._keys)

        # Sort the sampled columns in the order of the metadata
        # In multitable there may be missing columns in the sample such as foreign keys
        # And alternate keys. Thats the reason of ensuring that the metadata column is within
        # The sampled columns.
        sampled_columns = [
            column for column in self.metadata._columns.keys()
            if column in sampled_columns
        ]
        for column_name in sampled_columns:
            if column_name in self._anonymized_columns:
                column_data = anonymized_data[column_name]
            elif column_name in self._keys:
                column_data = generated_keys[column_name]
            else:
                column_data = reversed_data[column_name]

            dtype = self._dtypes[column_name]
            if pd.api.types.is_integer_dtype(dtype) and column_data.dtype != 'O':
                column_data = column_data.round()

            reversed_data[column_name] = column_data[column_data.notna()].astype(dtype)

        # reformat numerical columns using the NumericalFormatter
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
            'constraints_list': deepcopy(self._constraints_list),
            'constraints_to_reverse': constraints_to_reverse,
            'model_kwargs': deepcopy(self._model_kwargs)
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
            metadata=SingleTableMetadata._load_from_dict(metadata_dict['metadata']),
            enforce_rounding=enforce_rounding,
            enforce_min_max_values=enforce_min_max_values,
            model_kwargs=metadata_dict.get('model_kwargs')
        )

        instance._constraints_to_reverse = [
            Constraint.from_dict(cnt) for cnt in metadata_dict.get('constraints_to_reverse', [])
        ]
        instance._constraints_list = metadata_dict.get('constraints_list', [])

        return instance

    def to_json(self, path):
        """Dump this DataProcessor into a JSON file.

        Args:
            path (str):
                Path of the JSON file where this metadata will be stored.
        """
        with open(path, 'w') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

    @classmethod
    def from_json(cls, path):
        """Load a DataProcessor from a JSON.

        Args:
            path (str):
                Path of the JSON file to load
        """
        with open(path, 'r') as in_file:
            return cls.from_dict(json.load(in_file))
