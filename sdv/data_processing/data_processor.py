"""Single table data processing."""

import itertools
import json
import logging
from copy import deepcopy

import pandas as pd
import rdt

from sdv.constraints import Constraint
from sdv.constraints.errors import (
    AggregateConstraintsError, FunctionError, MissingConstraintColumnError)
from sdv.data_processing.errors import NotFittedError
from sdv.data_processing.numerical_formatter import NumericalFormatter
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
        learn_rounding_scheme (bool):
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
            model_missing_values=True,
        ),
        'categorical': rdt.transformers.LabelEncoder(add_noise=True),
        'boolean': rdt.transformers.BinaryEncoder(),
        'datetime': rdt.transformers.UnixTimestampEncoder(
            missing_value_replacement='mean',
            model_missing_values=True,
        )
    }
    _DTYPE_TO_SDTYPE = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }

    def _load_constraints(self):
        constraints = self.metadata._constraints or []
        loaded_constraints = [Constraint.from_dict(constraint) for constraint in constraints]
        return loaded_constraints

    def _update_numerical_transformer(self, learn_rounding_scheme, enforce_min_max_values):
        custom_float_formatter = rdt.transformers.FloatFormatter(
            missing_value_replacement='mean',
            model_missing_values=True,
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values
        )
        self._transformers_by_sdtype.update({'numerical': custom_float_formatter})

    def __init__(self, metadata, learn_rounding_scheme=True, enforce_min_max_values=True,
                 model_kwargs=None, table_name=None):
        self.metadata = metadata
        self._learn_rounding_scheme = learn_rounding_scheme
        self._enforce_min_max_values = enforce_min_max_values
        self._model_kwargs = model_kwargs or {}
        self._constraints = self._load_constraints()
        self._constraints_to_reverse = []
        self._transformers_by_sdtype = self._DEFAULT_TRANSFORMERS_BY_SDTYPE.copy()
        self._update_numerical_transformer(learn_rounding_scheme, enforce_min_max_values)
        self._hyper_transformer = None
        self.table_name = table_name
        self._dtypes = None
        self.fitted = False
        self.formatters = {}
        self._anonymized_columns = []
        self._primary_key = None
        self._primary_key_generator = None

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

    def create_primary_key_transformer(self, sdtype, column_metadata):
        """Create an instance for the primary key.

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
            self._primary_key_generator = itertools.count()
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
        self._primary_key = self.metadata._primary_key

        for column in set(data.columns) - columns_created_by_constraints:
            column_metadata = self.metadata._columns.get(column)
            sdtype = column_metadata.get('sdtype')
            sdtypes[column] = 'pii' if column_metadata.get('pii') else sdtype

            if column == self._primary_key:
                transformers[column] = self.create_primary_key_transformer(sdtype, column_metadata)

            elif column_metadata.get('pii'):
                transformers[column] = self.create_anonymized_transformer(sdtype, column_metadata)
                self._anonymized_columns.append(column)

            else:
                transformers[column] = self._transformers_by_sdtype.get(sdtype)

        for column in columns_created_by_constraints:
            dtype_kind = data[column].dtype.kind
            if dtype_kind in ('i', 'f'):
                sdtypes[column] = 'numerical'
                transformers[column] = rdt.transformers.FloatFormatter(
                    missing_value_replacement='mean',
                    model_missing_values=True,
                )
            else:
                sdtype = self._DTYPE_TO_SDTYPE.get(dtype_kind, 'categorical')
                sdtypes[column] = sdtype
                transformers[column] = self._transformers_by_sdtype[sdtype]

        return {'transformers': transformers, 'sdtypes': sdtypes}

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        if not self._hyper_transformer:
            raise NotFittedError(
                'The DataProcessor must be fitted before the transformers can be updated.')

        self._hyper_transformer.update_transformers(column_name_to_transformer)

    def _fit_hyper_transformer(self, data, columns_created_by_constraints):
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
        if self._hyper_transformer is None:
            self._hyper_transformer = rdt.HyperTransformer()

        if not self._hyper_transformer._fitted:
            config = self._create_config(data, columns_created_by_constraints)
            self._hyper_transformer.set_config(config)

            if not data.empty:
                self._hyper_transformer.fit(data)

    def _fit_numerical_formatters(self, data):
        """Fit a ``NumericalFormatter`` for each column in the data."""
        self.formatters = {}
        for column_name in data:
            column_metadata = self.metadata._columns.get(column_name)
            if column_metadata.get('sdtype') == 'numerical':
                representation = column_metadata.get('computer_representation', 'Float')
                self.formatters[column_name] = NumericalFormatter(
                    learn_rounding_scheme=self._learn_rounding_scheme,
                    enforce_min_max_values=self._enforce_min_max_values,
                    computer_representation=representation
                )
                self.formatters[column_name].learn_format(data[column_name])

    def fit(self, data):
        """Fit this metadata to the given data.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
        """
        LOGGER.info(f'Fitting table {self.table_name} metadata')
        self._dtypes = data[list(data.columns)].dtypes

        LOGGER.info(f'Fitting numerical formatters for table {self.table_name}')
        self._fit_numerical_formatters(data)

        LOGGER.info(f'Fitting constraints for table {self.table_name}')
        constrained = self._fit_transform_constraints(data)
        columns_created_by_constraints = set(constrained.columns) - set(data.columns)

        LOGGER.info(f'Fitting HyperTransformer for table {self.table_name}')
        self._fit_hyper_transformer(constrained, columns_created_by_constraints)
        self.fitted = True

    def generate_primary_keys(self, num_rows, reset_primary_key=False):
        """Generate the columns that are identified as ``primary keys``.

        Args:
            num_rows (int):
                Number of rows to be created. Must be an integer greater than 0.
            reset_primary_key (bool):
                Whether or not reset the primary keys generators. Defaults to ``False``.

        Returns:
            pandas.DataFrame:
                A data frame with the newly generated primary keys of the size ``num_rows``.
        """
        if self._hyper_transformer.field_transformers.get(self._primary_key) is None:
            if reset_primary_key:
                self._primary_key_generator = itertools.count()

            return pd.DataFrame({
                self._primary_key: [next(self._primary_key_generator) for _ in range(num_rows)]
            })

        return self._hyper_transformer.create_anonymized_columns(
            num_rows=num_rows,
            column_names=[self._primary_key],
        )

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

        LOGGER.debug(f'Transforming constraints for table {self.table_name}')
        data = self._transform_constraints(data, is_condition)

        LOGGER.debug(f'Transforming table {self.table_name}')
        if self._primary_key and not is_condition:
            # If it's numerical we have to drop it, else it's dropped by the hyper transformer
            drop_primary_key = bool(self._primary_key_generator)
            data = data.set_index(self._primary_key, drop=drop_primary_key)

        try:
            transformed = self._hyper_transformer.transform_subset(data)
        except (rdt.errors.NotFittedError, rdt.errors.Error):
            transformed = data

        return transformed

    def reverse_transform(self, data, reset_primary_key=False):
        """Reverse the transformed data to the original format.

        Args:
            data (pandas.DataFrame):
                Data to be reverse transformed.
            reset_primary_key (bool):
                Whether or not reset the primary keys generators. Defaults to ``False``.

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
        if self._anonymized_columns:
            anonymized_data = self._hyper_transformer.create_anonymized_columns(
                num_rows=num_rows,
                column_names=self._anonymized_columns,
            )
        if self._primary_key:
            primary_keys = self.generate_primary_keys(num_rows, reset_primary_key)

        original_columns = list(self.metadata._columns.keys())
        for column_name in original_columns:
            if column_name in self._anonymized_columns:
                column_data = anonymized_data[column_name]
            elif column_name == self._primary_key:
                column_data = primary_keys[column_name]
            else:
                column_data = reversed_data[column_name]

            dtype = self._dtypes[column_name]
            if pd.api.types.is_integer_dtype(dtype):
                column_data = column_data.round()

            reversed_data[column_name] = column_data[column_data.notna()].astype(dtype)

        # reformat numerical columns using the NumericalFormatter
        for column in original_columns:
            if column in self.formatters:
                data_to_format = reversed_data[column]
                reversed_data[column] = self.formatters[column].format_data(data_to_format)

        return reversed_data[original_columns]

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
            'constraints_to_reverse': constraints_to_reverse,
            'model_kwargs': deepcopy(self._model_kwargs)
        }

    @classmethod
    def from_dict(cls, metadata_dict, learn_rounding_scheme=True, enforce_min_max_values=True):
        """Load a DataProcessor from a metadata dict.

        Args:
            metadata_dict (dict):
                Dict metadata to load.
            learn_rounding_scheme (bool):
                If passed, set the ``learn_rounding_scheme`` on the new instance.
            enforce_min_max_values (bool):
                If passed, set the ``enforce_min_max_values`` on the new instance.
        """
        instance = cls(
            metadata=SingleTableMetadata._load_from_dict(metadata_dict['metadata']),
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values,
            model_kwargs=metadata_dict.get('model_kwargs')
        )
        instance._constraints_to_reverse = [
            Constraint.from_dict(cnt) for cnt in metadata_dict.get('constraints_to_reverse', [])
        ]

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
