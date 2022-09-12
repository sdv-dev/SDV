"""Single table data processing."""

import json
import logging
from copy import deepcopy

import pandas as pd
import rdt
import numpy as np

from sdv.constraints import Constraint
from sdv.constraints.errors import (
    AggregateConstraintsError, FunctionError, MissingConstraintColumnError)
from sdv.data_processing.errors import NotFittedError
from sdv.metadata.single_table import SingleTableMetadata

LOGGER = logging.getLogger(__name__)
MAX_DECIMALS = sys.float_info.dig - 1
INTEGER_BOUNDS = {
    'Int8': (-2**7, 2**7 - 1),
    'Int16': (-2**15, 2**15 - 1),
    'Int32': (-2**31, 2**31 - 1),
    'Int64': (-2**63, 2**63 - 1),
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
}

class NumericalFormatter:
    """?"""

    _dtype = None
    _min_value = None
    _max_value = None
    _rounding_digits = None

    def __init__(self, enforce_min_max_values, learn_rounding_scheme, computer_representation):
        self.enforce_min_max_values = enforce_min_max_values,
        self.learn_rounding_scheme = learn_rounding_scheme
        self.computer_representation = computer_representation

    @staticmethod
    def _learn_rounding_digits(data):
        # check if data has any decimals
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]
        if ((roundable_data % 1) != 0).any():
            if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
                for decimal in range(MAX_DECIMALS + 1):
                    if (roundable_data == roundable_data.round(decimal)).all():
                        return decimal

        return None

    def learn_format(self, column):
        """Learn the format of a column.

        Args:
            column (pandas.Series):
                Data to learn the format.
        """
        self._dtype = column.dtype
        if self.enforce_min_max_values:
            self._min_value = column.min()
            self._max_value = column.max()

        if self.learn_rounding_scheme:
            self._rounding_digits = self._learn_rounding_digits(column)
    

    def format_data(self, column):
        """Format a column according to the learned format.

        Args:
            column (numpy.ndarray):
                Data to format.

        Returns:
            numpy.ndarray containing the formatted data.
        """
        if self.enforce_min_max_values:
            column = column.clip(self._min_value, self._max_value)
        elif self.computer_representation != 'Float':
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            column = column.clip(min_bound, max_bound)

        is_integer = np.dtype(self._dtype).kind == 'i'
        if self.learn_rounding_scheme or is_integer:
            column = column.round(self._rounding_digits or 0)

        if pd.isna(column).any() and is_integer:
            return column

        return column.astype(self._dtype)


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
        'boolean': rdt.transformers.LabelEncoder(add_noise=True),
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
        self._model_kwargs = model_kwargs or {}
        self._constraints = self._load_constraints()
        self._constraints_to_reverse = []
        self._transformers_by_sdtype = self._DEFAULT_TRANSFORMERS_BY_SDTYPE.copy()
        self._update_numerical_transformer(learn_rounding_scheme, enforce_min_max_values)
        self._hyper_transformer = None
        self.table_name = table_name
        self._dtypes = None
        self.fitted = False

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

    def _create_config(self, data, columns_created_by_constraints):
        sdtypes = {}
        transformers = {}
        for column in set(data.columns) - columns_created_by_constraints:
            column_metadata = self.metadata._columns.get(column)
            sdtype = column_metadata.get('sdtype')
            sdtypes[column] = sdtype
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
        self._hyper_transformer = rdt.HyperTransformer()
        config = self._create_config(data, columns_created_by_constraints)
        self._hyper_transformer.set_config(config)

        if not data.empty:
            self._hyper_transformer.fit(data)

    def fit(self, data):
        """Fit this metadata to the given data.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
        """
        LOGGER.info(f'Fitting table {self.table_name} metadata')
        self._dtypes = data[list(data.columns)].dtypes

        LOGGER.info(f'Fitting constraints for table {self.table_name}')
        constrained = self._fit_transform_constraints(data)
        columns_created_by_constraints = set(constrained.columns) - set(data.columns)

        LOGGER.info(f'Fitting HyperTransformer for table {self.table_name}')
        self._fit_hyper_transformer(constrained, columns_created_by_constraints)
        self.fitted = True

    def transform(self, data, is_condition=False):
        """Transform the given data.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        if not self.fitted:
            raise NotFittedError()

        LOGGER.debug(f'Transforming constraints for table {self.table_name}')
        data = self._transform_constraints(data, is_condition)

        LOGGER.debug(f'Transforming table {self.table_name}')
        try:
            return self._hyper_transformer.transform_subset(data)
        except (rdt.errors.NotFittedError, rdt.errors.Error):
            return data

    def reverse_transform(self, data):
        """Reverse the transformed data to the original format.

        Args:
            data (pandas.DataFrame):
                Data to be reverse transformed.

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

        original_columns = list(self.metadata._columns.keys())
        for column_name, _ in self.metadata._columns.items():
            column_data = reversed_data[column_name]
            if pd.api.types.is_integer_dtype(self._dtypes[column_name]):
                column_data = column_data.round()

            dtype = self._dtypes[column_name]
            reversed_data[column_name] = column_data[column_data.notna()].astype(dtype)

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
