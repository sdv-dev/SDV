"""Single Table Metadata."""

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from numpy import isin

import pandas as pd

from sdv.constraints import Constraint
from sdv.metadata.errors import MetadataError
from torch import constant_pad_nd


class SingleTableMetadata:
    """Single Table Metadata class."""

    _EXPECTED_KWARGS = {
        'numerical': frozenset(['representation']),
        'datetime': frozenset(['datetime_format']),
        'categorical': frozenset(['order', 'order_by']),
        'boolean': frozenset([]),
        'text': frozenset(['regex_format']),
    }

    _DTYPES_TO_SDTYPES = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }

    _NUMERICAL_REPRESENTATIONS = frozenset([
        'int', 'int64', 'int32', 'int16', 'int8',
        'uint', 'uint64', 'uint32', 'uint16', 'uint8',
        'float', 'float64', 'float32', 'float16', 'float8',
    ])
    _KEYS = frozenset([
        'columns',
        'primary_key',
        'alternate_keys',
        'constraints',
        'SCHEMA_VERSION'
    ])
    SCHEMA_VERSION = 'SINGLE_TABLE_V1'

    def _validate_numerical(self, column_name, **kwargs):
        representation = kwargs.get('representation')
        if representation and representation not in self._NUMERICAL_REPRESENTATIONS:
            raise ValueError(
                f"Invalid value for 'representation' {representation} for column '{column_name}'.")

    @staticmethod
    def _validate_datetime(column_name, **kwargs):
        datetime_format = kwargs.get('datetime_format')
        if datetime_format:
            try:
                formated_date = datetime.now().strftime(datetime_format)
            except Exception as exception:
                raise ValueError(
                    f"Invalid datetime format string '{datetime_format}' "
                    f"for datetime column '{column_name}'."
                ) from exception

            matches = re.findall('(%.)|(%)', formated_date)
            if matches:
                raise ValueError(
                    f"Invalid datetime format string '{datetime_format}' "
                    f"for datetime column '{column_name}'."
                )

    @staticmethod
    def _validate_categorical(column_name, **kwargs):
        order = kwargs.get('order')
        order_by = kwargs.get('order_by')
        if order and order_by:
            raise ValueError(
                f"Categorical column '{column_name}' has both an 'order' and 'order_by' "
                'attribute. Only 1 is allowed.'
            )
        if order_by and order_by not in ('numerical_value', 'alphabetical'):
            raise ValueError(
                f"Unknown ordering method '{order_by}' provided for categorical column "
                f"'{column_name}'. Ordering method must be 'numerical_value' or 'alphabetical'."
            )
        if (isinstance(order, list) and not len(order)) or\
           (not isinstance(order, list) and order is not None):
            raise ValueError(
                f"Invalid order value provided for categorical column '{column_name}'. "
                "The 'order' must be a list with 1 or more elements."
            )

    @staticmethod
    def _validate_text(column_name, **kwargs):
        regex = kwargs.get('regex_format')
        try:
            re.compile(regex)
        except Exception as exception:
            raise ValueError(
                f"Invalid regex format string '{regex}' for text column '{column_name}'."
            ) from exception

    def __init__(self):
        self._columns = {}
        self._primary_key = None
        self._alternate_keys = []
        self._constraints = []
        self._version = self.SCHEMA_VERSION
        self._metadata = {
            'columns': self._columns,
            'primary_key': self._primary_key,
            'alternate_keys': self._alternate_keys,
            'constraints': self._constraints,
            'SCHEMA_VERSION': self.SCHEMA_VERSION
        }

    def _validate_unexpected_kwargs(self, column_name, sdtype, **kwargs):
        expected_kwargs = self._EXPECTED_KWARGS.get(sdtype, ['pii'])
        unexpected_kwargs = set(list(kwargs)) - set(expected_kwargs)
        if unexpected_kwargs:
            unexpected_kwargs = list(unexpected_kwargs)
            unexpected_kwargs.sort()
            unexpected_kwargs = ', '.join(unexpected_kwargs)
            raise ValueError(
                f"Invalid values '({unexpected_kwargs})' for {sdtype} column '{column_name}'.")

    def _validate_column_exists(self, column_name):
        if column_name not in self._columns:
            raise ValueError(
                f"Column name ('{column_name}') does not exist in the table. "
                "Use 'add_column' to add new column."
            )

    def _validate_column(self, column_name, sdtype, **kwargs):
        self._validate_unexpected_kwargs(column_name, sdtype, **kwargs)
        if sdtype == 'categorical':
            self._validate_categorical(column_name, **kwargs)
        elif sdtype == 'numerical':
            self._validate_numerical(column_name, **kwargs)
        elif sdtype == 'datetime':
            self._validate_datetime(column_name, **kwargs)
        elif sdtype == 'text':
            self._validate_text(column_name, **kwargs)

    def add_column(self, column_name, **kwargs):
        """Add a column to the ``SingleTableMetadata``.

        Args:
            column_name (str):
                The column name to be added.

            kwargs (type):
                Any additional key word arguments for the column, where ``sdtype`` is required.

        Raises:
            - ``ValueError`` if the column already exists.
            - ``ValueError`` if the ``kwargs`` do not contain ``sdtype``.
            - ``ValueError`` if the column has unexpected values or ``kwargs`` for the given
              ``sdtype``.
        """
        if column_name in self._columns:
            raise ValueError(
                f"Column name '{column_name}' already exists. Use 'update_column' "
                'to update an existing column.'
            )

        sdtype = kwargs.get('sdtype')
        if sdtype is None:
            raise ValueError(f"Please provide a 'sdtype' for column '{column_name}'.")

        self._validate_column(column_name, **kwargs)
        self._columns[column_name] = deepcopy(kwargs)

    def update_column(self, column_name, **kwargs):
        """Update an existing column in the ``SingleTableMetadata``.

        Args:
            column_name (str):
                The column name to be updated.
            **kwargs (type):
                Any key word arguments that describe metadata for the column.

        Raises:
            - ``ValueError`` if the column doesn't already exist in the ``SingleTableMetadata``.
            - ``ValueError`` if the column has unexpected values or ``kwargs`` for the current
              ``sdtype``.
        """
        self._validate_column_exists(column_name)
        _kwargs = deepcopy(kwargs)
        if 'sdtype' in kwargs:
            sdtype = kwargs.pop('sdtype')
        else:
            sdtype = self._columns[column_name]['sdtype']
            _kwargs['sdtype'] = sdtype

        self._validate_column(column_name, sdtype, **kwargs)
        self._columns[column_name] = _kwargs

    def detect_from_dataframe(self, data):
        """Detect the metadata from a ``pd.DataFrame`` object.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.

        Args:
            data (pd.DataFrame):
                ``pandas.DataFrame`` to detect the metadata from.
        """
        if self._columns:
            raise ValueError(
                'Metadata already exists. Create a new ``SingleTableMetadata`` '
                'object to detect from other data sources.'
            )

        for field in data:
            clean_data = data[field].dropna()
            kind = clean_data.infer_objects().dtype.kind
            self._columns[field] = {'sdtype': self._DTYPES_TO_SDTYPES.get(kind, 'categorical')}

        print('Detected metadata:')
        print(json.dumps(self.to_dict(), indent=4))

    def detect_from_csv(self, filepath, pandas_kwargs=None):
        """Detect the metadata from a ``csv`` file.

        This method automatically detects the ``sdtypes`` for a given ``csv`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``csv`` file.
            pandas_kwargs (dict):
                A python dictionary of with string and value accepted by ``pandas.read_csv``
                function. Defaults to ``None``.
        """
        if self._columns:
            raise ValueError(
                'Metadata already exists. Create a new ``SingleTableMetadata`` '
                'object to detect from other data sources.'
            )

        filepath = Path(filepath)
        pandas_kwargs = pandas_kwargs or {}
        data = pd.read_csv(filepath, **pandas_kwargs)
        self.detect_from_dataframe(data)

    def to_dict(self):
        """Return a python ``dict`` representation of the ``SingleTableMetadata``."""
        metadata = {}
        for key, value in self._metadata.items():
            if key == 'constraints' and value:
                constraints = []
                for constraint in value:
                    if not isinstance(constraint, dict):
                        constraints.append(constraint.to_dict())
                    else:
                        constraints.append(constraint)

                metadata[key] = constraints

            elif value:
                metadata[key] = value

        return deepcopy(metadata)

    def _set_metadata_dict(self, metadata):
        """Set a ``metadata`` dictionary to the current instance.

        Args:
            metadata (dict):
                Python dictionary representing a ``SingleTableMetadata`` object.
        """
        self._metadata = {}
        for key in self._KEYS:
            value = deepcopy(metadata.get(key))
            if key == 'constraints' and value:
                value = [Constraint.from_dict(constraint_dict) for constraint_dict in value]

            if value:
                self._metadata[key] = value
                setattr(self, f'_{key}', value)
            else:
                self._metadata[key] = getattr(self, f'_{key}')

    def _validate_constraint_columns_in_metadata(self, constraint_name, column_names):
        missing_columns = [column not in self._columns for column in column_names]
        if missing_columns:
            raise MetadataError(
                f' A {constraint_name} constraint is being applied to invalid column names {missing_columns}.'
                'The columns must exist in the table.'
            )

    def _validate_constraint_against_metadata(self, constraint_name, **kwargs):
        if constraint_name == 'Unique':
            column_names = kwargs.get('column_names')
            self._validate_constraint_columns_in_metadata(constraint_name, column_names)
            keys_for_table = []
            for column in column_names:
                primary_key = self._primary_key if isinstance(self._primary_key, tuple) else tuple(self._primary_key)
                if column in self._alternate_keys or column in primary_key:
                    keys_for_table.append(column)
            raise MetadataError(
                f'A Unique constraint is being applied to columns "{keys_for_table}". '
                'These columns are already a key for that table.'
            )
        if constraint_name == 'FixedCombinations':
            column_names = kwargs.get('column_names')
            self._validate_constraint_columns_in_metadata(constraint_name, column_names)
        if constraint_name == 'Inequality':
            column_names = [kwargs.get('high_column_name'), kwargs.get('low_column_name')]
            self._validate_constraint_columns_in_metadata(constraint_name, column_names)
            both_datetime = all(self._columns.get(column).sdtype == 'datetime' for column in column_names)
            both_numerical = all(self._columns.get(column).sdtype == 'numerical' for column in column_names)
            if not both_datetime or both_numerical:
                raise MetadataError(
                    f'An {constraint_name} constraint is being applied to mismatched sdtypes '
                    f'{column_names}. Both columns must be either numerical or datetime.'
                )
        if constraint_name == 'ScalarInequality':
            column_name = kwargs.get('column_name')
            if column_name not in self._columns:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to invalid column names '
                    f'({column_name}). The columns must exist in the table.'
                )
            sdtype = self._columns.get(column_name).get('sdtype')
            val = kwargs.get('value')
            if sdtype == 'numerical':
                if not isinstance(val, (int, float)):
                    raise MetadataError('"value" must be an int or float')
            elif sdtype == 'datetime':
                datetime_format = self._columns.get(column_name).get('datetime_format')
                matches_format = False # figure this out
                if not matches_format:
                    raise MetadataError('"value" must be a datetime string of the right format')
            else:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to mismatched sdtypes. '
                    'Numerical columns must be compared to integer or float values. '
                    'Datetimes column must be compared to datetime strings.'
                )
        if constraint_name == 'Range':
            column_names = kwargs.get('column_names')
            self._validate_constraint_columns_in_metadata(constraint_name, column_names)
            both_datetime = all(self._columns.get(column).sdtype == 'datetime' for column in column_names)
            both_numerical = all(self._columns.get(column).sdtype == 'numerical' for column in column_names)
            if not both_datetime or both_numerical:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to mismatched sdtypes '
                    f'{column_names}. All columns must be either numerical or datetime.'
                )
        if constraint_name == 'ScalarRange':
            column_name = kwargs.get('column_name')
            if column_name not in self._columns:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to invalid column names '
                    f'({column_name}). The columns must exist in the table.'
                )
            sdtype = self._columns.get(column_name).get('sdtype')
            high_value = kwargs.get('high_value')
            low_value = kwargs.get('low_value')
            if sdtype == 'numerical':
                if not isinstance(high_value, (int, float)) or not isinstance(low_value, (int, float)):
                    raise MetadataError('Both "high_value" and "low_value" must be ints or floats')
            elif sdtype == 'datetime':
                datetime_format = self._columns.get(column_name).get('datetime_format')
                matches_format = False # figure this out
                if not matches_format:
                    raise MetadataError('Both "high_value" and "low_value" must be a datetime string of the right format')
            else:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to mismatched sdtypes. '
                    'Numerical columns must be compared to integer or float values. '
                    'Datetimes column must be compared to datetime strings.'
                )
        if constraint_name == 'Positive':
            column_name = kwargs.get('column_name')
            if column_name not in self._columns:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to invalid column names '
                    f'({column_name}). The columns must exist in the table.'
                )
            sdtype = self._columns.get(column_name, {}).get('sdtype')
            if 'sdtype' != 'numerical':
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to an invalid column '
                    f'{column_name}. This constraint is only defined for numerical columns.'
                )
        if constraint_name == 'Negative':
            column_name = kwargs.get('column_name')
            if column_name not in self._columns:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to invalid column names '
                    f'({column_name}). The columns must exist in the table.'
                )
            sdtype = self._columns.get(column_name, {}).get('sdtype')
            if 'sdtype' != 'numerical':
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to an invalid column '
                    f'{column_name}. This constraint is only defined for numerical columns.'
                )
        if constraint_name == 'FixedIncrements':
            column_name = kwargs.get('column_name')
            if column_name not in self._columns:
                raise MetadataError(
                    f'A {constraint_name} constraint is being applied to invalid column names '
                    f'({column_name}). The columns must exist in the table.'
                )
        if constraint_name == 'OneHotEncoding':
            column_names = kwargs.get('column_names')
            self._validate_constraint_columns_in_metadata(constraint_name, column_names)

    def add_constraint(self, constraint_name, **kwargs):
        """Add a constraint to the single table metadata.

        Args:
            constraint_name (string):
                Name of the constraint class.

            **kwargs:
                Any other arguments the constraint requires.
        """
        constraint_dict = {'constraint': constraint_name}
        constraint_dict.update(**kwargs)
        try:
            constraint = Constraint.from_dict(constraint_dict)
        except Exception:
            raise MetadataError(f'Invalid constraint \'{constraint_name}\'.')
        # TODO: call validation
        self._constraints.append(constraint)

    @classmethod
    def _load_from_dict(cls, metadata):
        """Create a ``SingleTableMetadata`` instance from a python ``dict``.

        Args:
            metadata (dict):
                Python dictionary representing a ``SingleTableMetadata`` object.

        Returns:
            Instance of ``SingleTableMetadata``.
        """
        instance = cls()
        instance._set_metadata_dict(metadata)
        return instance

    @classmethod
    def load_from_json(cls, filepath):
        """Create an instance from a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file.

        Raises:
            - An ``Error`` if the path does not exist.
            - An ``Error`` if the ``json`` file does not contain the ``SCHEMA_VERSION``.

        Returns:
            A ``SingleTableMetadata`` instance.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError(
                f"A file named '{filepath.name}' does not exist. "
                'Please specify a different filename.'
            )

        with open(filepath, 'r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        if 'SCHEMA_VERSION' not in metadata:
            raise ValueError(
                'This metadata file is incompatible with the ``SingleTableMetadata`` '
                'class and version.'
            )

        return cls._load_from_dict(metadata)

    def save_to_json(self, filepath):
        """Save the current ``SingleTableMetadata`` in to a ``json`` file.

        Args:
            filepath (str):
                String that represent the ``path`` to the ``json`` file to be written.

        Raises:
            Raises an ``Error`` if the path already exists.
        """
        filepath = Path(filepath)
        if filepath.exists():
            raise ValueError(
                f"A file named '{filepath.name}' already exists in this folder. Please specify "
                'a different filename.'
            )

        metadata = self.to_dict()
        metadata['SCHEMA_VERSION'] = self.SCHEMA_VERSION
        with open(filepath, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

    def __repr__(self):
        """Pretty print the ``SingleTableMetadata```SingleTableMetadata``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed
