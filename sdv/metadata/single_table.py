"""Single Table Metadata."""

import json
import re
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from sdv.constraints import Constraint
from sdv.metadata.errors import InvalidMetadataError, MetadataError


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
        'constraints',
        'primary_key',
        'alternate_keys',
        'sequence_key',
        'sequence_index',
        'SCHEMA_VERSION'
    ])
    SCHEMA_VERSION = 'SINGLE_TABLE_V1'

    def _validate_numerical(self, column_name, **kwargs):
        representation = kwargs.get('representation')
        if representation and representation not in self._NUMERICAL_REPRESENTATIONS:
            raise ValueError(
                f"Invalid value for 'representation' '{representation}'"
                f" for column '{column_name}'."
            )

    @staticmethod
    def _validate_datetime(column_name, **kwargs):
        datetime_format = kwargs.get('datetime_format')
        if datetime_format is not None:
            try:
                # NOTE: I don't know if this ever crashes, it just returns the string as is
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
        if order is not None and order_by is not None:
            raise ValueError(
                f"Categorical column '{column_name}' has both an 'order' and 'order_by' "
                'attribute. Only 1 is allowed.'
            )
        if order_by is not None and order_by not in ('numerical_value', 'alphabetical'):
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
        self._constraints = []
        self._primary_key = None
        self._alternate_keys = []
        self._sequence_key = None
        self._sequence_index = None
        self._version = self.SCHEMA_VERSION
        self._metadata = {
            'columns': self._columns,
            'primary_key': None,
            'alternate_keys': [],
            'sequence_key': None,
            'sequence_index': None,
            'constraints': self._constraints,
            'SCHEMA_VERSION': self.SCHEMA_VERSION
        }

    def _validate_unexpected_kwargs(self, column_name, sdtype, **kwargs):
        expected_kwargs = self._EXPECTED_KWARGS.get(sdtype, ['pii'])
        unexpected_kwargs = set(list(kwargs)) - set(expected_kwargs)
        if unexpected_kwargs:
            unexpected_kwargs = sorted(unexpected_kwargs)
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

    @staticmethod
    def _validate_datatype(id):
        """Check whether id is a string or a tuple of strings."""
        return isinstance(id, str) or isinstance(id, tuple) and all(isinstance(i, str) for i in id)

    def _validate_key(self, id, key_type):
        """Validate the primary and sequence keys."""
        if not self._validate_datatype(id):
            raise ValueError(f"'{key_type}_key' must be a string or tuple of strings.")

        keys = {id} if isinstance(id, str) else set(id)
        invalid_ids = keys - set(self._columns)
        if invalid_ids:
            raise ValueError(
                f'Unknown {key_type} key values {invalid_ids}.'
                ' Keys should be columns that exist in the table.'
            )

    def set_primary_key(self, id):
        """Set the metadata primary key.

        Args:
            id (str, tuple):
                Name (or tuple of names) of the primary key column(s).
        """
        self._validate_key(id, 'primary')

        if self._metadata['primary_key'] is not None:
            warnings.warn(
                f"There is an existing primary key {self._metadata['primary_key']}."
                ' This key will be removed.'
            )

        self._primary_key = id

    def set_sequence_key(self, id):
        """Set the metadata sequence key.

        Args:
            id (str, tuple):
                Name (or tuple of names) of the sequence key column(s).
        """
        self._validate_key(id, 'sequence')

        if self._metadata['sequence_key'] is not None:
            warnings.warn(
                f"There is an existing sequence key {self._metadata['sequence_key']}."
                ' This key will be removed.'
            )

        self._sequence_key = id

    def _validate_alternate_keys(self, ids):
        if not isinstance(ids, list) or not all(self._validate_datatype(id) for id in ids):
            raise ValueError(
                "'alternate_keys' must be a list of strings or a list of tuples of strings."
            )

        keys = set()
        for id in ids:
            keys.update({id} if isinstance(id, str) else set(id))

        invalid_ids = keys - set(self._columns)
        if invalid_ids:
            raise ValueError(
                f'Unknown alternate key values {invalid_ids}.'
                ' Keys should be columns that exist in the table.'
            )

    def set_alternate_keys(self, ids):
        """Set the metadata alternate keys.

        Args:
            ids (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
        """
        self._validate_alternate_keys(ids)
        self._metadata['alternate_keys'] = ids

    def _validate_sequence_index(self, column_name):
        if not isinstance(column_name, str):
            raise ValueError("'sequence_index' must be a string.")

        if column_name not in self._columns:
            column_name = {column_name}
            raise ValueError(
                f'Unknown sequence key value {column_name}.'
                ' Keys should be columns that exist in the table.'
            )

    def set_sequence_index(self, column_name):
        """Set the metadata sequence index.

        Args:
            column_name (str):
                Name of the sequence index column.
        """
        self._validate_sequence_index(column_name)
        self._sequence_index = column_name

    def _validate_sequence_index_not_in_sequence_key(self):
        """Check that ``_sequence_index`` and ``_sequence_key`` don't overlap."""
        sk = self._sequence_key
        sequence_key = {sk} if isinstance(sk, str) else set(sk)
        if self._sequence_index in sequence_key or sk is None:
            index = {self._sequence_index}
            raise ValueError(
                f"'sequence_index' and 'sequence_key' have the same value {index}."
                ' These columns must be different.'
            )

    def validate(self):
        """Validate the metadata.

        Raises:
            - ``InvalidMetadataError`` if the metadata is invalid.
        """
        # Validate keys
        errors = []
        try:
            self._validate_key(self._primary_key, 'primary')
        except ValueError as e:
            errors.append(e)

        try:
            self._validate_key(self._sequence_key, 'sequence')
        except ValueError as e:
            errors.append(e)

        try:
            self._validate_alternate_keys(self._alternate_keys)
        except ValueError as e:
            errors.append(e)

        try:
            self._validate_sequence_index(self._sequence_index)
        except ValueError as e:
            errors.append(e)

        try:
            self._validate_sequence_index_not_in_sequence_key()
        except ValueError as e:
            errors.append(e)

        # Validate columns
        for column, kwargs in self._columns.items():
            try:
                self._validate_column(column, **kwargs)
            except ValueError as e:
                errors.append(e)

        if errors:
            raise InvalidMetadataError(
                'The following errors were found in the metadata:\n\n'
                + '\n'.join([str(e) for e in errors])
            )

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
        """Set the ``metadata`` dictionary to the current instance.

        Args:
            metadata (dict):
                Python dictionary representing a ``SingleTableMetadata`` object.
        """
        for key in self._KEYS:
            value = deepcopy(metadata.get(key))
            if key == 'constraints' and value:
                value = [Constraint.from_dict(constraint_dict) for constraint_dict in value]

            if value:
                self._metadata[key] = value
                setattr(self, f'_{key}', value)

    def add_constraint(self, constraint_name, **kwargs):
        """Add a constraint to the single table metadata.

        Args:
            constraint_name (string):
                Name of the constraint class.

            **kwargs:
                Any other arguments the constraint requires.
        """
        try:
            constraint_class = Constraint._get_class_from_dict(constraint_name)
        except KeyError:
            raise MetadataError(f"Invalid constraint ('{constraint_name}').")

        constraint_class._validate_metadata(self, **kwargs)
        constraint = constraint_class(**kwargs)
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
