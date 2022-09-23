"""Single Table Metadata."""

import json
import re
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from sdv.constraints import Constraint
from sdv.constraints.errors import AggregateConstraintsError
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata_upgrader import convert_metadata
from sdv.metadata.utils import cast_to_iterable, read_json, validate_file_does_not_exist


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
        'Float', 'Int64', 'Int32', 'Int16', 'Int8',
        'UInt64', 'UInt32', 'UInt16', 'UInt8',
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
        regex = kwargs.get('regex_format', '')
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

    def _validate_unexpected_kwargs(self, column_name, sdtype, **kwargs):
        expected_kwargs = self._EXPECTED_KWARGS.get(sdtype, ['pii'])
        unexpected_kwargs = set(kwargs) - set(expected_kwargs)
        if unexpected_kwargs:
            unexpected_kwargs = sorted(unexpected_kwargs)
            unexpected_kwargs = ', '.join(unexpected_kwargs)
            raise ValueError(
                f"Invalid values '({unexpected_kwargs})' for {sdtype} column '{column_name}'.")

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
        column_kwargs = deepcopy(kwargs)
        if sdtype not in self._EXPECTED_KWARGS:
            pii = column_kwargs.get('pii', True)
            column_kwargs['pii'] = pii

        self._columns[column_name] = column_kwargs

    def _validate_column_exists(self, column_name):
        if column_name not in self._columns:
            raise ValueError(
                f"Column name ('{column_name}') does not exist in the table. "
                "Use 'add_column' to add new column."
            )

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

    def _validate_constraint(self, constraint_name, **kwargs):
        """Validate a constraint against the single table metadata.

        Args:
            constraint_name (string):
                Name of the constraint class.
            **kwargs:
                Any arguments the constraint requires.
        """
        try:
            constraint_class = Constraint._get_class_from_dict(constraint_name)
        except KeyError:
            raise InvalidMetadataError(f"Invalid constraint ('{constraint_name}').")

        constraint_class._validate_metadata(self, **kwargs)

    def add_constraint(self, constraint_name, **kwargs):
        """Add a constraint to the single table metadata.

        Args:
            constraint_name (string):
                Name of the constraint class.
            **kwargs:
                Any other arguments the constraint requires.
        """
        self._validate_constraint(constraint_name, **kwargs)
        kwargs['constraint_name'] = constraint_name
        self._constraints.append(kwargs)

    def detect_from_dataframe(self, data):
        """Detect the metadata from a ``pd.DataFrame`` object.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.

        Args:
            data (pandas.DataFrame):
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

        print('Detected metadata:')  # noqa: T001
        print(json.dumps(self.to_dict(), indent=4))  # noqa: T001

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
    def _validate_datatype(column_name):
        """Check whether column_name is a string or a tuple of strings."""
        return isinstance(column_name, str) or \
            isinstance(column_name, tuple) and all(isinstance(i, str) for i in column_name)

    def _validate_keys_sdtype(self, keys, key_type):
        """Validate that no key is of type 'categorical'."""
        bad_sdtypes = ('boolean', 'categorical')
        categorical_keys = sorted(
            {key for key in keys if self._columns[key]['sdtype'] in bad_sdtypes}
        )
        if categorical_keys:
            raise ValueError(
                f"The {key_type}_keys {categorical_keys} cannot be type 'categorical' or "
                "'boolean'."
            )

    def _validate_key(self, column_name, key_type):
        """Validate the primary and sequence keys."""
        if column_name is not None:
            if not self._validate_datatype(column_name):
                raise ValueError(f"'{key_type}_key' must be a string or tuple of strings.")

            keys = {column_name} if isinstance(column_name, str) else set(column_name)
            invalid_ids = keys - set(self._columns)
            if invalid_ids:
                raise ValueError(
                    f'Unknown {key_type} key values {invalid_ids}.'
                    ' Keys should be columns that exist in the table.'
                )

            self._validate_keys_sdtype(keys, key_type)

    def set_primary_key(self, column_name):
        """Set the metadata primary key.

        Args:
            column_name (str, tuple):
                Name (or tuple of names) of the primary key column(s).
        """
        self._validate_key(column_name, 'primary')
        if self._primary_key is not None:
            warnings.warn(
                f'There is an existing primary key {self._primary_key}.'
                ' This key will be removed.'
            )

        self._primary_key = column_name

    def set_sequence_key(self, column_name):
        """Set the metadata sequence key.

        Args:
            column_name (str, tuple):
                Name (or tuple of names) of the sequence key column(s).
        """
        self._validate_key(column_name, 'sequence')
        if self._sequence_key is not None:
            warnings.warn(
                f'There is an existing sequence key {self._sequence_key}.'
                ' This key will be removed.'
            )

        self._sequence_key = column_name

    def _validate_alternate_keys(self, column_names):
        if not isinstance(column_names, list) or \
           not all(self._validate_datatype(column_name) for column_name in column_names):
            raise ValueError(
                "'alternate_keys' must be a list of strings or a list of tuples of strings."
            )

        keys = set()
        for column_name in column_names:
            keys.update({column_name} if isinstance(column_name, str) else set(column_name))

        invalid_ids = keys - set(self._columns)
        if invalid_ids:
            raise ValueError(
                f'Unknown alternate key values {invalid_ids}.'
                ' Keys should be columns that exist in the table.'
            )

        self._validate_keys_sdtype(keys, 'alternate')

    def set_alternate_keys(self, column_names):
        """Set the metadata alternate keys.

        Args:
            column_names (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
        """
        self._validate_alternate_keys(column_names)
        self._alternate_keys = column_names

    def _validate_sequence_index(self, column_name):
        if not isinstance(column_name, str):
            raise ValueError("'sequence_index' must be a string.")

        if column_name not in self._columns:
            column_name = {column_name}
            raise ValueError(
                f'Unknown sequence index value {column_name}.'
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
        seq_key = self._sequence_key
        sequence_key = set(cast_to_iterable(seq_key))
        if self._sequence_index in sequence_key or seq_key is None:
            index = {self._sequence_index}
            raise ValueError(
                f"'sequence_index' and 'sequence_key' have the same value {index}."
                ' These columns must be different.'
            )

    def _append_error(self, errors, method, *args, **kwargs):
        """Inplace, append the produced error to the passed ``errors`` list."""
        try:
            method(*args, **kwargs)
        except ValueError as e:
            errors.append(e)

    def validate(self):
        """Validate the metadata.

        Raises:
            - ``InvalidMetadataError`` if the metadata is invalid.
        """
        # Validate constraints
        errors = []
        for constraint_dict in self._constraints:
            constraint_dict = deepcopy(constraint_dict)
            constraint_name = constraint_dict.pop('constraint_name')
            try:
                self._validate_constraint(constraint_name, **constraint_dict)
            except AggregateConstraintsError as e:
                reformated_errors = '\n'.join(map(str, e.errors))
                errors.append(reformated_errors)

        # Validate keys
        self._append_error(errors, self._validate_key, self._primary_key, 'primary')
        self._append_error(errors, self._validate_key, self._sequence_key, 'sequence')
        if self._sequence_index:
            self._append_error(errors, self._validate_sequence_index, self._sequence_index)
            self._append_error(errors, self._validate_sequence_index_not_in_sequence_key)

        self._append_error(errors, self._validate_alternate_keys, self._alternate_keys)

        # Validate columns
        for column, kwargs in self._columns.items():
            self._append_error(errors, self._validate_column, column, **kwargs)

        if errors:
            raise InvalidMetadataError(
                'The following errors were found in the metadata:\n\n'
                + '\n'.join([str(e) for e in errors])
            )

    def to_dict(self):
        """Return a python ``dict`` representation of the ``SingleTableMetadata``."""
        metadata = {}
        for key in self._KEYS:
            value = getattr(self, f'_{key}') if key != 'SCHEMA_VERSION' else self._version
            if value:
                metadata[key] = value

        return deepcopy(metadata)

    def save_to_json(self, filepath):
        """Save the current ``SingleTableMetadata`` in to a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file to be written.

        Raises:
            Raises an ``Error`` if the path already exists.
        """
        validate_file_does_not_exist(filepath)
        metadata = self.to_dict()
        metadata['SCHEMA_VERSION'] = self.SCHEMA_VERSION
        with open(filepath, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

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
        for key in instance._KEYS:
            value = deepcopy(metadata.get(key))
            if value:
                setattr(instance, f'_{key}', value)

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
        metadata = read_json(filepath)
        if 'SCHEMA_VERSION' not in metadata:
            raise ValueError(
                'This metadata file is incompatible with the ``SingleTableMetadata`` '
                'class and version.'
            )

        return cls._load_from_dict(metadata)

    def __repr__(self):
        """Pretty print the ``SingleTableMetadata``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed

    @classmethod
    def _convert_metadata(cls, old_metadata):
        new_metadata = {}
        columns = {}
        fields = old_metadata.get('fields')
        alternate_keys = []
        primary_key = old_metadata.get('primary_key')
        for field, field_meta in fields.items():
            column_meta = {}
            old_type = field_meta['type']
            subtype = field_meta.get('subtype')
            column_meta['sdtype'] = old_type

            if old_type == 'numerical':
                if subtype == 'float':
                    column_meta['representation'] = 'Float'
                elif subtype == 'integer':
                    column_meta['representation'] = 'Int64'

            elif old_type == 'datetime':
                datetime_format = field_meta.get('format')
                if datetime_format:
                    column_meta['datetime_format'] = datetime_format

            elif old_type == 'id':
                if subtype == 'integer':
                    column_meta['sdtype'] = 'numerical'

                elif subtype == 'string':
                    column_meta['sdtype'] = 'text'
                    regex_format = field_meta.get('regex', '[A-Za-z]{5}')
                    if regex_format:
                        column_meta['regex_format'] = regex_format

                if field != primary_key:
                    alternate_keys.append(field)

            columns[field] = column_meta

        new_metadata['columns'] = columns
        new_metadata['primary_key'] = primary_key
        if alternate_keys:
            new_metadata['alternate_keys'] = alternate_keys

        return new_metadata

    @classmethod
    def upgrade_metadata(cls, old_filepath, new_filepath):
        """Upgrade an old metadata file to the ``V1`` schema.

        Args:
            old_filepath (str):
                String that represents the ``path`` to the old metadata ``json`` file.
            new_file_path (str):
                String that represents the ``path`` to save the upgraded metadata to.

        Raises:
            Raises a ``ValueError`` if the path already exists.
        """
        validate_file_does_not_exist(new_filepath)
        old_metadata = read_json(old_filepath)
        if 'tables' in old_metadata:
            tables = old_metadata.get('tables')
            if len(tables) > 1:
                raise ValueError(
                    'There are multiple tables specified in the JSON. '
                    'Try using the MultiTableMetadata class to upgrade this file.'
                )

            else:
                old_metadata = list(tables.values())[0]

        new_metadata = convert_metadata(old_metadata)
        metadata = cls._load_from_dict(new_metadata)
        metadata.save_to_json(new_filepath)

        try:
            metadata.validate()
        except InvalidMetadataError as error:
            message = (
                'Successfully converted the old metadata, but the metadata was not valid. '
                f'To use this with the SDV, please fix the following errors.\n {str(error)}'
            )
            warnings.warn(message)
