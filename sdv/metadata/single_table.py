"""Single Table Metadata."""

import copy
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from sdv.constraints import Constraint


class SingleTableMetadata:
    """Single Table Metadata class."""

    _EXPECTED_KWARGS = {
        'numerical': ['representation'],
        'datetime': ['datetime_format'],
        'categorical': ['order', 'order_by'],
        'text': ['regex_format'],
    }

    _DTYPES_TO_SDTYPES = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    KEYS = ['columns', 'primary_key', 'alternate_keys', 'constraints', 'SCHEMA_VERSION']
    SCHEMA_VERSION = 'SINGLE_TABLE_V1'

    @staticmethod
    def _validate_numerical(column_name, **kwargs):
        representations = [
            'int', 'int64', 'int32', 'int16', 'int8',
            'uint', 'uint64', 'uint32', 'uint16', 'uint8'
            'float', 'float64', 'float32', 'float16', 'float8'
        ]
        representation = kwargs.get('representation')
        if representation and representation not in representations:
            raise ValueError(
                f"Invalid value for 'representation' {representation} for column '{column_name}.")

    @staticmethod
    def _validate_datetime(self, column_name, kwargs):
        datetime_format = kwargs.get('datetime_format')
        if datetime_format:
            formated_date = datetime.now().strftime(datetime_format)
            match = re.search('%.', formated_date)
            if match:
                raise ValueError(
                    f"Invalid datetime fromat stringa '{match.group(0)}' "
                    f"for datetime column '{column_name}'."
                )

    @staticmethod
    def _validate_categorical(self, column_name, kwargs):
        order = kwargs.get('order')
        order_by = kwargs.get('order_by')
        if order and order_by:
            raise ValueError(
                f"Categorical column '{column_name}' has both an 'order' and 'order_by' "
                'attribute. Only 1 is allowed.'
            )
        elif order_by not in ('numerical_value', 'alphabetical'):
            raise ValueError(
                f"Unknown ordering method 'testing' provided for categorical column "
                "'{column_name}'. Ordering method must be 'numerical_value' or 'alphabetical'."
            )
        elif order and not isinstance(order, list):
            raise ValueError(
                f"Invalid order value provided for categorical column '{column_name}'. "
                "The 'order' must be a list with 1 or more elements."
            )

    @staticmethod
    def _validate_text(self, column_name, kwargs):
        regex = kwargs.get('regex')
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

    def _validate_unexpected_kwargs(self, column_name, sdtype, actual_kwargs):
        expected_kwargs = DEFAULT_KWARGS.get(sdtype, ['pii'])
        unexpected_kwargs = set(list(actual_kwargs)) - set(expected_kwargs)
        if unexpected_kwargs:
            unexpected_kwargs = ', '.join(unexpected_kwargs)
            raise ValueError(
                f"Invalid values '({unexpected_kwargs})' for {sdtype} column '{column_name}'.")

    def _validate_column_exists(self, column_name):
        if column_name not in self._columns:
            raise ValueError(
                f"Column name ('{column_name}') does not exist in the table. "
                "Use 'add_column'  to add new column."
            )

    def _validate_column(self, column_name, sdtype, kwargs):
        self._validate_unexpected_kwargs(column_name, sdtype, kwargs)
        if sdtype == 'categorical':
            self._validate_categorical(column_name, kwargs)
        elif sdtype == 'numerical':
            self._validate_numerical(column_name, kwargs)
        elif sdtype == 'datetime':
            self._validate_datetime(column_name, kwargs)

    def update_column(self, column_name, **kwargs):
        self._validate_column_exists(column_name)
        sdtype = self._columns[column_name]['sdtype']
        self._vlidate_column(column_name, sdtype, kwargs)
        self._columns[column_name] = deepcopy(kwargs)

    def add_column(column_name, **kwargs):
        if column_name in self._columns:
            raise ValueError(
                f"Column name '{column_name}' already exists. Use 'update_column' "
                'to update an existing column.'
            )

        sdtype = kwargs.get('sdtype')
        if sdtype is None:
            raise ValueError(
                f"Please provide a 'sdtype' for column '{column_name}'."
            )

        self._validate_column(column_name, kwargs)
        self._columns[column_name] = deepcopy(kwargs)

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

        return copy.deepcopy(metadata)

    def _set_metadata_dict(self, metadata):
        """Set a ``metadata`` dictionary to the current instance.

        Args:
            metadata (dict):
                Python dictionary representing a ``SingleTableMetadata`` object.
        """
        self._metadata = {}
        for key in self.KEYS:
            value = copy.deepcopy(metadata.get(key))
            if key == 'constraints' and value:
                value = [Constraint.from_dict(constraint_dict) for constraint_dict in value]

            if value:
                self._metadata[key] = value
                setattr(self, f'_{key}', value)
            else:
                self._metadata[key] = getattr(self, f'_{key}')

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
