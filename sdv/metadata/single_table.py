"""Single Table Metadata."""

import copy
import json
from pathlib import Path

import pandas as pd


class SingleTableMetadata:
    """Single Table Metadata class."""

    _DTYPES_TO_SDTYPES = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    KEYS = ['columns', 'primary_key', 'alternate_keys', 'constraints', 'SCHEMA_VERSION']
    SCHEMA_VERSION = 'SINGLE_TABLE_V1'

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
            self._columns[field] = {'sdtype': self._DTYPES_TO_SDTYPES[kind]}

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
            if value:
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

    def __repr__(self):
        """Pretty print the ``SingleTableMetadata```SingleTableMetadata``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed
