"""Single Table Metadata."""

import copy
import json


class SingleTableMetadata:
    """Single Table Metadata class."""

    SCHEMA_VERSION = 'SINGLE_TABLE_V1'
    KEYS = ['columns', 'primary_key', 'alternate_keys', 'constraints', 'SCHEMA_VERSION']

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
