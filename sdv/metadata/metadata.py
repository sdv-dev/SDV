"""Metadata."""

import warnings

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.utils import read_json


class Metadata(MultiTableMetadata):
    """Metadata class that handles all metadata."""

    METADATA_SPEC_VERSION = 'V1'
    DEFAULT_SINGLE_TABLE_NAME = 'default_table_name'

    @classmethod
    def load_from_json(cls, filepath, single_table_name=None):
        """Create a ``Metadata`` instance from a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file.

        Raises:
            - An ``Error`` if the path does not exist.
            - An ``Error`` if the ``json`` file does not contain the ``METADATA_SPEC_VERSION``.

        Returns:
            A ``Metadata`` instance.
        """
        metadata = read_json(filepath)
        return cls.load_from_dict(metadata, single_table_name)

    @classmethod
    def load_from_dict(cls, metadata_dict, single_table_name=None):
        """Create a ``Metadata`` instance from a python ``dict``.

        Args:
            metadata_dict (dict):
                Python dictionary representing a ``MultiTableMetadata``
                or ``SingleTableMetadata`` object.
            single_table_name (string):
                If the python dictionary represents a ``SingleTableMetadata`` then
                this arg is used for the name of the table.

        Returns:
            Instance of ``Metadata``.
        """
        instance = cls()
        instance._set_metadata_dict(metadata_dict, single_table_name)
        return instance

    def _set_metadata_dict(self, metadata, single_table_name=None):
        """Set a ``metadata`` dictionary to the current instance.

        Checks to see if the metadata is in the ``SingleTableMetadata`` or
        ``MultiTableMetadata`` format and converts it to a standard
        ``MultiTableMetadata`` format if necessary.

        Args:
            metadata (dict):
                Python dictionary representing a ``MultiTableMetadata`` or
                ``SingleTableMetadata`` object.
        """
        is_multi_table = 'tables' in metadata

        if is_multi_table:
            super()._set_metadata_dict(metadata)
        else:
            if single_table_name is None:
                single_table_name = self.DEFAULT_SINGLE_TABLE_NAME
            self.tables[single_table_name] = SingleTableMetadata.load_from_dict(metadata)

    def _get_single_table_name(self):
        """Get the table name if there is only one table.

        Checks to see if the metadata contains only a single table, if so
        return the name. Otherwise warn the user and return None.

        Args:
            metadata (dict):
                Python dictionary representing a ``MultiTableMetadata`` or
                ``SingleTableMetadata`` object.
        """
        if len(self.tables) != 1:
            warnings.warn(
                'This metadata does not contain only a single table. Could not determine '
                'single table name and will return None.'
            )
            return None

        return next(iter(self.tables), None)

    def _convert_to_single_table(self):
        if len(self.tables) > 1:
            raise InvalidMetadataError(
                'Metadata contains more than one table, use a MultiTableSynthesizer instead.'
            )

        return next(iter(self.tables.values()), SingleTableMetadata())

    def set_sequence_index(self, table_name, column_name):
        """Set the sequence index of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence index.
            column_name (str):
                Name of the sequence index column.
        """
        self._validate_table_exists(table_name)
        self.tables[table_name].set_sequence_index(column_name)

    def set_sequence_key(self, table_name, column_name):
        """Set the sequence key of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence key.
            column_name (str, tulple[str]):
                Name (or tuple of names) of the sequence key column(s).
        """
        self._validate_table_exists(table_name)
        self.tables[table_name].set_sequence_key(column_name)
