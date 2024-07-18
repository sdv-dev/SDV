"""Metadata."""

from pathlib import Path

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.utils import read_json


class Metadata(MultiTableMetadata):
    """Metadata class that handles all metadata."""

    METADATA_SPEC_VERSION = 'V1'

    @classmethod
    def load_from_json(cls, filepath):
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
        filename = Path(filepath).stem
        metadata = read_json(filepath)
        return cls.load_from_dict(metadata, filename)

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

    @classmethod
    def load_from_single_table_metadata(cls, single_metadata_table, table_name=None):
        """Return a unified Metadata object from a legacy SingleTableMetadata object.

        Args:
            single_metadata_table (SingleTableMetadata):
                ``SingleTableMetadata`` object to be converted to a ``Metadata`` object.
            table_name (string):
                The name of the table that will be stored in the ``Metadata object.

        Returns:
            Instance of ``Metadata``.
        """
        if not isinstance(single_metadata_table, SingleTableMetadata):
            raise InvalidMetadataError('Cannot convert given legacy metadata')
        instance = cls()
        instance._set_metadata_dict(single_metadata_table.to_dict())
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
                single_table_name = 'default_table_name'

            self.tables[single_table_name] = SingleTableMetadata.load_from_dict(metadata)

    def _convert_to_single_table(self):
        is_multi_table = len(self.tables) > 1
        if is_multi_table:
            raise InvalidMetadataError(
                'Metadata contains more than one table, use a MultiTableSynthesizer instead.'
            )

        if len(self.tables) == 0:
            return SingleTableMetadata()

        single_table_metadata = next(iter(self.tables.values()))
        return single_table_metadata
