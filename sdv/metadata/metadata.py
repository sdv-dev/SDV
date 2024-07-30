"""Metadata."""

import warnings
from pathlib import Path

import pandas as pd

from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.utils import read_json

DEFAULT_TABLE_NAME = 'default_table_name'
SINGLE_DEPRECATION_MSG = (
    "'SingleTableMetadata' is deprecated. Please use the new 'Metadata' class for synthesizers."
)
MULTI_DEPRECATION_MSG = (
    "'MultiTableMetadata' is deprecated. Please use the new 'Metadata' class for synthesizers."
)


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
    def _convert_to_unified_metadata(cls, metadata):
        """Convert the metadata to Metadata."""
        metadata_type = type(metadata)
        if metadata_type in (SingleTableMetadata, MultiTableMetadata):
            metadata = Metadata().load_from_dict(metadata.to_dict())
            warnings.warn(
                SINGLE_DEPRECATION_MSG
                if metadata_type is SingleTableMetadata
                else MULTI_DEPRECATION_MSG,
                FutureWarning,
            )
        return metadata

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
                single_table_name = DEFAULT_TABLE_NAME

            if 'columns' in metadata:
                self.tables[single_table_name] = SingleTableMetadata.load_from_dict(metadata)

    def _get_table_or_default(self, table_name=None):
        if table_name is None:
            if len(self.tables) > 1:
                raise ValueError(
                    'This metadata contains more than one table. '
                    'Please provide a table name in the method.'
                )

            single_table = next(iter(self.tables.values()), None)
            return single_table

        return self.tables[table_name]

    def _get_table_name(self, table_name=None):
        if table_name is None:
            if len(self.tables) > 1:
                raise ValueError(
                    'This metadata contains more than one table. '
                    'Please provide a table name in the method.'
                )

            table_name = next(iter(self.tables), None)

        return table_name

    def get_column_relationships(self, table_name=None):
        """Get column relationships for a table.

        Grab the column relationships for a single table given a name or the first table if no
        table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            list:
                Returns a list of column relationships dicts
        """
        table = self._get_table_or_default(table_name)
        return getattr(table, 'column_relationships', [])

    def get_primary_key(self, table_name=None):
        """Get the primary key for a table.

        Grab the primary key for a single table given a name or the first table if no
        table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            str:
                Returns the name of the primary key for the table.
        """
        table = self._get_table_or_default(table_name)
        return getattr(table, 'primary_key', None)

    def get_alternate_keys(self, table_name=None):
        """Get the alternate keys for a table.

        Grab the alternate keys for a single table given a name or the first table if no
        table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            list:
                List of alternate keys found in the table.
        """
        table = self._get_table_or_default(table_name)
        return getattr(table, 'alternate_keys', [])

    def get_columns(self, table_name=None):
        """Get the columns for a table.

        Grab the columns for a single table given a name or the first table if no
        table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            dict:
                Returns dict describing columns of the given table.
        """
        table = self._get_table_or_default(table_name)
        return getattr(table, 'columns', {})

    def validate_data(self, data):
        """Validate the data matches the metadata.

        Checks the following rules:
            * all tables of the metadata are present in the data
            * every table of the data satisfies its own metadata
            * all foreign keys belong to a primay key

        Args:
            data (pd.DataFrame or dict):
                The data to validate. If the data is a single dataframe it is converted to
                a dictionary and mapped to a metadata single table (if there is only one otherwise
                it throws an error).

        Raises:
            InvalidDataError:
                - This error is being raised if the data is not matching its sdtype requirements.
                - A dataframe is passed but there is more than one table that exists.

        Warns:
            A warning is being raised if ``datetime_format`` is missing from a column represented
            as ``object`` in the dataframe and its sdtype is ``datetime``.
        """
        data_to_validate = data
        if isinstance(data, pd.DataFrame):
            table_name = self._get_table_name()
            data_to_validate = {table_name: data}

        super().validate_data(data_to_validate)

    def add_column(self, column_name, table_name=None, **kwargs):
        """Add a column to a table in the ``MultiTableMetadata``.

        Args:
            column_name (str):
                The column name to be added.
            table_name (str or None):
                Name of the table to add the column to. If None is passed, default to only table
                if there is only one table.
            **kwargs (type):
                Any additional key word arguments for the column, where ``sdtype`` is required.

        Raises:
            - ``InvalidMetadataError`` if the column already exists.
            - ``InvalidMetadataError`` if the ``kwargs`` do not contain ``sdtype``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              given ``sdtype``.
            - ``InvalidMetadataError`` if the table doesn't exist in the ``MultiTableMetadata``.
        """
        table_name = self._get_table_name(table_name)
        if table_name is None:
            self.tables[DEFAULT_TABLE_NAME] = SingleTableMetadata()
            table_name = DEFAULT_TABLE_NAME

        super().add_column(table_name, column_name, **kwargs)

    def set_sequence_key(self, column_name, table_name=None):
        """Set the sequence key of a table.

        Args:
            table_name (str or None):
                Name of the table to set the sequence key. If None, defaults to the single
                table if there is only one.
            column_name (str, tuple[str]):
                Name (or tuple of names) of the sequence key column(s).
        """
        table_name = self._get_table_name(table_name)
        super().set_sequence_key(table_name, column_name)

    def add_alternate_keys(self, column_names, table_name=None):
        """Set the alternate keys of a table.

        Args:
            table_name (str or None):
                Name of the table to set the sequence key. If None, defaults to the single
                table if there is only one.
            column_names (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
        """
        table_name = self._get_table_name(table_name)
        super().add_alternate_keys(table_name, column_names)

    def get_primary_and_alternate_keys(self, table_name=None):
        """Get primary and alternate keys for a table.

        Grab the primary and alternate keys  for a single table given a name
        or the first table if no table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            set:
                Set of keys.
        """
        table = self._get_table_or_default(table_name)
        return table._get_primary_and_alternate_keys()

    def get_set_of_sequence_keys(self, table_name=None):
        """Get sequence keys for a table.

        Grab the sequence keys for a single table given a name
        or the first table if no table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            set:
                Set of keys.
        """
        table = self._get_table_or_default(table_name)
        return table._get_set_of_sequence_keys()

    def set_primary_key(self, column_name, table_name=None):
        """Set the primary key for a table.

        Set the primary key for a single table given a name
        or the first table if no table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.
        """
        table_name = self._get_table_name(table_name)
        return super().set_primary_key(table_name, column_name)

    def detect_from_dataframes(self, data):
        """Detect the metadata for all tables in a dictionary of dataframes.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.
        All data column names are converted to strings.

        Args:
            data (dict or DataFrame):
                Dictionary of table names to dataframes. If the data is a single dataframe
                it is converted to a dictionary and mapped to a metadata single table (if
                there is only one otherwise it throws an error).
        """
        data_to_detect = data
        if isinstance(data, pd.DataFrame):
            table_name = self._get_table_name()
            data_to_detect = {table_name: data}

        super().detect_from_dataframes(data_to_detect)

    def get_sequence_key(self, table_name=None):
        """Get sequence key for a table.

        Grab the sequence key for a single table given a name
        or the first table if no table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            str:
                Sequence key name
        """
        table = self._get_table_or_default(table_name)
        return getattr(table, 'sequence_key', None)

    def get_sequence_index(self, table_name=None):
        """Get sequence index for a table.

        Grab the sequence index for a single table given a name
        or the first table if no table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.

        Returns:
            str:
                Sequence Index name for the given table.
        """
        table = self._get_table_or_default(table_name)
        return getattr(table, 'sequence_index', None)

    def get_valid_column_relationships(self, table_name=None):
        """Get valid columns relationships for a table.

        Grab the valid columns relationships for a single table given a name
        or the first table if no table is given. (Must contain only one table to use None)

        Args:
            table_name (str or None):
                Table name, if there is no table name provided, chooses the default
                table only if there is a single table.
        """
        table = self._get_table_or_default(table_name)
        return table._valid_column_relationships

    def add_column_relationship(self, relationship_type, column_names, table_name=None):
        """Add a column relationship to a table in the metadata.

        Args:
            table_name (str):
                The name of the table to add this relationship to. If there is no table
                name provided, chooses the default table only if there is a single table.
            relationship_type (str):
                The type of the relationship.
            column_names (list[str]):
                The list of column names involved in this relationship.
        """
        table_name = self._get_table_name(table_name)
        super().add_column_relationship(table_name, relationship_type, column_names)

    def update_column(self, column_name, table_name=None, **kwargs):
        """Update an existing column for a table in the ``Metadata``.

        Args:
            table_name (str or None):
                Name of table the column belongs to. If there is no table name provided,
                chooses the default table only if there is a single table.
            column_name (str):
                The column name to be updated.
            **kwargs (type):
                Any key word arguments that describe metadata for the column.

        Raises:
            - ``InvalidMetadataError`` if the column doesn't already exist in the
              ``SingleTableMetadata``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              current ``sdtype``.
            - ``InvalidMetadataError`` if the table doesn't exist in the ``MultiTableMetadata``.
        """
        table_name = self._get_table_name(table_name)
        return super().update_column(table_name, column_name, **kwargs)
