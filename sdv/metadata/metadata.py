"""Metadata."""

import warnings

import pandas as pd

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.utils import read_json


class Metadata(MultiTableMetadata):
    """Metadata class that handles all metadata."""

    METADATA_SPEC_VERSION = 'V1'
    DEFAULT_SINGLE_TABLE_NAME = 'table'

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
        if metadata.get('METADATA_SPEC_VERSION') == 'SINGLE_TABLE_V1':
            single_table_name = single_table_name or cls.DEFAULT_SINGLE_TABLE_NAME
            warnings.warn(
                'You are loading an older SingleTableMetadata object. This will be converted into'
                f" the new Metadata object with a placeholder table name ('{single_table_name}')."
                ' Please save this new object for future usage.'
            )

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

    @staticmethod
    def _validate_infer_sdtypes(infer_sdtypes):
        if not isinstance(infer_sdtypes, bool):
            raise ValueError("'infer_sdtypes' must be a boolean value.")

    @staticmethod
    def _validate_foreign_key_inference_algorithm(foreign_key_inference_algorithm):
        if foreign_key_inference_algorithm != 'column_name_match':
            raise ValueError("'foreign_key_inference_algorithm' must be 'column_name_match'")

    @classmethod
    def _detect_from_dataframes(
        cls,
        data,
        infer_sdtypes=True,
        infer_keys='primary_and_foreign',
        foreign_key_inference_algorithm='column_name_match',
    ):
        if not data or not all(isinstance(df, pd.DataFrame) for df in data.values()):
            raise ValueError('The provided dictionary must contain only pandas DataFrame objects.')
        if infer_keys not in ['primary_and_foreign', 'primary_only', None]:
            raise ValueError(
                "'infer_keys' must be one of: 'primary_and_foreign', 'primary_only', None."
            )
        cls._validate_foreign_key_inference_algorithm(foreign_key_inference_algorithm)
        cls._validate_infer_sdtypes(infer_sdtypes)

        metadata = Metadata()
        for table_name, dataframe in data.items():
            metadata.detect_table_from_dataframe(
                table_name, dataframe, infer_sdtypes, None if infer_keys is None else 'primary_only'
            )

        if infer_keys == 'primary_and_foreign':
            metadata._detect_relationships(data, foreign_key_inference_algorithm)

        return metadata

    @classmethod
    def detect_from_dataframes(
        cls,
        data,
        infer_sdtypes=True,
        infer_keys='primary_and_foreign',
        foreign_key_inference_algorithm='column_name_match',
    ):
        """Detect the metadata for all tables in a dictionary of dataframes.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrames``.
        All data column names are converted to strings.

        Args:
            data (dict):
                Dictionary of table names to dataframes.
            infer_sdtypes (bool):
                A boolean describing whether to infer the sdtypes of each column.
                If True it infers the sdtypes based on the data.
                If False it does not infer the sdtypes and all columns are marked as unknown.
                Defaults to True.
            infer_keys (str):
                A string describing whether to infer the primary and/or foreign keys. Options are:
                    - 'primary_and_foreign': Infer the primary keys in each table,
                       and the foreign keys in other tables that refer to them
                    - 'primary_only': Infer only the primary keys of each table
                    - None: Do not infer any keys
                Defaults to 'primary_and_foreign'.
            foreign_key_inference_algorithm (str):
                Which algorithm to use for detecting foreign keys. Currently only one option,
                'column_name_match'. Defaults to 'column_name_match'.

        Returns:
            Metadata:
                A new metadata object with the sdtypes detected from the data.
        """
        return cls._detect_from_dataframes(
            data=data,
            infer_sdtypes=infer_sdtypes,
            infer_keys=infer_keys,
            foreign_key_inference_algorithm=foreign_key_inference_algorithm,
        )

    @classmethod
    def detect_from_dataframe(
        cls,
        data,
        table_name=DEFAULT_SINGLE_TABLE_NAME,
        infer_sdtypes=True,
        infer_keys='primary_only',
    ):
        """Detect the metadata for a DataFrame.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.
        All data column names are converted to strings.

        Args:
            data (pandas.DataFrame):
                Dictionary of table names to dataframes.
            infer_sdtypes (bool):
                A boolean describing whether to infer the sdtypes of each column.
                If True it infers the sdtypes based on the data.
                If False it does not infer the sdtypes and all columns are marked as unknown.
                Defaults to True.
            infer_keys (str):
                A string describing whether to infer the primary keys. Options are:
                    - 'primary_only': Infer only the primary keys of each table
                    - None: Do not infer any keys
                Defaults to 'primary_only'.

        Returns:
            Metadata:
                A new metadata object with the sdtypes detected from the data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError('The provided data must be a pandas DataFrame object.')
        if infer_keys not in ['primary_only', None]:
            raise ValueError("'infer_keys' must be one of: 'primary_only', None.")
        cls._validate_infer_sdtypes(infer_sdtypes)

        metadata = Metadata()
        metadata.detect_table_from_dataframe(table_name, data, infer_sdtypes, infer_keys)
        return metadata

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
                warnings.warn(
                    'No table name was provided to metadata containing only one table. '
                    f'Assigning name: {single_table_name}'
                )

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

    def _handle_table_name(self, table_name):
        if len(self.tables) == 0:
            raise ValueError('Metadata does not contain any tables. No columns can be added.')
        if table_name is None:
            if len(self.tables) == 1:
                table_name = next(iter(self.tables))
            else:
                raise ValueError(
                    'Metadata contains more than one table, please specify the `table_name`.'
                )

        return table_name

    def set_sequence_index(self, column_name, table_name=None):
        """Set the sequence index of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence index.
            column_name (str):
                Name of the sequence index column.
        """
        table_name = self._handle_table_name(table_name)
        self._validate_table_exists(table_name)
        self.tables[table_name].set_sequence_index(column_name)

    def set_sequence_key(self, column_name, table_name=None):
        """Set the sequence key of a table.

        Args:
            column_name (str, tulple[str]):
                Name (or tuple of names) of the sequence key column(s).
            table_name (str):
                Name of the table to set the sequence key.
                Defaults to None.
        """
        table_name = self._handle_table_name(table_name)
        self._validate_table_exists(table_name)
        self.tables[table_name].set_sequence_key(column_name)

    def validate_table(self, data, table_name=None):
        """Validate a table against the metadata.

        Args:
            data (pandas.DataFrame):
                Data to validate.
            table_name (str):
                Name of the table to validate.
        """
        if table_name is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                table_name = self._get_single_table_name()

        if not table_name:
            raise InvalidMetadataError(
                'Metadata contains more than one table, please specify the `table_name` '
                'to validate.'
            )

        return self.validate_data({table_name: data})

    def get_column_names(self, table_name=None, **kwargs):
        """Return a list of column names that match the given metadata keyword arguments."""
        table_name = self._handle_table_name(table_name)
        return super().get_column_names(table_name, **kwargs)

    def update_column(self, column_name, table_name=None, **kwargs):
        """Update an existing column for a table in the ``Metadata``."""
        table_name = self._handle_table_name(table_name)
        super().update_column(table_name, column_name, **kwargs)

    def update_columns(self, column_names, table_name=None, **kwargs):
        """Update the metadata of multiple columns."""
        table_name = self._handle_table_name(table_name)
        super().update_columns(table_name, column_names, **kwargs)

    def update_columns_metadata(self, column_metadata, table_name=None):
        """Update the metadata of multiple columns."""
        table_name = self._handle_table_name(table_name)
        super().update_columns_metadata(table_name, column_metadata)

    def add_column(self, column_name, table_name=None, **kwargs):
        """Add a column to the metadata."""
        table_name = self._handle_table_name(table_name)
        super().add_column(table_name, column_name, **kwargs)

    def _remove_matching_relationships(self, element, keys):
        """Remove relationships where the element matches the keys to check."""
        updated_relationships = []
        for relationship in self.relationships:
            matching_keys = [relationship[key] for key in keys]
            if element not in matching_keys:
                updated_relationships.append(relationship)

        self.relationships = updated_relationships

    def remove_table(self, table_name):
        """Remove a table from the metadata.

        This method removes a table from the metadata as well as all relationships that table
        contains.

        Args:
            table_name (str):
                The name of the table to remove.
        """
        self._validate_table_exists(table_name)

        # Remove relationships
        self._remove_matching_relationships(table_name, ['parent_table_name', 'child_table_name'])
        del self.tables[table_name]
        self._multi_table_updated = True

    def remove_column(self, column_name, table_name=None):
        """Remove a column from a table in the metadata.

        This method will remove the column from the metadata, delete any relationships the
        column is in, delete any column relationships the column is in and remove it from any keys
        or special columns it is a part of (eg. sequence index).

        Args:
            column_name (str):
                The name of the column to remove.
            table_name (str):
                The name of the table the column belongs to. Required if there is more than one
                table.
        """
        if table_name:
            self._validate_table_exists(table_name)
        else:
            table_name = self._get_single_table_name()

        if table_name is None:
            raise ValueError(
                "'table_name must be provided if there is more than 1 table in the metadata."
            )

        table_metadata = self.tables[table_name]
        table_metadata._validate_column_exists(column_name)

        # Remove relationships
        self._remove_matching_relationships(
            column_name, ['parent_primary_key', 'child_foreign_key']
        )
        updated_column_relationships = []
        for column_relationship in table_metadata.column_relationships:
            if column_name not in column_relationship.get('column_names', []):
                updated_column_relationships.append(column_relationship)

        table_metadata.column_relationships = updated_column_relationships

        # Remove keys and special columns
        if table_metadata.primary_key == column_name:
            table_metadata.remove_primary_key()

        if column_name in table_metadata.alternate_keys:
            table_metadata.alternate_keys.remove(column_name)

        if column_name == table_metadata.sequence_key:
            table_metadata.set_sequence_key(None)

        if column_name == table_metadata.sequence_index:
            table_metadata.remove_sequence_index()

        del table_metadata.columns[column_name]

        self._multi_table_updated = True

    def add_column_relationship(
        self,
        relationship_type,
        column_names,
        table_name=None,
    ):
        """Add a column relationship to the metadata."""
        table_name = self._handle_table_name(table_name)
        super().add_column_relationship(table_name, relationship_type, column_names)

    def set_primary_key(self, column_name, table_name=None):
        """Set the primary key of a table."""
        table_name = self._handle_table_name(table_name)
        super().set_primary_key(table_name, column_name)

    def remove_primary_key(self, table_name=None):
        """Remove the primary key of a table."""
        table_name = self._handle_table_name(table_name)
        super().remove_primary_key(table_name)

    def add_alternate_keys(self, column_names, table_name=None):
        """Add alternate keys to a table."""
        table_name = self._handle_table_name(table_name)
        super().add_alternate_keys(table_name, column_names)

    def get_table_metadata(self, table_name=None):
        """Return the metadata for a table.

        Args:
            table_name (str):
                The name of the table to get the metadata for.

        Returns:
            Metadata:
                The metadata for the given table.
        """
        table_name = self._handle_table_name(table_name)
        table_metadata = super().get_table_metadata(table_name)
        return Metadata.load_from_dict(table_metadata.to_dict(), single_table_name=table_name)

    def copy(self):
        """Return a copy of the metadata.

        Returns:
            Metadata:
                Copy of current metadata.
        """
        return Metadata.load_from_dict(self.to_dict())

    def anonymize(self):
        """Anonymize metadata by obfuscating column names.

        Returns:
            MultiTableMetadata:
                An anonymized MultiTableMetadata instance.
        """
        anonymized_metadata = self._get_anonymized_dict()

        return Metadata.load_from_dict(anonymized_metadata)
