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

    @classmethod
    def detect_from_dataframes(cls, data):
        """Detect the metadata for all tables in a dictionary of dataframes.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrames``.
        All data column names are converted to strings.

        Args:
            data (dict):
                Dictionary of table names to dataframes.

        Returns:
            Metadata:
                A new metadata object with the sdtypes detected from the data.
        """
        if not data or not all(isinstance(df, pd.DataFrame) for df in data.values()):
            raise ValueError('The provided dictionary must contain only pandas DataFrame objects.')

        metadata = Metadata()
        for table_name, dataframe in data.items():
            metadata.detect_table_from_dataframe(table_name, dataframe)

        metadata._detect_relationships(data)
        return metadata

    @classmethod
    def detect_from_dataframe(cls, data, table_name=DEFAULT_SINGLE_TABLE_NAME):
        """Detect the metadata for a DataFrame.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.
        All data column names are converted to strings.

        Args:
            data (pandas.DataFrame):
                Dictionary of table names to dataframes.

        Returns:
            Metadata:
                A new metadata object with the sdtypes detected from the data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError('The provided data must be a pandas DataFrame object.')

        metadata = Metadata()
        metadata.detect_table_from_dataframe(table_name, data)
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

    def set_sequence_key(self, *args, **kwargs):
        """Set the sequence key of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence key.
            column_name (str, tulple[str]):
                Name (or tuple of names) of the sequence key column(s).
        """
        parameters = self._resolve_arguments(['column_name'], *args, **kwargs)
        table_name = parameters['table_name']
        column_name = parameters['column_name']
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

    def _resolve_arguments(self, arg_names, *args, **kwargs):
        """Resolves the arguments from the provided args and kwargs.

        Args:
            arg_names (list):
                List of argument names to resolve.
        """
        parameters = {}
        is_single_table = len(self.tables) == 1
        args_table_name = True
        if is_single_table:
            parameters['table_name'] = next(iter(self.tables))
            if len(arg_names) != len(args):
                args_table_name = False

        else:
            table_name = kwargs.get('table_name')
            if table_name is None:
                table_name = args[0]
                args_table_name = False

            parameters['table_name'] = table_name

        parameters.update({
            arg_name: arg for arg_name, arg in zip(arg_names, args[not args_table_name :])
        })
        for parameter_name, parameter in parameters.items():
            kwargs_value = kwargs.get(parameter_name)
            if kwargs_value is not None and kwargs_value != parameter:
                raise ValueError(
                    f"Conflicting values for '{parameter_name}': '{parameter}' and '{kwargs_value}'"
                )

        kwargs = {key: value for key, value in kwargs.items() if key not in parameters}
        parameters.update(kwargs)

        return parameters

    def get_column_names(self, *args, **kwargs):
        """Return a list of column names that match the given metadata keyword arguments."""
        parameters = self._resolve_arguments([], *args, **kwargs)
        return super().get_column_names(**parameters)

    def update_column(self, *args, **kwargs):
        """Update an existing column for a table in the ``Metadata``."""
        parameters = self._resolve_arguments(['column_name'], *args, **kwargs)
        super().update_column(**parameters)

    def update_columns(self, *args, **kwargs):
        """Update the metadata of multiple columns."""
        parameters = self._resolve_arguments(['column_names'], *args, **kwargs)
        super().update_columns(**parameters)

    def update_columns_metadata(self, *args, **kwargs):
        """Update the metadata of multiple columns."""
        parameters = self._resolve_arguments(['column_metadata'], *args, **kwargs)
        super().update_columns_metadata(**parameters)

    def add_column(self, *args, **kwargs):
        """Add a column to the metadata."""
        parameters = self._resolve_arguments(['column_name'], *args, **kwargs)
        super().add_column(**parameters)

    def add_column_relationship(self, *args, **kwargs):
        """Add a column relationship to the metadata."""
        parameters = self._resolve_arguments(['relationship_type', 'column_names'], *args, **kwargs)
        super().add_column_relationship(**parameters)

    def set_primary_key(self, *args, **kwargs):
        """Set the primary key of a table."""
        parameters = self._resolve_arguments(['column_name'], *args, **kwargs)
        super().set_primary_key(**parameters)

    def remove_primary_key(self, *args, **kwargs):
        """Remove the primary key of a table."""
        parameters = self._resolve_arguments([], *args, **kwargs)
        super().remove_primary_key(**parameters)

    def add_alternate_keys(self, *args, **kwargs):
        """Add alternate keys to a table."""
        parameters = self._resolve_arguments(['column_names'], *args, **kwargs)
        super().add_alternate_keys(**parameters)
