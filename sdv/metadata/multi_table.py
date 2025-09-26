"""Multi Table Metadata."""

import datetime
import json
import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd

from sdv._utils import _cast_to_iterable, _load_data_from_csv
from sdv.errors import InvalidDataError
from sdv.logging import get_sdv_logger
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata_upgrader import convert_metadata
from sdv.metadata.single_table import INT_REGEX_ZERO_ERROR_MESSAGE, SingleTableMetadata
from sdv.metadata.utils import _validate_file_mode, read_json, validate_file_does_not_exist
from sdv.metadata.visualization import (
    create_columns_node,
    create_summarized_columns_node,
    visualize_graph,
)

LOGGER = logging.getLogger(__name__)
MULTITABLEMETADATA_LOGGER = get_sdv_logger('MultiTableMetadata')
WARNINGS_COLUMN_ORDER = ['Table Name', 'Column Name', 'sdtype', 'datetime_format']


class MultiTableMetadata:
    """Multi Table Metadata class."""

    METADATA_SPEC_VERSION = 'MULTI_TABLE_V1'

    def __init__(self):
        self.tables = {}
        self.relationships = []
        self._multi_table_updated = False

    def _check_updated_flag(self):
        is_single_table_updated = any(table._updated for table in self.tables.values())
        if is_single_table_updated or self._multi_table_updated:
            return True

        return False

    def _reset_updated_flag(self):
        for table in self.tables.values():
            table._updated = False

        self._multi_table_updated = False

    def _validate_missing_relationship_keys(
        self, parent_table_name, parent_primary_key, child_table_name, child_foreign_key
    ):
        parent_table = self.tables.get(parent_table_name)
        child_table = self.tables.get(child_table_name)
        if parent_table.primary_key is None:
            raise InvalidMetadataError(
                f"The parent table '{parent_table_name}' does not have a primary key set. "
                "Please use 'set_primary_key' in order to set one."
            )

        missing_keys = set()
        parent_primary_key = _cast_to_iterable(parent_primary_key)
        table_primary_keys = set(_cast_to_iterable(parent_table.primary_key))
        for key in parent_primary_key:
            if key not in table_primary_keys:
                missing_keys.add(key)

        if missing_keys:
            raise InvalidMetadataError(
                f'Relationship between tables ({parent_table_name}, {child_table_name}) contains '
                f'an unknown primary key {missing_keys}.'
            )

        for key in set(_cast_to_iterable(child_foreign_key)):
            if key not in child_table.columns:
                missing_keys.add(key)

        if missing_keys:
            raise InvalidMetadataError(
                f'Relationship between tables ({parent_table_name}, {child_table_name}) '
                f'contains an unknown foreign key {missing_keys}.'
            )

    @staticmethod
    def _validate_no_missing_tables_in_relationship(parent_table_name, child_table_name, tables):
        missing_table_names = {parent_table_name, child_table_name} - set(tables)
        if missing_table_names:
            if len(missing_table_names) == 1:
                raise InvalidMetadataError(
                    f'Relationship contains an unknown table {missing_table_names}.'
                )
            else:
                raise InvalidMetadataError(
                    f'Relationship contains unknown tables {missing_table_names}.'
                )

    @staticmethod
    def _validate_relationship_key_length(
        parent_table_name, parent_primary_key, child_table_name, child_foreign_key
    ):
        pk_len = len(set(_cast_to_iterable(parent_primary_key)))
        fk_len = len(set(_cast_to_iterable(child_foreign_key)))
        if pk_len != fk_len:
            raise InvalidMetadataError(
                f"Relationship between tables ('{parent_table_name}', '{child_table_name}') is "
                f'invalid. Primary key has length {pk_len} but the foreign key has '
                f'length {fk_len}.'
            )

    def _validate_relationship_sdtypes(
        self, parent_table_name, parent_primary_key, child_table_name, child_foreign_key
    ):
        parent_table_columns = self.tables.get(parent_table_name).columns
        child_table_columns = self.tables.get(child_table_name).columns
        parent_primary_key = _cast_to_iterable(parent_primary_key)
        child_foreign_key = _cast_to_iterable(child_foreign_key)
        for pk, fk in zip(parent_primary_key, child_foreign_key):
            if parent_table_columns[pk]['sdtype'] != child_table_columns[fk]['sdtype']:
                raise InvalidMetadataError(
                    f"Relationship between tables ('{parent_table_name}', '{child_table_name}') "
                    'is invalid. The primary and foreign key columns are not the same type.'
                )

    def _validate_circular_relationships(
        self, parent, children=None, visited=None, child_map=None, errors=None
    ):
        """Validate that there is no circular relationship in the metadata."""
        visited = set() if visited is None else visited
        if children is None:
            children = child_map[parent]

        if parent in children:
            errors.append(parent)

        for child in children:
            if child in visited:
                continue

            visited.add(child)
            self._validate_circular_relationships(
                parent,
                children=child_map.get(child, set()),
                child_map=child_map,
                visited=visited,
                errors=errors,
            )

    def _validate_child_map_circular_relationship(self, child_map):
        errors = []
        for table_name in self.tables.keys():
            self._validate_circular_relationships(table_name, child_map=child_map, errors=errors)

        if errors:
            raise InvalidMetadataError(
                'The relationships in the dataset describe a circular dependency between '
                f'tables {errors}.'
            )

    def _validate_foreign_child_key(self, child_table_name, parent_table_name, child_foreign_key):
        child_primary_key = _cast_to_iterable(self.tables[child_table_name].primary_key)
        child_foreign_key = _cast_to_iterable(child_foreign_key)
        if set(child_foreign_key).intersection(set(child_primary_key)):
            raise InvalidMetadataError(
                f"Invalid relationship between table '{parent_table_name}' and table "
                f"'{child_table_name}'. A relationship must connect a primary key "
                'with a non-primary key.'
            )

    def _validate_new_foreign_key_is_not_reused(
        self, parent_table_name, parent_primary_key, child_table_name, child_foreign_key
    ):
        for relationship in self.relationships:
            foreign_key_already_used = (
                relationship['child_table_name'] == child_table_name
                and relationship['child_foreign_key'] == child_foreign_key
            )
            parent_matches = (
                relationship['parent_table_name'] == parent_table_name
                and relationship['parent_primary_key'] == parent_primary_key
            )
            if foreign_key_already_used and not parent_matches:
                raise InvalidMetadataError(
                    f'Relationship between tables ({parent_table_name}, {child_table_name}) uses '
                    f"a foreign key column ('{child_foreign_key}') that is already used in another "
                    'relationship.'
                )

    def _validate_foreign_key_uniqueness_across_relationships(
        self,
        parent_table_name,
        parent_primary_key,
        child_table_name,
        child_foreign_key,
        seen_foreign_keys,
    ):
        key = (child_table_name, child_foreign_key)
        current_relationship = (parent_table_name, parent_primary_key)

        if key in seen_foreign_keys:
            existing_relationship = seen_foreign_keys[key]
            if existing_relationship != current_relationship:
                raise InvalidMetadataError(
                    f'Relationship between tables ({parent_table_name}, {child_table_name}) uses '
                    f"a foreign key column ('{child_foreign_key}') that is already used in another "
                    'relationship.'
                )
        else:
            seen_foreign_keys[key] = current_relationship

    def _validate_relationship_does_not_exist(
        self, parent_table_name, parent_primary_key, child_table_name, child_foreign_key
    ):
        for relationship in self.relationships:
            already_exists = (
                relationship['parent_table_name'] == parent_table_name
                and relationship['parent_primary_key'] == parent_primary_key
                and relationship['child_table_name'] == child_table_name
                and relationship['child_foreign_key'] == child_foreign_key
            )
            if already_exists:
                raise InvalidMetadataError('This relationship has already been added.')

    def _validate_relationship(
        self, parent_table_name, child_table_name, parent_primary_key, child_foreign_key
    ):
        self._validate_no_missing_tables_in_relationship(
            parent_table_name, child_table_name, self.tables.keys()
        )

        self._validate_missing_relationship_keys(
            parent_table_name, parent_primary_key, child_table_name, child_foreign_key
        )
        self._validate_relationship_key_length(
            parent_table_name, parent_primary_key, child_table_name, child_foreign_key
        )

        self._validate_foreign_child_key(child_table_name, parent_table_name, child_foreign_key)

        self._validate_relationship_sdtypes(
            parent_table_name, parent_primary_key, child_table_name, child_foreign_key
        )

    def _get_parent_map(self):
        parent_map = defaultdict(set)
        for relation in self.relationships:
            parent_name = relation['parent_table_name']
            child_name = relation['child_table_name']
            parent_map[child_name].add(parent_name)

        return parent_map

    def _get_child_map(self):
        child_map = defaultdict(set)
        for relation in self.relationships:
            parent_name = relation['parent_table_name']
            child_name = relation['child_table_name']
            child_map[parent_name].add(child_name)

        return child_map

    def _get_foreign_keys(self, parent_table_name, child_table_name):
        """Get all foreign keys for the parent table."""
        foreign_keys = []
        for relation in self.relationships:
            if (
                parent_table_name == relation['parent_table_name']
                and child_table_name == relation['child_table_name']
            ):
                foreign_keys.append(deepcopy(relation['child_foreign_key']))

        return foreign_keys

    def _get_all_foreign_keys(self, table_name):
        foreign_keys = []
        for relation in self.relationships:
            if table_name == relation['child_table_name']:
                foreign_keys.append(deepcopy(relation['child_foreign_key']))

        return foreign_keys

    def _get_all_keys(self, table_name):
        foreign_keys = self._get_all_foreign_keys(table_name)
        return set(foreign_keys).union(self.tables[table_name]._get_primary_and_alternate_keys())

    def add_relationship(
        self, parent_table_name, child_table_name, parent_primary_key, child_foreign_key
    ):
        """Add a relationship between two tables.

        Args:
            parent_table_name (str):
                A string representing the name of the parent table.
            child_table_name (str):
                A string representing the name of the child table.
            parent_primary_key (str or tuple):
                A string or tuple of strings representing the primary key of the parent.
            child_foreign_key (str or tuple):
                A string or tuple of strings representing the foreign key of the child.

        Raises:
            - ``InvalidMetadataError`` if a table is missing.
            - ``InvalidMetadataError`` if the ``parent_primary_key`` or ``child_foreign_key`` are
              missing.
            - ``InvalidMetadataError`` if the ``parent_primary_key`` and ``child_foreign_key``
              have different
              size.
            - ``InvalidMetadataError`` if the ``parent_primary_key`` and ``child_foreign_key`` are
              different
              ``sdtype``.
            - ``InvalidMetadataError`` if the relationship causes a circular dependency.
            - ``InvalidMetadataError`` if ``child_foreign_key`` is a primary key.
        """
        self._validate_relationship(
            parent_table_name, child_table_name, parent_primary_key, child_foreign_key
        )
        self._validate_new_foreign_key_is_not_reused(
            parent_table_name, parent_primary_key, child_table_name, child_foreign_key
        )
        child_map = self._get_child_map()
        child_map[parent_table_name].add(child_table_name)
        self._validate_relationship_does_not_exist(
            parent_table_name, parent_primary_key, child_table_name, child_foreign_key
        )
        self._validate_child_map_circular_relationship(child_map)

        self.relationships.append({
            'parent_table_name': parent_table_name,
            'child_table_name': child_table_name,
            'parent_primary_key': deepcopy(parent_primary_key),
            'child_foreign_key': deepcopy(child_foreign_key),
        })
        self._multi_table_updated = True

    def remove_relationship(self, parent_table_name, child_table_name):
        """Remove the relationship between two tables.

        Args:
            parent_table_name (str):
                The name of the parent table.
            child_table_name (str):
                The name of the child table.
        """
        relationships_to_remove = []
        for relation in self.relationships:
            if (
                relation['parent_table_name'] == parent_table_name
                and relation['child_table_name'] == child_table_name
            ):
                relationships_to_remove.append(relation)

        if not relationships_to_remove:
            warning_msg = (
                f"No existing relationships found between parent table '{parent_table_name}' and "
                f"child table '{child_table_name}'."
            )
            warnings.warn(warning_msg)

        else:
            for relation in relationships_to_remove:
                self.relationships.remove(relation)

        self._multi_table_updated = True

    def remove_primary_key(self, table_name):
        """Remove the primary key from the given table.

        Removes the primary key from the given table. Also removes any relationships that
        reference that table's primary key, including all relationships in which the given
        table is a parent table.

        Args:
            table_name (str):
                The name of the table to remove the primary key from.
        """
        self._validate_table_exists(table_name)
        primary_key = self.tables[table_name].primary_key
        self.tables[table_name].remove_primary_key()

        for relationship in self.relationships[:]:
            parent_table = relationship['parent_table_name']
            child_table = relationship['child_table_name']
            foreign_key = relationship['child_foreign_key']
            if (
                child_table == table_name and foreign_key == primary_key
            ) or parent_table == table_name:
                other_table = child_table if parent_table == table_name else parent_table
                info_msg = (
                    f"Relationship between '{table_name}' and '{other_table}' removed because "
                    f"the primary key for '{table_name}' was removed."
                )
                LOGGER.info(info_msg)
                self.relationships.remove(relationship)

        self._multi_table_updated = True

    def _validate_table_exists(self, table_name):
        if table_name not in self.tables:
            raise InvalidMetadataError(f"Unknown table name ('{table_name}').")

    def add_column(self, table_name, column_name, **kwargs):
        """Add a column to a table in the ``MultiTableMetadata``.

        Args:
            table_name (str):
                Name of the table to add the column to.
            column_name (str):
                The column name to be added.
            **kwargs (type):
                Any additional key word arguments for the column, where ``sdtype`` is required.

        Raises:
            - ``InvalidMetadataError`` if the column already exists.
            - ``InvalidMetadataError`` if the ``kwargs`` do not contain ``sdtype``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              given ``sdtype``.
            - ``InvalidMetadataError`` if the table doesn't exist in the ``MultiTableMetadata``.
        """
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.add_column(column_name, **kwargs)

    def update_column(self, table_name, column_name, **kwargs):
        """Update an existing column for a table in the ``MultiTableMetadata``.

        Args:
            table_name (str):
                Name of table the column belongs to.
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
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.update_column(column_name, **kwargs)

    def update_columns(self, table_name, column_names, **kwargs):
        """Update multiple columns with the same metadata kwargs.

        Args:
            table_name (str):
                Name of the table to update the columns.
            column_names (list[str]):
                List of column names to update.
            **kwargs:
                Any key word arguments that describe metadata for the columns.
        """
        if not isinstance(column_names, list):
            raise InvalidMetadataError('Please pass in a list to column_names arg.')
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.update_columns(column_names, **kwargs)

    def update_columns_metadata(self, table_name, column_metadata):
        """Update the metadata of multiple columns at once.

        Args:
            table_name (str):
                Name of the table to update the columns.
            column_metadata (dict):
                Dictionary of column names and their metadata to update.
        """
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.update_columns_metadata(column_metadata)

    def add_constraints(self, table_name, constraint_name, **kwargs):
        """Add a constraint to a table in the multi-table metadata.

        Args:
            table_name (str):
                Name of the table to add the column to.
            constraint_name (string):
                Name of the constraint class.
            **kwargs:
                Any other arguments the constraint requires.
        """
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.add_constraints(constraint_name, **kwargs)

    def _validate_table_not_detected(self, table_name):
        if table_name in self.tables:
            raise InvalidMetadataError(
                f"Metadata for table '{table_name}' already exists. Specify a new table name or "
                'create a new MultiTableMetadata object for other data sources.'
            )

    @staticmethod
    def _log_detected_table(single_table_metadata):
        table_dict = single_table_metadata.to_dict()
        table_dict.pop('METADATA_SPEC_VERSION', None)
        table_json = json.dumps(table_dict, indent=4)
        LOGGER.info(f'Detected metadata:\n{table_json}')

    def _validate_all_tables_connected(self, parent_map, child_map):
        """Get the connection status of all tables.

        Args:
            parent_map (dict):
                Dictionary mapping each parent table to its child tables.
            child_map (dict):
                Dictionary mapping each child table to its parent tables.

        Returns:
            dict specifying whether each table is connected the other tables.
        """
        nodes = list(self.tables.keys())
        if len(nodes) == 1:
            return

        parent_nodes = list(parent_map.keys())
        queue = [parent_nodes[0]] if parent_map else []
        connected = {table_name: False for table_name in nodes}

        while queue:
            node = queue.pop()
            connected[node] = True
            for child in list(child_map[node]) + list(parent_map[node]):
                if not connected[child] and child not in queue:
                    queue.append(child)

        if not all(connected.values()):
            disconnected_tables = [table for table, value in connected.items() if not value]
            if len(disconnected_tables) > 1:
                table_msg = (
                    f'Tables {disconnected_tables} are not connected to any of the other tables.'
                )
            else:
                table_msg = (
                    f'Table {disconnected_tables} is not connected to any of the other tables.'
                )

            raise InvalidMetadataError(
                f'The relationships in the dataset are disjointed. {table_msg}'
            )

    def _detect_foreign_keys_by_column_name(self, data):
        """Detect the foreign keys based on if a column name matches a primary key.

        Args:
            data (dict):
                Dictionary of table names to dataframes.
                NOTE: this is only used in SDV-Enterprise.
        """
        for parent_candidate in self.tables.keys():
            primary_key = self.tables[parent_candidate].primary_key
            for child_candidate in self.tables.keys() - {parent_candidate}:
                child_meta = self.tables[child_candidate]
                if primary_key in child_meta.columns.keys():
                    try:
                        original_foreign_key_sdtype = child_meta.columns[primary_key]['sdtype']
                        if original_foreign_key_sdtype != 'id':
                            self.update_column(
                                table_name=child_candidate, column_name=primary_key, sdtype='id'
                            )

                        self.add_relationship(
                            parent_candidate, child_candidate, primary_key, primary_key
                        )
                    except InvalidMetadataError:
                        self.update_column(
                            table_name=child_candidate,
                            column_name=primary_key,
                            sdtype=original_foreign_key_sdtype,
                        )
                        continue

    def _detect_relationships(self, data=None, foreign_key_inference_algorithm='column_name_match'):
        """Automatically detect relationships between tables.

        Args:
            data (dict):
                Dictionary of table names to dataframes.
                NOTE: this is only used in SDV-Enterprise.
            foreign_key_inference_algorithm (str):
                Which algorithm to use for detecting foreign keys. Currently only one option,
                'column_name_match'.
        """
        if foreign_key_inference_algorithm == 'column_name_match':
            self._detect_foreign_keys_by_column_name(data)

    def detect_table_from_dataframe(
        self, table_name, data, infer_sdtypes=True, infer_keys='primary_only'
    ):
        """Detect the metadata for a table from a dataframe.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``,
        for a specified table. All data column names are converted to strings.

        Args:
            table_name (str):
                Name of the table to detect.
            data (pandas.DataFrame):
                ``pandas.DataFrame`` to detect the metadata from.
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
        """
        self._validate_table_not_detected(table_name)
        table = SingleTableMetadata()
        table._detect_columns(data, table_name, infer_sdtypes, infer_keys)
        self.tables[table_name] = table
        self._log_detected_table(table)

    def detect_from_dataframes(self, data):
        """Detect the metadata for all tables in a dictionary of dataframes.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.
        All data column names are converted to strings.

        Args:
            data (dict):
                Dictionary of table names to dataframes.
        """
        if not data or not all(isinstance(df, pd.DataFrame) for df in data.values()):
            raise ValueError('The provided dictionary must contain only pandas DataFrame objects.')

        for table_name, dataframe in data.items():
            self.detect_table_from_dataframe(table_name, dataframe)

        self._detect_relationships(data)

    def detect_from_csvs(self, folder_name, read_csv_parameters=None):
        """Detect the metadata for all tables in a folder of csv files.

        Args:
            folder_name (str):
                Name of the folder to detect the metadata from.
            read_csv_parameters (dict):
                A python dictionary of with string and value accepted by ``pandas.read_csv``
                function. Defaults to ``None``.
        """
        folder_path = Path(folder_name)

        if folder_path.is_dir():
            csv_files = list(folder_path.rglob('*.csv'))
        else:
            raise ValueError(f"The folder '{folder_name}' does not exist.")

        if not csv_files:
            raise ValueError(f"No CSV files detected in the folder '{folder_name}'.")

        data = {}
        for csv_file in csv_files:
            table_name = csv_file.stem
            data[table_name] = _load_data_from_csv(csv_file, read_csv_parameters)
            self.detect_table_from_dataframe(table_name, data[table_name])

        self._detect_relationships(data)

    def set_primary_key(self, table_name, column_name):
        """Set the primary key of a table.

        Args:
            table_name (str):
                Name of the table to set the primary key.
            column_name (str, tulple[str]):
                Name (or tuple of names) of the primary key column(s).
        """
        self._validate_table_exists(table_name)
        self.tables[table_name].set_primary_key(column_name)

    def set_sequence_key(self, table_name, column_name):
        """Set the sequence key of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence key.
            column_name (str, tulple[str]):
                Name (or tuple of names) of the sequence key column(s).
        """
        self._validate_table_exists(table_name)
        warnings.warn('Sequential modeling is not yet supported on SDV Multi Table models.')
        self.tables[table_name].set_sequence_key(column_name)

    def add_alternate_keys(self, table_name, column_names):
        """Set the alternate keys of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence key.
            column_names (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
        """
        self._validate_table_exists(table_name)
        self.tables[table_name].add_alternate_keys(column_names)

    def set_sequence_index(self, table_name, column_name):
        """Set the sequence index of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence index.
            column_name (str):
                Name of the sequence index column.
        """
        self._validate_table_exists(table_name)
        warnings.warn('Sequential modeling is not yet supported on SDV Multi Table models.')
        self.tables[table_name].set_sequence_index(column_name)

    def _validate_column_relationships_foreign_keys(self, table_column_relationships, foreign_keys):
        """Validate that a table's column relationships do not use any foreign keys.

        Args:
            table_column_relationships (list[dict]):
                The list of column relationships for the table.
            foreign_keys (list):
                The list of foreign keys in the table.

        Raises:
            - ``InvalidMetadataError`` if foreign keys are used in any column relationships.
        """
        for column_relationship in table_column_relationships:
            column_names = set(column_relationship.get('column_names', []))
            invalid_columns = column_names.intersection(foreign_keys)
            if invalid_columns:
                raise InvalidMetadataError(
                    f'Cannot use foreign keys {invalid_columns} in column relationship.'
                )

    def add_column_relationship(self, table_name, relationship_type, column_names):
        """Add a column relationship to a table in the metadata.

        Args:
            table_name (str):
                The name of the table to add this relationship to.
            relationship_type (str):
                The type of the relationship.
            column_names (list[str]):
                The list of column names involved in this relationship.
        """
        self._validate_table_exists(table_name)
        foreign_keys = self._get_all_foreign_keys(table_name)
        relationships = [{'type': relationship_type, 'column_names': column_names}] + self.tables[
            table_name
        ].column_relationships
        self._validate_column_relationships_foreign_keys(relationships, foreign_keys)
        self.tables[table_name].add_column_relationship(relationship_type, column_names)

    def _validate_single_table(self, errors):
        foreign_key_cols = defaultdict(list)
        for relationship in self.relationships:
            child_table = relationship.get('child_table_name')
            child_foreign_key = relationship.get('child_foreign_key')
            foreign_key_cols[child_table].append(child_foreign_key)

        for table_name, table in self.tables.items():
            if len(table.columns) == 0:
                error_message = (
                    f"Table '{table_name}' has 0 columns. Use 'add_column' to specify its columns."
                )
                errors.append(error_message)
            try:
                table.validate()

            except Exception as error:
                errors.append('\n')
                title = f'Table: {table_name}'
                error = str(error).replace(
                    'The following errors were found in the metadata:\n', title
                )
                errors.append(error)

            try:
                self._validate_column_relationships_foreign_keys(
                    table.column_relationships, foreign_key_cols[table_name]
                )
            except Exception as col_relationship_error:
                errors.append(str(col_relationship_error))

    def _append_relationships_errors(self, errors, method, *args, **kwargs):
        try:
            method(*args, **kwargs)
        except Exception as error:
            if '\nRelationships:' not in errors:
                errors.append('\nRelationships:')

            errors.append(error)

    def validate(self):
        """Validate the metadata.

        Raises:
            - ``InvalidMetadataError`` if the metadata is invalid.
        """
        errors = []
        self._validate_single_table(errors)
        seen_foreign_keys = {}
        for relation in self.relationships:
            self._append_relationships_errors(errors, self._validate_relationship, **relation)
            self._append_relationships_errors(
                errors,
                self._validate_foreign_key_uniqueness_across_relationships,
                **relation,
                seen_foreign_keys=seen_foreign_keys,
            )

        child_map = self._get_child_map()

        self._append_relationships_errors(
            errors, self._validate_child_map_circular_relationship, child_map
        )
        if errors:
            raise InvalidMetadataError(
                'The metadata is not valid' + '\n'.join(str(e) for e in errors)
            )

    def _validate_missing_tables(self, data):
        """Validate the data doesn't have all the columns in the metadata."""
        errors = []
        missing_tables = set(self.tables) - set(data)
        if missing_tables:
            errors.append(f'The provided data is missing the tables {missing_tables}.')

        return errors

    def _validate_all_tables(self, data):
        """Validate every table of the data has a valid table/metadata pair."""
        errors = []
        warning_dataframes = []
        for table_name, table_data in data.items():
            table_sdtype_warnings = defaultdict(list)
            try:
                with warnings.catch_warnings(record=True):
                    self.tables[table_name].validate_data(table_data, table_sdtype_warnings)

            except InvalidDataError as error:
                if INT_REGEX_ZERO_ERROR_MESSAGE in str(error) and len(self.tables) > 1:
                    raise InvalidDataError([
                        f'Primary key for table "{table_name}" {INT_REGEX_ZERO_ERROR_MESSAGE}'
                    ])

                error_msg = f'Errors in {table_name}:'
                for _error in error.errors:
                    error_msg += f'\nError: {_error}'

                errors.append(error_msg)

            except ValueError as error:
                errors.append(str(error))

            except KeyError:
                continue

            finally:
                if table_sdtype_warnings:
                    table_sdtype_warnings['Table Name'].extend(
                        [table_name] * len(table_sdtype_warnings['Column Name'])
                    )
                    df = pd.DataFrame(table_sdtype_warnings, columns=WARNINGS_COLUMN_ORDER)
                    warning_dataframes.append(df)

        if warning_dataframes:
            warning_df = pd.concat(warning_dataframes)
            warning_msg = (
                "No 'datetime_format' is present in the metadata for the following columns:\n "
                f'{warning_df.to_string(index=False)}\n'
                'Without this specification, SDV may not be able to accurately parse the data. '
                "We recommend adding datetime formats using 'update_column'."
            )
            warnings.warn(warning_msg)

        return errors

    def _validate_foreign_keys(self, data):
        """Validate all foreign key relationships."""
        error_msg = None
        errors = []
        for relation in self.relationships:
            child_table = data.get(relation['child_table_name'])
            parent_table = data.get(relation['parent_table_name'])

            if isinstance(child_table, pd.DataFrame) and isinstance(parent_table, pd.DataFrame):
                child_column = child_table[relation['child_foreign_key']]
                parent_column = parent_table[relation['parent_primary_key']]
                missing_values = child_column[~child_column.isin(parent_column)].unique()
                missing_values = missing_values[~pd.isna(missing_values)]

                if any(missing_values):
                    message = ', '.join(missing_values[:5].astype(str))
                    if len(missing_values) > 5:
                        message = f'({message}, + more)'
                    else:
                        message = f'({message})'

                    errors.append(
                        f"Error: foreign key column '{relation['child_foreign_key']}' contains "
                        f'unknown references: {message}. Please use the method'
                        " 'drop_unknown_references' from sdv.utils to clean the data."
                    )

            if errors:
                error_msg = 'Relationships:\n'
                error_msg += '\n'.join(errors)

        return [error_msg] if error_msg else []

    def _validate_data(self, data, table_name=None):
        """Validate the given data matches the metadata.

        Checks the following rules:
            * every table of the data satisfies its own metadata
            * if no table_name provided, all tables of the metadata are present in the data
            * if no table_name provided, that all foreign keys belong to a primay key

        Args:
            data (dict):
                A dictionary of table names to pd.DataFrames.
            table_name (str, optional):
                The specific table to validate. If set, only validates the data for the
                table. If None, validates the data for all tables. Defaults to None.

        Raises:
            InvalidDataError:
                This error is being raised if the data is not matching its sdtype requirements.

        Warns:
            A warning is being raised if ``datetime_format`` is missing from a column represented
            as ``object`` in the dataframe and its sdtype is ``datetime``.
        """
        if not isinstance(data, dict):
            raise InvalidMetadataError('Please pass in a dictionary mapping tables to dataframes.')

        errors = []
        errors += self._validate_missing_tables(data) if not table_name else []
        errors += self._validate_all_tables(data)
        errors += self._validate_foreign_keys(data) if not table_name else []

        if errors:
            raise InvalidDataError(errors)

    def validate_data(self, data):
        """Validate the data matches the metadata.

        Checks the following rules:
            * every table of the data satisfies its own metadata
            * all tables of the metadata are present in the data
            * all foreign keys belong to a primay key

        Args:
            data (dict):
                A dictionary of table names to pd.DataFrames.

        Raises:
            InvalidDataError:
                This error is being raised if the data is not matching its sdtype requirements.

        Warns:
            A warning is being raised if ``datetime_format`` is missing from a column represented
            as ``object`` in the dataframe and its sdtype is ``datetime``.
        """
        self._validate_data(data)

    def add_table(self, table_name):
        """Add a table to the metadata.

        Args:
            table_name (str):
                The name of the table to add to the metadata.

        Raises:
            Raises ``InvalidMetadataError`` if ``table_name`` is not valid.
        """
        if not isinstance(table_name, str) or table_name == '':
            raise InvalidMetadataError(
                "Invalid table name (''). The table name must be a non-empty string."
            )

        if table_name in self.tables:
            raise InvalidMetadataError(
                f"Cannot add a table named '{table_name}' because it already exists in the "
                'metadata. Please choose a different name.'
            )

        self.tables[table_name] = SingleTableMetadata()
        self._multi_table_updated = True

    def get_column_names(self, table_name, **kwargs):
        """Return a list of columns from the given table that match the metadata keyword arguments.

        Args:
            table_name (str):
                The name of the table to get column names for.
            **kwargs:
                Metadata keywords to filter on, for example sdtype='id' or pii=True.

        Returns:
            list:
                The list of columns that match the metadata kwargs for the given table.
        """
        self._validate_table_exists(table_name)
        return self.tables[table_name].get_column_names(**kwargs)

    def get_table_metadata(self, table_name):
        """Return the metadata for a table.

        Args:
            table_name (str):
                The name of the table to get the metadata for.

        Returns:
            SingleTableMetadata:
                The metadata for the given table.
        """
        self._validate_table_exists(table_name)
        return deepcopy(self.tables[table_name])

    def _get_anonymized_dict(self):
        anonymized_metadata = {'tables': {}, 'relationships': []}
        anonymized_table_map = {}
        counter = 1
        for table, table_metadata in self.tables.items():
            anonymized_table_name = f'table{counter}'
            anonymized_table_map[table] = anonymized_table_name

            anonymized_metadata['tables'][anonymized_table_name] = (
                table_metadata.anonymize().to_dict()
            )
            counter += 1

        for relationship in self.relationships:
            parent_table = relationship['parent_table_name']
            anonymized_parent_table = anonymized_table_map[parent_table]

            child_table = relationship['child_table_name']
            anonymized_child_table = anonymized_table_map[child_table]

            foreign_key = relationship['child_foreign_key']
            anonymized_foreign_key = self.tables[child_table]._anonymized_column_map[foreign_key]

            primary_key = relationship['parent_primary_key']
            anonymized_primary_key = self.tables[parent_table]._anonymized_column_map[primary_key]

            anonymized_metadata['relationships'].append({
                'parent_table_name': anonymized_parent_table,
                'child_table_name': anonymized_child_table,
                'child_foreign_key': anonymized_foreign_key,
                'parent_primary_key': anonymized_primary_key,
            })

        return anonymized_metadata

    def anonymize(self):
        """Anonymize metadata by obfuscating column names.

        Returns:
            MultiTableMetadata:
                An anonymized MultiTableMetadata instance.
        """
        anonymized_metadata = self._get_anonymized_dict()

        return MultiTableMetadata.load_from_dict(anonymized_metadata)

    def _get_table_info(self, table_name, show_table_details):
        node_info = {}
        table_meta = self.tables[table_name]

        if show_table_details in ['full', 'summarized']:
            node_info['primary_key'] = f'Primary key: {table_meta.primary_key}'
            if table_meta.sequence_key:
                node_info['sequence_key'] = f'Sequence key: {table_meta.sequence_key}'
            if table_meta.sequence_index:
                node_info['sequence_index'] = f'Sequence index: {table_meta.sequence_index}'

        if show_table_details == 'full':
            node_info['columns'] = create_columns_node(table_meta.columns)
        elif show_table_details == 'summarized':
            node_info['columns'] = create_summarized_columns_node(table_meta.columns)
        elif show_table_details is None:
            return

        return node_info

    def visualize(
        self, show_table_details='full', show_relationship_labels=True, output_filepath=None
    ):
        """Create a visualization of the multi-table dataset.

        Args:
            show_table_details (str or None):
                If 'full', the column names, primary and foreign keys are all shown along with
                the table names. If 'summarized', primary and foreign keys are shown and a count
                of the different sdtypes is shown. If None only the table names are shown. Defaults
                to 'full'.
            show_relationship_labels (bool):
                If True, every edge is labeled with the column names (eg. purchaser_id -> user_id).
                Defaults to True.
            output_filepath (str):
                Full path of where to save the visualization. If None, the visualization is not
                saved. Defaults to None.

        Returns:
            ``graphviz.Digraph`` object.
        """
        if show_table_details not in (None, True, False, 'full', 'summarized'):
            raise ValueError(
                "'show_table_details' parameter should be 'full', 'summarized' or None."
            )

        if isinstance(show_table_details, bool):
            if show_table_details:
                future_warning_msg = (
                    'Using True or False for show_table_details is deprecated. Use '
                    "show_table_details='full' to show all table details."
                )
                show_table_details = 'full'

            else:
                future_warning_msg = (
                    "Using True or False for 'show_table_details' is deprecated. "
                    'Use show_table_details=None to hide table details.'
                )
                show_table_details = None

            warnings.warn(future_warning_msg, FutureWarning)

        nodes = {}
        edges = []

        for table_name in self.tables.keys():
            nodes[table_name] = self._get_table_info(table_name, show_table_details)

        for relationship in self.relationships:
            parent = relationship.get('parent_table_name')
            child = relationship.get('child_table_name')
            foreign_key = relationship.get('child_foreign_key')
            primary_key = self.tables.get(parent).primary_key
            edge_label = f'  {foreign_key} → {primary_key}' if show_relationship_labels else ''
            edges.append((parent, child, edge_label))

            if show_table_details is not None:
                child_node = nodes.get(child)
                foreign_key_text = f'Foreign key ({parent}): {foreign_key}'
                if 'foreign_keys' in child_node:
                    child_node.get('foreign_keys').append(foreign_key_text)
                else:
                    child_node['foreign_keys'] = [foreign_key_text]

        for table, info in nodes.items():
            if show_table_details:
                foreign_keys = r'\l'.join(info.get('foreign_keys', []))
                keys = r'\l'.join(
                    filter(
                        bool,
                        [
                            info.get('primary_key'),
                            info.get('sequence_key'),
                            info.get('sequence_index'),
                            foreign_keys,
                        ],
                    )
                )
                label = rf'{{{table}|{info["columns"]}\l|{keys}\l}}'

            else:
                label = f'{table}'

            nodes[table] = label

        return visualize_graph(nodes, edges, output_filepath)

    def to_dict(self):
        """Return a python ``dict`` representation of the ``MultiTableMetadata``."""
        metadata = {'tables': {}, 'relationships': []}
        for table_name, single_table_metadata in self.tables.items():
            table_dict = single_table_metadata.to_dict()
            table_dict.pop('METADATA_SPEC_VERSION', None)
            metadata['tables'][table_name] = table_dict

        metadata['relationships'] = deepcopy(self.relationships)
        metadata['METADATA_SPEC_VERSION'] = self.METADATA_SPEC_VERSION
        return metadata

    def _set_metadata_dict(self, metadata):
        """Set a ``metadata`` dictionary to the current instance.

        Args:
            metadata (dict):
                Python dictionary representing a ``MultiTableMetadata`` object.
        """
        for table_name, table_dict in metadata.get('tables', {}).items():
            try:
                self.tables[table_name] = SingleTableMetadata.load_from_dict(table_dict)
            except ValueError as error:
                raise ValueError(
                    f"Invalid metadata dict for table '{table_name}':\n {str(error)}"
                ) from error

        for relationship in metadata.get('relationships', []):
            type_safe_relationships = {
                key: str(value) if not isinstance(value, str) else value
                for key, value in relationship.items()
            }
            self.relationships.append(type_safe_relationships)

    def _valdiate_no_extra_keys_metadata_dict(self, metadata_dict):
        """Validate that the metadata dictionary does not contain extra keys."""
        expected_keys = {'tables', 'relationships', 'METADATA_SPEC_VERSION'}
        extra_keys = set(metadata_dict.keys()) - expected_keys
        if extra_keys:
            extra_keys = "', '".join(sorted(extra_keys))
            valid_keys = "', '".join(sorted(expected_keys))
            raise ValueError(
                f"The metadata dictionary contains extra keys: '{extra_keys}'. "
                f"Valid keys are: '{valid_keys}'."
            )

    @classmethod
    def load_from_dict(cls, metadata_dict):
        """Create a ``MultiTableMetadata`` instance from a python ``dict``.

        Args:
            metadata_dict (dict):
                Python dictionary representing a ``MultiTableMetadata`` object.

        Returns:
            Instance of ``MultiTableMetadata``.
        """
        instance = cls()
        instance._valdiate_no_extra_keys_metadata_dict(metadata_dict)
        instance._set_metadata_dict(metadata_dict)
        return instance

    def save_to_json(self, filepath, mode='write'):
        """Save the current ``MultiTableMetadata`` in to a ``json`` file.

        Args:
            filepath (str):
                String that represent the ``path`` to the ``json`` file to be written.
            mode (str):
                String that determines the mode of the function. Defaults to ``write``.
                'write' mode will create and write a file if it does not exist.
                'overwrite' mode will overwrite a file if that file does exist.

        Raises:
            Raises a ``ValueError`` if the path already exists and the mode is 'write'.
        """
        _validate_file_mode(mode)
        if mode == 'write':
            validate_file_does_not_exist(filepath)
        metadata = self.to_dict()
        total_columns = 0
        for table in self.tables.values():
            total_columns += len(table.columns)

        MULTITABLEMETADATA_LOGGER.info(
            '\nMetadata Save:\n'
            '  Timestamp: %s\n'
            '  Statistics about the metadata:\n'
            '    Total number of tables: %s\n'
            '    Total number of columns: %s\n'
            '    Total number of relationships: %s',
            datetime.datetime.now(),
            len(self.tables),
            total_columns,
            len(self.relationships),
        )
        with open(filepath, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

        self._reset_updated_flag()

    @classmethod
    def load_from_json(cls, filepath):
        """Create a ``MultiTableMetadata`` instance from a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file.

        Raises:
            - An ``Error`` if the path does not exist.
            - An ``Error`` if the ``json`` file does not contain the ``METADATA_SPEC_VERSION``.

        Returns:
            A ``MultiTableMetadata`` instance.
        """
        metadata = read_json(filepath)
        return cls.load_from_dict(metadata)

    def __repr__(self):
        """Pretty print the ``MultiTableMetadata``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed

    @classmethod
    def _convert_foreign_keys(cls, old_metadata, parent, child):
        foreign_keys = []
        child_table = old_metadata.get('tables', {}).get(child, {})
        for name, field in child_table.get('fields').items():
            ref = field.get('ref')
            if ref and ref['table'] == parent:
                foreign_keys.append(name)

        return foreign_keys

    @classmethod
    def _convert_relationships(cls, old_metadata):
        tables = old_metadata.get('tables')
        parents = defaultdict(set)
        for table, table_meta in tables.items():
            for field_meta in table_meta['fields'].values():
                ref = field_meta.get('ref')
                if ref:
                    parent = ref['table']
                    parents[table].add(parent)

        relationships = [
            {
                'parent_table_name': parent,
                'parent_primary_key': tables.get(parent).get('primary_key'),
                'child_table_name': table,
                'child_foreign_key': foreign_key,
            }
            for table in tables
            for parent in list(parents[table])
            for foreign_key in cls._convert_foreign_keys(old_metadata, parent, table)
        ]
        return relationships

    @classmethod
    def upgrade_metadata(cls, filepath):
        """Upgrade an old metadata file to the ``V1`` schema.

        Args:
            filepath (str):
                String that represents the ``path`` to the old metadata ``json`` file.

        Raises:
            Raises a ``ValueError`` if the filepath does not exist.

        Returns:
            A ``MultiTableMetadata`` instance.
        """
        old_metadata = read_json(filepath)
        tables_metadata = {}

        for table_name, metadata in old_metadata.get('tables', {}).items():
            tables_metadata[table_name] = convert_metadata(metadata)

        relationships = cls._convert_relationships(old_metadata)
        metadata_dict = {
            'tables': tables_metadata,
            'relationships': relationships,
            'METADATA_SPEC_VERSION': cls.METADATA_SPEC_VERSION,
        }
        metadata = cls.load_from_dict(metadata_dict)

        try:
            metadata.validate()
        except InvalidMetadataError as error:
            message = (
                'Successfully converted the old metadata, but the metadata was not valid.'
                f'To use this with the SDV, please fix the following errors.\n {str(error)}'
            )
            warnings.warn(message)

        return metadata
