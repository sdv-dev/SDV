"""Metadata."""

import datetime
import json
import logging
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd

from sdv._utils import (
    _cast_to_iterable,
    _format_invalid_values_string,
    _get_max_child_depth,
    _get_root_tables,
    _get_unreferenced_keys,
    _is_datetime_type,
    _is_numerical,
    _load_data_from_csv,
    _validate_boolean_parameter,
)
from sdv.errors import InvalidDataError
from sdv.logging import get_sdv_logger
from sdv.metadata._single_table import INT_REGEX_ZERO_ERROR_MESSAGE, _SingleTableMetadata
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata_upgrader import convert_metadata
from sdv.metadata.utils import _validate_file_mode, read_json, validate_file_does_not_exist
from sdv.metadata.visualization import (
    create_columns_node,
    create_summarized_columns_node,
    visualize_graph,
)

LOGGER = logging.getLogger(__name__)
METADATA_LOGGER = get_sdv_logger('Metadata')
WARNINGS_COLUMN_ORDER = ['Table Name', 'Column Name', 'sdtype', 'datetime_format']


class Metadata:
    """Metadata class that handles all metadata."""

    METADATA_SPEC_VERSION = 'V2'
    DEFAULT_SINGLE_TABLE_NAME = 'table'

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

        parent_primary_key = _cast_to_iterable(parent_primary_key)
        table_primary_keys = set(_cast_to_iterable(parent_table.primary_key))
        if set(parent_primary_key) != table_primary_keys:
            raise InvalidMetadataError(
                f'Relationship between tables ({parent_table_name}, {child_table_name}) '
                f'has a mismatched primary key {sorted(parent_primary_key)}.'
            )

        missing_fk = set()
        for key in set(_cast_to_iterable(child_foreign_key)):
            if key not in child_table.columns:
                missing_fk.add(key)

        if missing_fk:
            raise InvalidMetadataError(
                f'Relationship between tables ({parent_table_name}, {child_table_name}) '
                f'contains an unknown foreign key {missing_fk}.'
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

    def _validate_foreign_key_range_info(self, child_table_name, child_foreign_key):
        child_table_columns = self.tables[child_table_name].columns
        key = _cast_to_iterable(child_foreign_key)
        range_keys = {'range_min', 'range_max', 'range_values'}
        invalid_keys = []
        for key_col in key:
            key_metadata = child_table_columns[key_col]
            if range_keys.intersection(set(key_metadata.keys())):
                invalid_keys.append(key_col)

        if invalid_keys:
            raise InvalidMetadataError(
                f"Foreign key column(s) {invalid_keys} in table '{child_table_name}' "
                'cannot contain range information. Only `range_is_nullable` '
                'is allowed for foreign key columns.'
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
                child_foreign_key = (
                    f"('{child_foreign_key}')"
                    if isinstance(child_foreign_key, str)
                    else f'({child_foreign_key})'
                )
                raise InvalidMetadataError(
                    f'Relationship between tables ({parent_table_name}, {child_table_name}) uses '
                    f'a foreign key {child_foreign_key} that is already used in another '
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
        key = (
            tuple(_cast_to_iterable(child_table_name)),
            tuple(_cast_to_iterable(child_foreign_key)),
        )
        current_relationship = (parent_table_name, parent_primary_key)

        if key in seen_foreign_keys:
            existing_relationship = seen_foreign_keys[key]
            if existing_relationship != current_relationship:
                child_foreign_key = (
                    f"('{child_foreign_key}')"
                    if isinstance(child_foreign_key, str)
                    else f'({child_foreign_key})'
                )
                raise InvalidMetadataError(
                    f'Relationship between tables ({parent_table_name}, {child_table_name}) uses '
                    f'a foreign key {child_foreign_key} that is already used in another '
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

        self._validate_relationship_sdtypes(
            parent_table_name, parent_primary_key, child_table_name, child_foreign_key
        )

        self._validate_foreign_key_range_info(child_table_name, child_foreign_key)

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

    def _get_max_schema_depth(self):
        """Calculate the maximum depth of this schema.

        This method traverses all relationships and returns the length of the longest relationship
        chain between tables.

        Returns:
            int:
                The maximum depth of the schema.
        """
        max_depth = 1
        child_map = self._get_child_map()
        for root_table in _get_root_tables(self.relationships):
            root_depth = _get_max_child_depth(child_map, root_table)
            max_depth = root_depth if root_depth > max_depth else max_depth

        return max_depth

    def add_relationship(
        self, parent_table_name, child_table_name, parent_primary_key, child_foreign_key
    ):
        """Add a relationship between two tables.

        Args:
            parent_table_name (str):
                A string representing the name of the parent table.
            child_table_name (str):
                A string representing the name of the child table.
            parent_primary_key (str or list[str]):
                A string or list of strings representing the primary key of the parent.
            child_foreign_key (str or list[str]):
                A string or list of strings representing the foreign key of the child.

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

    def remove_primary_key(self, table_name=None):
        """Remove the primary key from the given table.

        Removes the primary key from the given table. Also removes any relationships that
        reference that table's primary key, including all relationships in which the given
        table is a parent table.

        Args:
            table_name (str, optional):
                The name of the table to remove the primary key from.
        """
        table_name = self._handle_table_name(table_name)
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

    def _remove_relationships_by_table(self, element, keys):
        """Remove relationships where the element matches the keys to check."""
        updated_relationships = []
        for relationship in self.relationships:
            matching_keys = [relationship[key] for key in keys]
            if element not in matching_keys:
                updated_relationships.append(relationship)

        self.relationships = updated_relationships

    def _remove_relationships_by_column(self, table_name, column_name):
        """Remove relationships where the column is a key for the given table."""
        updated_relationships = []
        for relationship in self.relationships:
            should_remove = (
                relationship['child_foreign_key'] == column_name
                and relationship['child_table_name'] == table_name
            ) or (
                relationship['parent_primary_key'] == column_name
                and relationship['parent_table_name'] == table_name
            )
            if not should_remove:
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
        self._remove_relationships_by_table(table_name, ['parent_table_name', 'child_table_name'])
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
        self._remove_relationships_by_column(table_name, column_name)
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

    def _validate_table_exists(self, table_name):
        if table_name not in self.tables:
            raise InvalidMetadataError(f"Unknown table name ('{table_name}').")

    def add_column(self, column_name, table_name=None, **kwargs):
        """Add a column to a table in the ``Metadata``.

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
            - ``InvalidMetadataError`` if the table doesn't exist in the ``Metadata``.
        """
        table_name = self._handle_table_name(table_name)
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.add_column(column_name, **kwargs)

    def update_column(self, column_name, table_name=None, **kwargs):
        """Update an existing column for a table in the ``Metadata``.

        Args:
            column_name (str):
                The column name to be updated.
            table_name (str, optional):
                Name of table the column belongs to.
            **kwargs (type):
                Any key word arguments that describe metadata for the column.

        Raises:
            - ``InvalidMetadataError`` if the column doesn't already exist in the
              ``_SingleTableMetadata``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              current ``sdtype``.
            - ``InvalidMetadataError`` if the table doesn't exist in the ``Metadata``.
        """
        table_name = self._handle_table_name(table_name)
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.update_column(column_name, **kwargs)

    def update_columns(self, column_names, table_name=None, **kwargs):
        """Update multiple columns with the same metadata kwargs.

        Args:
            table_name (str):
                Name of the table to update the columns.
            column_names (list[str]):
                List of column names to update.
            **kwargs:
                Any key word arguments that describe metadata for the columns.
        """
        table_name = self._handle_table_name(table_name)
        if not isinstance(column_names, list):
            raise InvalidMetadataError('Please pass in a list to column_names arg.')
        self._validate_table_exists(table_name)
        table = self.tables.get(table_name)
        table.update_columns(column_names, **kwargs)

    def update_columns_metadata(self, column_metadata, table_name=None):
        """Update the metadata of multiple columns at once.

        Args:
            column_metadata (dict):
                Dictionary of column names and their metadata to update.
            table_name (str, optional):
                Name of the table to update the columns.
        """
        table_name = self._handle_table_name(table_name)
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
                'create a new Metadata object for other data sources.'
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

    def _detect_foreign_keys_by_column_name(self, data, verbose=False):
        """Detect the foreign keys based on if a column name matches a primary key.

        If a column name (a child table) is a primary key, it will also be considered
        to be a valid candidate for a foreign key.

        Args:
            data (dict):
                Dictionary of table names to dataframes.
                NOTE: this is only used in SDV-Enterprise.
            verbose (bool):
                A boolean that determines if information should be printed regarding detection.
                If True, it prints out information about what is detected.
                If False, it does not print out any information about what is detected.
                Defaults to False.
        """
        is_foreign_keys_found = False
        if verbose:
            sys.stdout.write('\nDetecting foreign keys:\n')
        for parent_candidate in self.tables.keys():
            primary_key = self.tables[parent_candidate].primary_key
            if primary_key is None:
                continue

            pk_sdtype = self.tables[parent_candidate].columns[primary_key]['sdtype']
            for child_candidate in self.tables.keys() - {parent_candidate}:
                child_meta = self.tables[child_candidate]
                if primary_key in child_meta.columns.keys():
                    original_fk_meta = deepcopy(child_meta.columns[primary_key])
                    original_fk_sdtype = original_fk_meta['sdtype']
                    if pk_sdtype != 'id' and original_fk_sdtype != pk_sdtype:
                        continue

                    try:
                        sdtype_updated = False
                        if pk_sdtype == 'id' and original_fk_sdtype != 'id':
                            self.update_column(
                                table_name=child_candidate,
                                column_name=primary_key,
                                sdtype='id',
                            )
                            sdtype_updated = True
                        self.add_relationship(
                            parent_candidate, child_candidate, primary_key, primary_key
                        )
                        is_foreign_keys_found = True
                        if verbose:
                            child_col = f"'{child_candidate}.{primary_key}'"
                            parent_col = f"'{parent_candidate}.{primary_key}'"
                            suffix = " (updating sdtype to 'id')" if sdtype_updated else ''
                            sys.stdout.write(
                                f'- Column {child_col} refers to column {parent_col}{suffix}\n'
                            )

                    except InvalidMetadataError:
                        # circular relationship
                        if pk_sdtype == 'id' and original_fk_sdtype != 'id':
                            self.update_column(
                                table_name=child_candidate,
                                column_name=primary_key,
                                **original_fk_meta,
                            )
                        continue
        if verbose and not is_foreign_keys_found:
            sys.stdout.write('- No foreign keys found\n')

    def _detect_relationships(
        self, data=None, foreign_key_inference_algorithm='column_name_match', verbose=False
    ):
        """Automatically detect relationships between tables.

        Args:
            data (dict):
                Dictionary of table names to dataframes.
                NOTE: this is only used in SDV-Enterprise.
            foreign_key_inference_algorithm (str):
                Which algorithm to use for detecting foreign keys. Currently only one option,
                'column_name_match'.
            verbose (bool):
                A boolean that determines if information should be printed regarding detection.
                If True, it prints out information about what is detected.
                If False, it does not print out any information about what is detected.
                Defaults to False.
        """
        if foreign_key_inference_algorithm == 'column_name_match':
            self._detect_foreign_keys_by_column_name(data, verbose)

    def detect_table_from_dataframe(
        self,
        table_name,
        data,
        infer_sdtypes=True,
        infer_keys='primary_only',
        verbose=False,
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
            verbose (bool):
                A boolean that determines if information should be printed regarding detection.
                If True, it prints out information about what is detected.
                If False, it does not print out any information about what is detected.
                Defaults to False.
        """
        self._validate_table_not_detected(table_name)
        table = _SingleTableMetadata()
        table._detect_columns(data, table_name, infer_sdtypes, infer_keys, verbose)
        self.tables[table_name] = table
        self._log_detected_table(table)

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
        verbose=False,
    ):
        if not data or not all(isinstance(df, pd.DataFrame) for df in data.values()):
            raise ValueError('The provided dictionary must contain only pandas DataFrame objects.')
        if infer_keys not in ['primary_and_foreign', 'primary_only', None]:
            raise ValueError(
                "'infer_keys' must be one of: 'primary_and_foreign', 'primary_only', None."
            )
        cls._validate_foreign_key_inference_algorithm(foreign_key_inference_algorithm)
        _validate_boolean_parameter(infer_sdtypes, 'infer_sdtypes')

        metadata = Metadata()
        for table_name, dataframe in data.items():
            metadata.detect_table_from_dataframe(
                table_name,
                dataframe,
                infer_sdtypes,
                None if infer_keys is None else 'primary_only',
                verbose,
            )

        if infer_keys == 'primary_and_foreign':
            metadata._detect_relationships(data, foreign_key_inference_algorithm, verbose)

        return metadata

    @classmethod
    def detect_from_dataframes(
        cls,
        data,
        infer_sdtypes=True,
        infer_keys='primary_and_foreign',
        foreign_key_inference_algorithm='column_name_match',
        verbose=False,
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
            verbose (bool):
                A boolean that determines if information should be printed regarding detection.
                If True, it prints out information about what is detected.
                If False, it does not print out any information about what is detected.
                Defaults to False.

        Returns:
            Metadata:
                A new metadata object with the sdtypes detected from the data.
        """
        return cls._detect_from_dataframes(
            data=data,
            infer_sdtypes=infer_sdtypes,
            infer_keys=infer_keys,
            foreign_key_inference_algorithm=foreign_key_inference_algorithm,
            verbose=verbose,
        )

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

    @classmethod
    def _detect_from_dataframe(
        cls, data, table_name=None, infer_sdtypes=True, infer_keys='primary_only', verbose=False
    ):
        """Detect the metadata for a DataFrame."""
        table_name = table_name or cls.DEFAULT_SINGLE_TABLE_NAME
        if not isinstance(data, pd.DataFrame):
            raise ValueError('The provided data must be a pandas DataFrame object.')
        if infer_keys not in ['primary_only', None]:
            raise ValueError("'infer_keys' must be one of: 'primary_only', None.")

        _validate_boolean_parameter(infer_sdtypes, 'infer_sdtypes')
        metadata = Metadata()
        metadata.detect_table_from_dataframe(table_name, data, infer_sdtypes, infer_keys, verbose)
        return metadata

    @classmethod
    def detect_from_dataframe(
        cls,
        data,
        table_name=DEFAULT_SINGLE_TABLE_NAME,
        infer_sdtypes=True,
        infer_keys='primary_only',
        verbose=False,
    ):
        """Detect the metadata for a DataFrame.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.
        All data column names are converted to strings.

        Args:
            data (pandas.DataFrame):
                The data to detect metadata from.
            table_name (str):
                The name of the table to detect. If None, a default name will be used.
                Defaults to None.
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
            verbose (bool):
                A boolean that determines if information should be printed regarding detection.
                If True, it prints out information about what is detected.
                If False, it does not print out any information about what is detected.
                Defaults to False.

        Returns:
            Metadata:
                A new metadata object with the sdtypes detected from the data.
        """
        return cls._detect_from_dataframe(
            data=data,
            table_name=table_name,
            infer_sdtypes=infer_sdtypes,
            infer_keys=infer_keys,
            verbose=verbose,
        )

    def set_primary_key(self, column_name, table_name=None):
        """Set the primary key of a table.

        Args:
            column_name (str, list[str]):
                Name (or list of names) of the primary key column(s).
            table_name (str, optional):
                Name of the table to set the primary key.
        """
        table_name = self._handle_table_name(table_name)
        self._validate_table_exists(table_name)
        self.tables[table_name].set_primary_key(column_name)

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

    def add_alternate_keys(self, column_names, table_name=None):
        """Set the alternate keys of a table.

        Args:
            column_names (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
            table_name (str, optional):
                Name of the table to set the sequence key.
        """
        table_name = self._handle_table_name(table_name)
        self._validate_table_exists(table_name)
        self.tables[table_name].add_alternate_keys(column_names)

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

    def add_column_relationship(
        self,
        relationship_type,
        column_names,
        table_name=None,
    ):
        """Add a column relationship to a table in the metadata.

        Args:
            relationship_type (str):
                The type of the relationship.
            column_names (list[str]):
                The list of column names involved in this relationship.
            table_name (str, optional):
                The name of the table to add this relationship to.
        """
        table_name = self._handle_table_name(table_name)
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

        self._validate_table_exists(table_name)
        return self._validate_data({table_name: data}, table_name)

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
            error_message = '\n'.join(str(error) for error in errors)
            separator = '' if error_message.startswith('\n') else '\n'
            raise InvalidMetadataError('The metadata is not valid' + separator + error_message)

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
        tables_with_mismatching_columns_order = []
        for table_name, table_data in data.items():
            table_sdtype_warnings = defaultdict(list)
            try:
                with warnings.catch_warnings(record=True):
                    self.tables[table_name].validate_data(table_data, table_sdtype_warnings)
                    if not self.tables[table_name]._check_data_columns_order(table_data.columns):
                        tables_with_mismatching_columns_order.append(table_name)

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

        if len(tables_with_mismatching_columns_order):
            affected_tables = ', '.join(map(repr, tables_with_mismatching_columns_order))
            warnings.warn(
                'The metadata lists columns in a different order than the data. '
                'This may result in the synthetic data having a different order.\n'
                f'Affected tables: {affected_tables}.'
            )

        return errors

    def _validate_foreign_keys(self, data):
        """Validate all foreign key relationships."""
        error_msg = None
        errors = []
        for relation in self.relationships:
            child_table = data.get(relation['child_table_name'])
            parent_table = data.get(relation['parent_table_name'])

            if isinstance(child_table, pd.DataFrame) and isinstance(parent_table, pd.DataFrame):
                child_columns = child_table[_cast_to_iterable(relation['child_foreign_key'])]
                parent_columns = parent_table[_cast_to_iterable(relation['parent_primary_key'])]
                missing_values = _get_unreferenced_keys(parent_columns, child_columns)
                missing_values = missing_values.drop_duplicates()
                if not missing_values.empty:
                    foreign_key = relation['child_foreign_key']
                    if not isinstance(foreign_key, list):
                        foreign_key = f"'{foreign_key}'"

                    message = f'\n{_format_invalid_values_string(missing_values, 5)}'
                    errors.append(
                        f'Error: foreign key column {foreign_key} contains '
                        f'unknown references:{message}\n'
                        "Please use the method 'drop_unknown_references' from sdv.utils "
                        'to clean the data.'
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

        for current_table_name, table_data in data.items():
            table_metadata = self.tables.get(current_table_name)
            if table_metadata is None:
                continue

            for column_name, column_metadata in table_metadata.columns.items():
                datetime_format = column_metadata.get('datetime_format')
                if not datetime_format:
                    continue

                column_data = table_data[column_name]
                has_datetime_objects = any(
                    not isinstance(value, str)
                    and not _is_numerical(value)
                    and _is_datetime_type(value)
                    for value in column_data.dropna().head(1000)
                )
                if has_datetime_objects:
                    warnings.warn(
                        f"The datetime format for column '{column_name}' "
                        f"(table '{current_table_name}') could not be verified because the data "
                        f"is represented as dtype '{column_data.dtype}'.\n"
                        'Please omit the datetime format string from the metadata or cast the '
                        'data to strings with the right format.'
                    )

    def validate_data(self, data):
        """Validate the data matches the metadata.

        Checks the following rules:
            * every table of the data satisfies its own metadata
            * all tables of the metadata are present in the data
            * all foreign keys belong to a primary key

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

        self.tables[table_name] = _SingleTableMetadata()
        self._multi_table_updated = True

    def get_column_names(self, table_name=None, **kwargs):
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
        table_name = self._handle_table_name(table_name)
        self._validate_table_exists(table_name)
        return self.tables[table_name].get_column_names(**kwargs)

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
        self._validate_table_exists(table_name)
        table_metadata = deepcopy(self.tables[table_name])
        return Metadata.load_from_dict(table_metadata.to_dict(), single_table_name=table_name)

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
            Metadata:
                An anonymized Metadata instance.
        """
        anonymized_metadata = self._get_anonymized_dict()

        return Metadata.load_from_dict(anonymized_metadata)

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
        if show_table_details not in (None, 'full', 'summarized'):
            raise ValueError(
                "'show_table_details' parameter should be 'full', 'summarized' or None."
            )

        nodes = {}
        edges = []

        for table_name in self.tables.keys():
            nodes[table_name] = self._get_table_info(table_name, show_table_details)

        for relationship in self.relationships:
            parent = relationship.get('parent_table_name')
            child = relationship.get('child_table_name')
            foreign_key = relationship.get('child_foreign_key')
            primary_key = relationship.get('parent_primary_key')
            edge_label = f'  {foreign_key} → {primary_key}' if show_relationship_labels else ''
            child_primary_key = self.tables.get(child).primary_key
            if foreign_key == child_primary_key:
                edges.append((parent, child, edge_label, 'one-to-one'))
            else:
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
        """Return a python ``dict`` representation of the ``Metadata``."""
        metadata = {'tables': {}, 'relationships': []}
        for table_name, single_table_metadata in self.tables.items():
            table_dict = single_table_metadata.to_dict()
            table_dict.pop('METADATA_SPEC_VERSION', None)
            metadata['tables'][table_name] = table_dict

        metadata['relationships'] = deepcopy(self.relationships)
        metadata['METADATA_SPEC_VERSION'] = self.METADATA_SPEC_VERSION
        return metadata

    def _validate_no_extra_keys_metadata_dict(self, metadata_dict):
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

    def _set_metadata_dict(self, metadata, single_table_name=None):
        """Set a ``metadata`` dictionary to the current instance.

        Checks to see if the metadata is in the ``_SingleTableMetadata`` or
        ``Metadata`` format and converts it to a standard
        ``Metadata`` format if necessary.

        Args:
            metadata (dict):
                Python dictionary representing a ``Metadata`` or
                ``_SingleTableMetadata`` object.
        """
        version = metadata.get('METADATA_SPEC_VERSION', 'V2')
        if 'tables' not in metadata:
            if single_table_name is None:
                single_table_name = self.DEFAULT_SINGLE_TABLE_NAME
                warnings.warn(
                    'No table name was provided to metadata containing only one table. '
                    f'Assigning name: {single_table_name}'
                )
            metadata = {'tables': {single_table_name: metadata}}

        self._validate_no_extra_keys_metadata_dict(metadata)
        for table_name, table_dict in metadata.get('tables', {}).items():
            try:
                self.tables[table_name] = _SingleTableMetadata._load_from_dict(
                    table_dict, version=version
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid metadata dict for table '{table_name}':\n {str(error)}"
                ) from error

        for relationship in metadata.get('relationships', []):
            parent_pk = relationship.get('parent_primary_key')
            child_fk = relationship.get('child_foreign_key')
            type_safe_pk = (
                [str(col) for col in parent_pk] if isinstance(parent_pk, list) else str(parent_pk)
            )
            type_safe_fk = (
                [str(col) for col in child_fk] if isinstance(parent_pk, list) else str(child_fk)
            )
            type_safe_relationships = {
                'parent_table_name': str(relationship.get('parent_table_name')),
                'child_table_name': str(relationship.get('child_table_name')),
                'parent_primary_key': type_safe_pk,
                'child_foreign_key': type_safe_fk,
            }
            self.relationships.append(type_safe_relationships)

    @classmethod
    def load_from_dict(cls, metadata_dict, single_table_name=None):
        """Create a ``Metadata`` instance from a python ``dict``.

        Args:
            metadata_dict (dict):
                Python dictionary representing a ``Metadata``
                or ``_SingleTableMetadata`` object.
            single_table_name (string):
                If the python dictionary represents a ``_SingleTableMetadata`` then
                this arg is used for the name of the table.

        Returns:
            Instance of ``Metadata``.
        """
        instance = cls()
        instance._set_metadata_dict(metadata_dict, single_table_name)
        return instance

    def save_to_json(self, filepath, mode='write'):
        """Save the current ``Metadata`` in to a ``json`` file.

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

        METADATA_LOGGER.info(
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
    def load_from_json(cls, filepath, single_table_name=None):
        """Create a ``Metadata`` instance from a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file.

        Returns:
            A ``Metadata`` instance.

        Raises:
            - An ``Error`` if the path does not exist.
            - An ``Error`` if the ``json`` file does not contain the ``METADATA_SPEC_VERSION``.
        """
        metadata = read_json(filepath)
        if metadata.get('METADATA_SPEC_VERSION', '').startswith('SINGLE_TABLE'):
            single_table_name = single_table_name or cls.DEFAULT_SINGLE_TABLE_NAME
            warnings.warn(
                'You are loading an older _SingleTableMetadata object. This will be converted into'
                f" the new Metadata object with a placeholder table name ('{single_table_name}')."
                ' Please save this new object for future usage.'
            )

        return cls.load_from_dict(metadata, single_table_name)

    def __repr__(self):
        """Pretty print the ``Metadata``."""
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

    def _get_single_table_name(self):
        """Get the table name if there is only one table.

        Checks to see if the metadata contains only a single table, if so
        return the name. Otherwise warn the user and return None.

        Args:
            metadata (dict):
                Python dictionary representing a ``Metadata`` or
                ``_SingleTableMetadata`` object.
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

        return next(iter(self.tables.values()), _SingleTableMetadata())

    @classmethod
    def upgrade_metadata(cls, filepath):
        """Upgrade an old metadata file to the ``V1`` schema.

        Args:
            filepath (str):
                String that represents the ``path`` to the old metadata ``json`` file.

        Returns:
            A ``Metadata`` instance.

        Raises:
            Raises a ``ValueError`` if the filepath does not exist.
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

    def copy(self):
        """Return a copy of the metadata.

        Returns:
            Metadata:
                Copy of current metadata.
        """
        return Metadata.load_from_dict(self.to_dict())
