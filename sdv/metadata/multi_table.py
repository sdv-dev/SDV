"""Multi Table Metadata."""

import json
import logging
import warnings
from collections import defaultdict
from copy import deepcopy

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata_upgrader import convert_metadata
from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.utils import read_json, validate_file_does_not_exist
from sdv.metadata.visualization import visualize_graph
from sdv.utils import cast_to_iterable

LOGGER = logging.getLogger(__name__)


class MultiTableMetadata:
    """Multi Table Metadata class."""

    METADATA_SPEC_VERSION = 'MULTI_TABLE_V1'

    def __init__(self):
        self.tables = {}
        self.relationships = []

    def _validate_missing_relationship_keys(self, parent_table_name, parent_primary_key,
                                            child_table_name, child_foreign_key):
        parent_table = self.tables.get(parent_table_name)
        child_table = self.tables.get(child_table_name)
        if parent_table.primary_key is None:
            raise InvalidMetadataError(
                f"The parent table '{parent_table_name}' does not have a primary key set. "
                "Please use 'set_primary_key' in order to set one."
            )

        missing_keys = set()
        parent_primary_key = cast_to_iterable(parent_primary_key)
        table_primary_keys = set(cast_to_iterable(parent_table.primary_key))
        for key in parent_primary_key:
            if key not in table_primary_keys:
                missing_keys.add(key)

        if missing_keys:
            raise InvalidMetadataError(
                f'Relationship between tables ({parent_table_name}, {child_table_name}) contains '
                f'an unknown primary key {missing_keys}.'
            )

        for key in set(cast_to_iterable(child_foreign_key)):
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
                    f'Relationship contains an unknown table {missing_table_names}.')
            else:
                raise InvalidMetadataError(
                    f'Relationship contains unknown tables {missing_table_names}.')

    @staticmethod
    def _validate_relationship_key_length(parent_table_name, parent_primary_key,
                                          child_table_name, child_foreign_key):
        pk_len = len(set(cast_to_iterable(parent_primary_key)))
        fk_len = len(set(cast_to_iterable(child_foreign_key)))
        if pk_len != fk_len:
            raise InvalidMetadataError(
                f"Relationship between tables ('{parent_table_name}', '{child_table_name}') is "
                f'invalid. Primary key has length {pk_len} but the foreign key has '
                f'length {fk_len}.'
            )

    def _validate_relationship_sdtypes(self, parent_table_name, parent_primary_key,
                                       child_table_name, child_foreign_key):
        parent_table_columns = self.tables.get(parent_table_name).columns
        child_table_columns = self.tables.get(child_table_name).columns
        parent_primary_key = cast_to_iterable(parent_primary_key)
        child_foreign_key = cast_to_iterable(child_foreign_key)
        for pk, fk in zip(parent_primary_key, child_foreign_key):
            if parent_table_columns[pk]['sdtype'] != child_table_columns[fk]['sdtype']:
                raise InvalidMetadataError(
                    f"Relationship between tables ('{parent_table_name}', '{child_table_name}') "
                    'is invalid. The primary and foreign key columns are not the same type.'
                )

    def _validate_circular_relationships(self, parent, children=None,
                                         parents=None, child_map=None, errors=None):
        """Validate that there is no circular relationship in the metadata."""
        parents = set() if parents is None else parents
        if children is None:
            children = child_map[parent]

        if parent in children:
            errors.append(parent)

        for child in children:
            if child in parents:
                break

            parents.add(child)
            self._validate_circular_relationships(
                parent,
                children=child_map.get(child, set()),
                child_map=child_map,
                parents=parents,
                errors=errors
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
        child_primary_key = cast_to_iterable(self.tables[child_table_name].primary_key)
        child_foreign_key = cast_to_iterable(child_foreign_key)
        if set(child_foreign_key).intersection(set(child_primary_key)):
            raise InvalidMetadataError(
                f"Invalid relationship between table '{parent_table_name}' and table "
                f"'{child_table_name}'. A relationship must connect a primary key "
                'with a non-primary key.'
            )

    def _validate_relationship_does_not_exist(self, parent_table_name, parent_primary_key,
                                              child_table_name, child_foreign_key):
        for relationship in self.relationships:
            already_exists = (
                relationship['parent_table_name'] == parent_table_name and
                relationship['parent_primary_key'] == parent_primary_key and
                relationship['child_table_name'] == child_table_name and
                relationship['child_foreign_key'] == child_foreign_key
            )
            if already_exists:
                raise InvalidMetadataError('This relationship has already been added.')

    def _validate_relationship(self, parent_table_name, child_table_name,
                               parent_primary_key, child_foreign_key):
        self._validate_no_missing_tables_in_relationship(
            parent_table_name, child_table_name, self.tables.keys())

        self._validate_missing_relationship_keys(
            parent_table_name,
            parent_primary_key,
            child_table_name,
            child_foreign_key
        )
        self._validate_relationship_key_length(
            parent_table_name,
            parent_primary_key,
            child_table_name,
            child_foreign_key
        )

        self._validate_foreign_child_key(child_table_name, parent_table_name, child_foreign_key)

        self._validate_relationship_sdtypes(
            parent_table_name,
            parent_primary_key,
            child_table_name,
            child_foreign_key
        )

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
            if parent_table_name == relation['parent_table_name'] and\
               child_table_name == relation['child_table_name']:
                foreign_keys.append(deepcopy(relation['child_foreign_key']))

        return foreign_keys

    def _get_all_foreign_keys(self, table_name):
        foreign_keys = []
        for relation in self.relationships:
            if table_name == relation['child_table_name']:
                foreign_keys.append(deepcopy(relation['child_foreign_key']))

        return foreign_keys

    def add_relationship(self, parent_table_name, child_table_name,
                         parent_primary_key, child_foreign_key):
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
            parent_table_name, child_table_name, parent_primary_key, child_foreign_key)

        child_map = self._get_child_map()
        child_map[parent_table_name].add(child_table_name)
        self._validate_relationship_does_not_exist(
            parent_table_name,
            parent_primary_key,
            child_table_name,
            child_foreign_key
        )
        self._validate_child_map_circular_relationship(child_map)

        self.relationships.append({
            'parent_table_name': parent_table_name,
            'child_table_name': child_table_name,
            'parent_primary_key': deepcopy(parent_primary_key),
            'child_foreign_key': deepcopy(child_foreign_key),
        })

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

    def add_constraint(self, table_name, constraint_name, **kwargs):
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
        table.add_constraint(constraint_name, **kwargs)

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

    def detect_table_from_dataframe(self, table_name, data):
        """Detect the metadata for a table from a dataframe.

        Args:
            table_name (str):
                Name of the table to detect.
            data (pandas.DataFrame):
                ``pandas.DataFrame`` to detect the metadata from.
        """
        self._validate_table_not_detected(table_name)
        table = SingleTableMetadata()
        table._detect_columns(data)
        self.tables[table_name] = table
        self._log_detected_table(table)

    def detect_table_from_csv(self, table_name, filepath):
        """Detect the metadata for a table from a csv file.

        Args:
            table_name (str):
                Name of the table to detect.
            filepath (str):
                String that represents the ``path`` to the ``csv`` file.
        """
        self._validate_table_not_detected(table_name)
        table = SingleTableMetadata()
        data = table._load_data_from_csv(filepath)
        table._detect_columns(data)
        self.tables[table_name] = table
        self._log_detected_table(table)

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

    def _validate_single_table(self, errors):
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
                    'The following errors were found in the metadata:\n', title)
                errors.append(error)

    def _validate_all_tables_connected(self, parent_map, child_map):
        nodes = list(self.tables.keys())
        queue = [nodes[0]]
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
                f'The relationships in the dataset are disjointed. {table_msg}')

    def _append_relationships_errors(self, errors, method, *args, **kwargs):
        try:
            method(*args, **kwargs)
        except Exception as error:
            if '\nRelationships:' not in errors:
                errors.append('\nRelationships:')

            errors.append(error)

    def _get_parent_map(self):
        parent_map = defaultdict(set)
        for relation in self.relationships:
            parent_name = relation['parent_table_name']
            child_name = relation['child_table_name']
            parent_map[child_name].add(parent_name)

        return parent_map

    def validate(self):
        """Validate the metadata.

        Raises:
            - ``InvalidMetadataError`` if the metadata is invalid.
        """
        errors = []
        self._validate_single_table(errors)
        for relation in self.relationships:
            self._append_relationships_errors(errors, self._validate_relationship, **relation)

        parent_map = self._get_parent_map()
        child_map = self._get_child_map()

        self._append_relationships_errors(
            errors, self._validate_child_map_circular_relationship, child_map)
        self._append_relationships_errors(
            errors, self._validate_all_tables_connected, parent_map, child_map)

        if errors:
            raise InvalidMetadataError(
                'The metadata is not valid' + '\n'.join(str(e) for e in errors)
            )

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

    def visualize(self, show_table_details=True, show_relationship_labels=True,
                  output_filepath=None):
        """Create a visualization of the multi-table dataset.

        Args:
            show_table_details (bool):
                If True, the column names, primary and foreign keys are all shown along with the
                table names. If False, only the table names are shown. Defaults to True.
            show_relationship_labels (bool):
                If True, every edge is labeled with the column names (eg. purchaser_id -> user_id).
                Defaults to True.
            output_filepath (str):
                Full path of where to savve the visualization. If None, the visualization is not
                saved. Defaults to None.

        Returns:
            ``graphviz.Digraph`` object.
        """
        nodes = {}
        edges = []
        if show_table_details:
            for table_name, table_meta in self.tables.items():
                column_dict = table_meta.columns.items()
                columns = [f"{name} : {meta.get('sdtype')}" for name, meta in column_dict]
                nodes[table_name] = {
                    'columns': r'\l'.join(columns),
                    'primary_key': f'Primary key: {table_meta.primary_key}'
                }

        else:
            nodes = {table_name: None for table_name in self.tables}

        for relationship in self.relationships:
            parent = relationship.get('parent_table_name')
            child = relationship.get('child_table_name')
            foreign_key = relationship.get('child_foreign_key')
            primary_key = self.tables.get(parent).primary_key
            edge_label = f'  {foreign_key} → {primary_key}' if show_relationship_labels else ''
            edges.append((parent, child, edge_label))

            if show_table_details:
                child_node = nodes.get(child)
                foreign_key_text = f'Foreign key ({parent}): {foreign_key}'
                if 'foreign_keys' in child_node:
                    child_node.get('foreign_keys').append(foreign_key_text)
                else:
                    child_node['foreign_keys'] = [foreign_key_text]

        for table, info in nodes.items():
            if show_table_details:
                foreign_keys = r'\l'.join(info.get('foreign_keys', []))
                keys = r'\l'.join([info['primary_key'], foreign_keys])
                label = fr"{{{table}|{info['columns']}\l|{keys}\l}}"

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
            self.tables[table_name] = SingleTableMetadata.load_from_dict(table_dict)

        for relationship in metadata.get('relationships', []):
            self.relationships.append(relationship)

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
        instance._set_metadata_dict(metadata_dict)
        return instance

    def save_to_json(self, filepath):
        """Save the current ``MultiTableMetadata`` in to a ``json`` file.

        Args:
            filepath (str):
                String that represent the ``path`` to the ``json`` file to be written.

        Raises:
            Raises a ``ValueError`` if the path already exists.
        """
        validate_file_does_not_exist(filepath)
        metadata = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

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
                'child_foreign_key': foreign_key
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
            'METADATA_SPEC_VERSION': cls.METADATA_SPEC_VERSION
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
