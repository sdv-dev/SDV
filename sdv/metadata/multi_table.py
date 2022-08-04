"""Multi Table Metadata."""

import json
import warnings
from collections import defaultdict
from copy import deepcopy

from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.utils import cast_to_iterable, read_json, validate_file_does_not_exist
from sdv.metadata.visualization import visualize_graph


class MultiTableMetadata:
    """Multi Table Metadata class."""

    def __init__(self):
        self._tables = {}
        self._relationships = []

    @staticmethod
    def _validate_missing_relationship_keys(parent_table, parent_table_name, parent_primary_key,
                                            child_table_name, child_foreign_key):
        missing_keys = set()
        parent_primary_key = cast_to_iterable(parent_primary_key)
        table_primary_keys = set(cast_to_iterable(parent_table._primary_key))
        for key in parent_primary_key:
            if key not in table_primary_keys:
                missing_keys.add(key)

        if missing_keys:
            raise ValueError(
                f'Relationship between tables ({parent_table_name}, {child_table_name}) contains '
                f'an unknown primary key {missing_keys}.'
            )

        for key in set(cast_to_iterable(child_foreign_key)):
            if key not in parent_table._columns:
                missing_keys.add(key)

        if missing_keys:
            raise ValueError(
                f'Relationship between tables ({parent_table_name}, {child_table_name}) '
                f'contains an unknown foreign key {missing_keys}.'
            )

    @staticmethod
    def _validate_missing_relationship_tables(parent_table_name, child_table_name, tables):
        missing_table_name = set([parent_table_name, child_table_name]) - set(tables)
        if missing_table_name:
            raise ValueError(f'Relationship contains an unknown table {missing_table_name}.')

    @staticmethod
    def _validate_relationship_key_length(parent_table_name, parent_primary_key,
                                          child_table_name, child_foreign_key):
        pk_len = len(set(cast_to_iterable(parent_primary_key)))
        fk_len = len(set(cast_to_iterable(child_foreign_key)))
        if pk_len != fk_len:
            raise ValueError(
                f"Relationship between tables ('{parent_table_name}', '{child_table_name}') is "
                f'invalid. Primary key has length {pk_len} but the foreign key has '
                f'length {fk_len}.'
            )

    @staticmethod
    def _validate_relationship_sdtypes(parent_table, parent_table_name, parent_primary_key,
                                       child_table_name, child_foreign_key):
        parent_columns = parent_table._columns
        error = False
        if isinstance(parent_primary_key, (list, tuple)):
            for pk, fk in zip(parent_primary_key, child_foreign_key):
                if parent_columns[pk]['sdtype'] != parent_columns[fk]['sdtype']:
                    error = True
                    break

        else:
            pk_sdtype = parent_columns[parent_primary_key]['sdtype']
            fk_sdtype = parent_columns[child_foreign_key]['sdtype']
            if pk_sdtype != fk_sdtype:
                error = True

        if error:
            raise ValueError(
                f"Relationship between tables ('{parent_table_name}', '{child_table_name}') is "
                'invalid. The primary and foreign key columns are not the same type.'
            )

    def _validate_circular_relationships(self, parent, children=None,
                                         parents=None, child_map=None, errors=None):
        """Validate that there is no circular relatioship in the metadata."""
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
            - ``ValueError`` if a table is missing.
            - ``ValueError`` if the ``parent_primary_key`` or ``child_foreign_key`` are missing.
            - ``ValueError`` if the ``parent_primary_key`` and ``child_foreign_key`` have different
              size.
            - ``ValueError`` if the ``parent_primary_key`` and ``child_foreign_key`` are different
              ``sdtype``.
            - ``ValueError`` if the relationship causes a circular dependency.
        """
        parent_table = self._tables.get(parent_table_name)
        self._validate_missing_relationship_tables(
            parent_table_name, child_table_name, self._tables.keys())

        self._validate_missing_relationship_keys(
            parent_table,
            parent_table_name,
            parent_primary_key,
            child_table_name,
            child_foreign_key
        )
        self._validate_relationship_key_length(
            parent_table_name, parent_primary_key, child_table_name, child_foreign_key)

        self._validate_relationship_sdtypes(
            parent_table,
            parent_table_name,
            parent_primary_key,
            child_table_name,
            child_foreign_key
        )

        child_map = defaultdict(set)
        for relation in self._relationships:
            parent_name = relation['parent_table_name']
            child_name = relation['child_table_name']
            child_map[parent_name].add(child_name)

        child_map[parent_table_name].add(child_table_name)
        errors = []
        for table_name in self._tables.keys():
            self._validate_circular_relationships(table_name, child_map=child_map, errors=errors)

        if errors:
            raise ValueError(
                'The relationships in the dataset describe a circular dependency between '
                f'tables {set(errors)}.'
            )

        self._relationships.append({
            'parent_table_name': parent_table_name,
            'child_table_name': child_table_name,
            'parent_primary_key': deepcopy(parent_primary_key),
            'child_foreign_key': deepcopy(child_foreign_key),
        })

    def to_dict(self):
        """Return a python ``dict`` representation of the ``MultiTableMetadata``."""
        metadata = {'tables': {}, 'relationships': []}
        for table_name, single_table_metadata in self._tables.items():
            metadata['tables'][table_name] = single_table_metadata.to_dict()

        metadata['relationships'] = deepcopy(self._relationships)
        return metadata

    def _set_metadata_dict(self, metadata):
        """Set a ``metadata`` dictionary to the current instance.

        Args:
            metadata (dict):
                Python dictionary representing a ``MultiTableMetadata`` object.
        """
        for table_name, table_dict in metadata.get('tables', {}).items():
            self._tables[table_name] = SingleTableMetadata._load_from_dict(table_dict)

        for relationship in metadata.get('relationships', []):
            self._relationships.append(relationship)

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
            for table_name, table_meta in self._tables.items():
                column_dict = table_meta._columns.items()
                columns = [f"{name} : {meta.get('sdtype')}" for name, meta in column_dict]
                nodes[table_name] = {
                    'columns': r'\l'.join(columns),
                    'primary_key': f'Primary key: {table_meta._primary_key}'
                }

        else:
            nodes = {table_name: None for table_name in self._tables}

        for relationship in self._relationships:
            parent = relationship.get('parent_table_name')
            child = relationship.get('child_table_name')
            foreign_key = relationship.get('child_foreign_key')
            primary_key = self._tables.get(parent)._primary_key
            edge_label = f'  {foreign_key} → {primary_key}' if show_relationship_labels else ''
            edges.append((parent, child, edge_label))

            if show_table_details:
                child_node = nodes.get(child)
                foreign_key_text = f"Foreign key ({parent}): {foreign_key}"
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

    @classmethod
    def _load_from_dict(cls, metadata):
        """Create a ``MultiTableMetadata`` instance from a python ``dict``.

        Args:
            metadata (dict):
                Python dictionary representing a ``MultiTableMetadata`` object.

        Returns:
            Instance of ``MultiTableMetadata``.
        """
        instance = cls()
        instance._set_metadata_dict(metadata)
        return instance

    def __repr__(self):
        """Pretty print the ``MultiTableMetadata``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed

    def _validate_table_exists(self, table_name):
        if table_name not in self._tables:
            raise ValueError(f"Unknown table name ('{table_name}').")

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
            - ``ValueError`` if the column doesn't already exist in the ``SingleTableMetadata``.
            - ``ValueError`` if the column has unexpected values or ``kwargs`` for the current
              ``sdtype``.
            - ``ValueError`` if the table doesn't exist in the ``MultiTableMetadata``.
        """
        self._validate_table_exists(table_name)
        table = self._tables.get(table_name)
        table.update_column(column_name, **kwargs)

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
            - ``ValueError`` if the column already exists.
            - ``ValueError`` if the ``kwargs`` do not contain ``sdtype``.
            - ``ValueError`` if the column has unexpected values or ``kwargs`` for the given
              ``sdtype``.
            - ``ValueError`` if the table doesn't exist in the ``MultiTableMetadata``.
        """
        self._validate_table_exists(table_name)
        table = self._tables.get(table_name)
        table.add_column(column_name, **kwargs)

    def _validate_table_not_detected(self, table_name):
        if table_name in self._tables:
            raise InvalidMetadataError(
                f"Metadata for table '{table_name}' already exists. Specify a new table name or "
                'create a new MultiTableMetadata object for other data sources.'
            )

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
        table.detect_from_csv(filepath)
        self._tables[table_name] = table

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
        table.detect_from_dataframe(data)
        self._tables[table_name] = table

    def set_primary_key(self, table_name, id):
        """Set the primary key of a table.

        Args:
            table_name (str):
                Name of the table to set the primary key.
            id (str, tulple[str]):
                Name (or tuple of names) of the primary key column(s).
        """
        self._validate_table_exists(table_name)
        self._tables[table_name].set_primary_key(id)

    def set_sequence_key(self, table_name, id):
        """Set the sequence key of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence key.
            id (str, tulple[str]):
                Name (or tuple of names) of the sequence key column(s).
        """
        self._validate_table_exists(table_name)
        warnings.warn('Sequential modeling is not yet supported on SDV Multi Table models.')
        self._tables[table_name].set_sequence_key(id)

    def set_alternate_keys(self, table_name, ids):
        """Set the alternate keys of a table.

        Args:
            table_name (str):
                Name of the table to set the sequence key.
            ids (list[str], list[tuple]):
                List of names (or tuple of names) of the alternate key columns.
        """
        self._validate_table_exists(table_name)
        self._tables[table_name].set_alternate_keys(ids)

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
        self._tables[table_name].set_sequence_index(column_name)

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
        table = self._tables.get(table_name)
        table.add_constraint(constraint_name, **kwargs)

    @classmethod
    def load_from_json(cls, filepath):
        """Create a ``MultiTableMetadata`` instance from a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file.

        Raises:
            - An ``Error`` if the path does not exist.
            - An ``Error`` if the ``json`` file does not contain the ``SCHEMA_VERSION``.

        Returns:
            A ``MultiTableMetadata`` instance.
        """
        metadata = read_json(filepath)
        return cls._load_from_dict(metadata)

    def save_to_json(self, filepath):
        """Save the current ``MultiTableMetadata`` in to a ``json`` file.

        Args:
            filepath (str):
                String that represent the ``path`` to the ``json`` file to be written.

        Raises:
            Raises an ``Error`` if the path already exists.
        """
        validate_file_does_not_exist(filepath)
        metadata = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
