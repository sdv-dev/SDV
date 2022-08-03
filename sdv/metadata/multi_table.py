"""Multi Table Metadata."""

import json
from copy import deepcopy

from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.visualization import visualize_graph


class MultiTableMetadata:
    """Multi Table Metadata class."""

    def __init__(self):
        self._tables = {}
        self._relationships = []

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
                    'primary_key': f"Primary key: {table_meta._metadata['primary_key']}"
                }

        else:
            nodes = {table_name: None for table_name in self._tables}

        for relationship in self._relationships:
            parent = relationship.get('parent_table_name')
            child = relationship.get('child_table_name')
            foreign_key = relationship.get('child_foreign_key')
            primary_key = self._tables.get(parent)._metadata.get('primary_key')
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

    def _validate_table_name(self, table_name):
        if table_name not in self._tables:
            raise ValueError(f"Unknown table name ('{table_name}')")

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
        self._validate_table_name(table_name)
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
        self._validate_table_name(table_name)
        table = self._tables.get(table_name)
        table.add_column(column_name, **kwargs)
