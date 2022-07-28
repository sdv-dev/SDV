"""Multi Table Metadata."""

import json
import warnings
from copy import deepcopy

import graphviz

from sdv.metadata.single_table import SingleTableMetadata
from sdv.metadata.visualization import _get_graphviz_extension


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

    def visualize(self, show_table_details, show_relationship_labels, output_filepath=None):
        """Create a visualization of the multi-table dataset.
        
        Args:
            show_table_details (bool):
                If True, the column names, primary and foreign keys are all shown along with the
                table names. If False, only the table names are shown.
            show_relationship_labels (bool):
                If True, every edge is labeled with the column names (eg. purchaser_id -> user_id).
            output_filepath (str):
                Full path of where to savve the visualization. If None, the visualization is not
                saved. Defaults to None.
        
        Returns:
            ``graphviz.Digraph`` object.
        """
        filename, graphviz_extension = _get_graphviz_extension(output_filepath)
        digraph = graphviz.Digraph(
            'Metadata',
            format=graphviz_extension,
            node_attr={
                "shape": "Mrecord",
                "fillcolor": "lightgoldenrod1",
                "style": "filled"
            },
        )

        nodes = {}
        edges = {}
        for table_name, table_meta in self._tables.items():
            columns = []
            for column_name, column_meta in table_meta._columns.items():
                columns.append(f"{column_name} : {column_meta.get('sdtype')}")
            nodes[table_name] = {
                'columns': r'\l'.join(columns),
                'primary_key': f"Primary key: {table_meta._metadata['primary_key']}"
            }

        for relationship in self._relationships:
            parent = relationship.get('parent_table_name')
            child_node = nodes.get(relationship.get('child_table_name'))
            foreign_key = f"Foreign key: ({parent}): {relationship.get('child_foreign_key')}"
            if 'foreign_key' in child_node:
                child_node.get('foreign_keys').append(foreign_key)
            else:
                child_node['foreign_keys'] = [foreign_key]
        
        for table, info in nodes.items():
            foreign_keys = r'\l'.join(info.get('foreign_keys', []))
            keys = r'\l'.join([info['primary_key'], foreign_keys])
            label = f"{table}|{info['columns']}|{keys}"
            digraph.node(table, label=label)
        
        if filename:
            digraph.render(filename=filename, cleanup=True, format=graphviz_extension)
        else:
            try:
                graphviz.version()
            except graphviz.ExecutableNotFound:
                warning_message = (
                    'Graphviz does not seem to be installed on this system. For full '
                    'metadata visualization capabilities, please make sure to have its '
                    'binaries propertly installed: https://graphviz.gitlab.io/download/'
                )
                warnings.warn(warning_message, RuntimeWarning)

        return digraph

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
