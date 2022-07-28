"""Multi Table Metadata."""

import json
from copy import deepcopy

from sdv.metadata.single_table import SingleTableMetadata


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
        pass

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
