"""Utility functions."""
from sdv._utils import _get_relationship_idx_for_child, _get_rows_to_drop
from sdv.errors import InvalidDataError


def drop_unknown_references(metadata, data, drop_missing_values=True):
    """Drop rows with unknown foreign keys.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        drop_missing_values (bool):
            Boolean describing whether or not to also drop foreign keys with missing values
            If True, drop rows with missing values in the foreign keys.
            Defaults to True.

    Returns:
        dict:
            Dictionary with the dataframes ensuring referential integrity.
    """
    result = data.copy()
    table_to_idx_to_drop = _get_rows_to_drop(metadata, result)
    for table, idx_to_drop in table_to_idx_to_drop.items():
        result[table] = result[table].drop(idx_to_drop)
        if drop_missing_values:
            relationship_idx = _get_relationship_idx_for_child(metadata.relationships, table)
            for idx in relationship_idx:
                relationship = metadata.relationships[idx]
                child_column = relationship['child_foreign_key']
                result[table] = result[table].dropna(subset=[child_column])

        if result[table].empty:
            raise InvalidDataError([
                f"All references in table '{table}' are unknown and must be dropped."
                'Try providing different data for this table.'
            ])

    return result
