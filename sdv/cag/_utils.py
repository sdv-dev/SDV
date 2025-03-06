from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata


def _validate_table_name(table_name, metadata):
    """Validates that if the table_name is None, the metadata only contains a single table."""
    if table_name is None and len(metadata.tables) > 1:
        raise PatternNotMetError(
            'Metadata contains more than 1 table but no ``table_name`` provided.'
        )


def _validate_columns_in_metadata(table_name, columns, metadata):
    if not set(columns).issubset(set(metadata.tables[table_name].columns)):
        missing_columns = columns - set(metadata.tables[table_name].columns)
        missing_columns = "', '".join(sorted(missing_columns))
        raise PatternNotMetError(f"Table '{table_name}' is missing columns '{missing_columns}'.")


def _validate_table_and_column_names(table_name, columns, metadata):
    """Validate the table and column names for the pattern."""
    _validate_table_name(table_name, metadata)

    if table_name is None:
        table_name = metadata._get_single_table_name()
    elif table_name not in metadata.tables:
        raise PatternNotMetError(f"Table '{table_name}' missing from metadata.")

    _validate_columns_in_metadata(table_name, columns, metadata)


def _validate_table_name_if_defined(table_name):
    if table_name and not isinstance(table_name, str):
        raise ValueError('`table_name` must be a string or None.')


def _remove_columns_from_metadata(metadata, table_name, columns_to_drop):
    """Remove columns from metadata, including column relationships."""
    metadata = metadata.to_dict()
    for column in columns_to_drop:
        primary_key = metadata['tables'][table_name].get('primary_key')
        if primary_key and primary_key == column:
            raise ValueError('Cannot remove primary key from Metadata')
        del metadata['tables'][table_name]['columns'][column]

    metadata['tables'][table_name]['column_relationships'] = [
        rel
        for rel in metadata['tables'][table_name].get('column_relationships', [])
        if set(rel['column_names']).isdisjoint(columns_to_drop)
    ]

    return Metadata.load_from_dict(metadata)


def _is_list_of_strings(values):
    return isinstance(values, list) and all(isinstance(value, str) for value in values)
