from sdv.cag._errors import PatternNotMetError


def _validate_table_and_column_names(table_name, columns, metadata):
    """Validate the table and column names for the pattern."""
    if table_name is None and len(metadata.tables) > 1:
        raise PatternNotMetError(
            'Metadata contains more than 1 table but no ``table_name`` provided.'
        )
    if table_name is None:
        table_name = metadata._get_single_table_name()
    elif table_name not in metadata.tables:
        raise PatternNotMetError(f"Table '{table_name}' missing from metadata.")

    if not set(columns).issubset(set(metadata.tables[table_name].columns)):
        missing_columns = columns - set(metadata.tables[table_name].columns)
        missing_columns = "', '".join(sorted(missing_columns))
        raise PatternNotMetError(f"Table '{table_name}' is missing columns '{missing_columns}'.")
