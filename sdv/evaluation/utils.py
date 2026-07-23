import sys
import warnings

from sdv._utils import _check_is_dict_of_dataframes
from sdv.metadata import Metadata


def print_referential_integrity(
    metadata, synthetic_data, table_name, foreign_key_column_name, num_rows=10
):
    """Check if referential integrity is meet by sampling a few rows.

    Args:
        metadata (sdv.metadata.Metadata): The metadata object
        synthetic_data (dict[str, pd.DataFrame]): A dictionary that maps each table name to a
            dataframe containing the synthetic data for it
        table_name (str): A string containing the table name that has the foreign key to check
        foreign_key_column_name (str): A string with the column of the foreign key to check
        num_rows (int, optional): An integer containing the number of columns to check.
            Defaults to 10 rows.

    Returns:
        None
    """
    if not isinstance(metadata, Metadata):
        raise TypeError('metadata must be of Metadata type')

    _check_is_dict_of_dataframes(synthetic_data, 'synthetic_data')

    if not isinstance(table_name, str):
        raise TypeError('table_name must be a string')

    if table_name not in metadata.tables:
        raise ValueError(f"table_name: '{table_name}' not found in metadata")

    if foreign_key_column_name not in metadata.tables[table_name].columns:
        raise ValueError(
            f"foreign_key_column_name: '{foreign_key_column_name}' not in Metadata for table_name:'{table_name}'"
        )

    if foreign_key_column_name not in synthetic_data[table_name].columns:
        raise ValueError(
            f"foreign_key_column_name: '{foreign_key_column_name}' not found in synthetic_data"
        )

    if not isinstance(num_rows, int) or num_rows < 0:
        raise TypeError("'num_rows' must be an integer greater than 0")

    nrows_child_table = len(synthetic_data[table_name])
    if nrows_child_table < num_rows:
        msg = f"The synthetic data contains '{nrows_child_table}' rows which is less "
        msg += f"than num_rows: '{num_rows}'. Changing num_rows to '{nrows_child_table}'."
        warnings.warn(msg)
        num_rows = nrows_child_table

    parent_table_name = None
    parent_primary_key = None
    for relation in metadata.relationships:
        if (
            table_name == relation['child_table_name']
            and relation['child_foreign_key'] == foreign_key_column_name
        ):
            parent_table_name = relation['parent_table_name']
            parent_primary_key = relation['parent_primary_key']
            # what about if multiple relations between two tables?
            # what about if composite keys
            # what if no relationships found?

    if parent_table_name is None:
        raise ValueError(
            f"Unable to find a relationship in metadata given table_name: '{table_name}' "
            f"and foreign_key_column_name: '{foreign_key_column_name}'"
        )

    child_data = synthetic_data[table_name]
    parent_data = synthetic_data[parent_table_name]

    child_primary_key = metadata.tables[table_name].primary_key

    random_rows = child_data.sample(n=num_rows, replace=False)
    parent_primary_keys = set(parent_data[parent_primary_key])
    for _, child_row in random_rows.iterrows():
        heading = f'Picking random {table_name} row'
        if child_primary_key:
            heading += f': {child_row[child_primary_key]}'
        sys.stdout.write(heading + '\n')

        foreign_key_value = child_row[foreign_key_column_name]
        if foreign_key_value in parent_primary_keys:
            result = f'✅ Found {parent_table_name} row! {parent_primary_key}: {foreign_key_value}'
        else:
            result = f'❌ Unable to find the linked {parent_table_name} row'
        result += '\n\n'
        sys.stdout.write(result)
