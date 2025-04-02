"""Utility functions for the MultiTable models."""

import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from sdv._utils import _get_root_tables
from sdv.errors import InvalidDataError, SamplingError
from sdv.multi_table import HMASynthesizer
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS

MODELABLE_SDTYPE = ['categorical', 'numerical', 'datetime', 'boolean']


def _get_child_tables(relationships):
    parent_tables = {rel['parent_table_name'] for rel in relationships}
    child_tables = {rel['child_table_name'] for rel in relationships}
    return child_tables - parent_tables


def _get_relationships_for_child(relationships, child_table):
    return [rel for rel in relationships if rel['child_table_name'] == child_table]


def _get_relationships_for_parent(relationships, parent_table):
    return [rel for rel in relationships if rel['parent_table_name'] == parent_table]


def _get_n_order_descendants(relationships, parent_table, order):
    """Get the descendants of the parent table until a given order.

    Args:
        relationships (list[dict]):
            List of relationships between the tables.
        parent_table (str):
            Name of the parent table.
        order (int):
            Order of the descendants.

    Returns:
        dict:
            Dictionary that maps the order to the descendants.
    """
    descendants = {}
    order_1_descendants = _get_relationships_for_parent(relationships, parent_table)
    descendants['order_1'] = [rel['child_table_name'] for rel in order_1_descendants]
    for i in range(2, order + 1):
        descendants[f'order_{i}'] = []
        prov_descendants = []
        for child_table in descendants[f'order_{i - 1}']:
            order_i_descendants = _get_relationships_for_parent(relationships, child_table)
            prov_descendants.extend([rel['child_table_name'] for rel in order_i_descendants])

        descendants[f'order_{i}'] = prov_descendants

    return descendants


def _get_all_descendant_per_root_at_order_n(relationships, order):
    """Get all the descendants of the root tables at a given order.

    Args:
        relationships (list[dict]):
            List of relationships between the tables.
        order (int):
            Order of the descendants.

    Returns:
        dict:
            Dictionary that maps the root table to the descendants at the given order.
    """
    root_tables = _get_root_tables(relationships)
    all_descendants = {}
    for root in root_tables:
        all_descendants[root] = {}
        order_descendants = _get_n_order_descendants(relationships, root, order)
        all_descendant_root = set()
        for order_key in order_descendants:
            all_descendant_root.update(order_descendants[order_key])

        all_descendants[root] = order_descendants
        all_descendants[root]['num_descendants'] = len(all_descendant_root)

    return all_descendants


def _get_ancestors(relationships, child_table):
    """Get the ancestors of the child table."""
    ancestors = set()
    parent_relationships = _get_relationships_for_child(relationships, child_table)
    for relationship in parent_relationships:
        parent_table = relationship['parent_table_name']
        ancestors.add(parent_table)
        ancestors.update(_get_ancestors(relationships, parent_table))

    return ancestors


def _get_disconnected_roots_from_table(relationships, table):
    """Get the disconnected roots table from the given table."""
    root_tables = _get_root_tables(relationships)
    child_tables = _get_child_tables(relationships)
    if table in child_tables:
        return root_tables - _get_ancestors(relationships, table)

    connected_roots = set()
    for child in child_tables:
        child_ancestor = _get_ancestors(relationships, child)
        if table in child_ancestor:
            connected_roots.update(root_tables.intersection(child_ancestor))

    return root_tables - connected_roots


def _simplify_relationships_and_tables(metadata, tables_to_drop):
    """Simplify the relationships and tables of the metadata.

    Removes the relationships that are not direct child or grandchild of the root table.
    Removes the tables that are not direct child or grandchild of the root table.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        tables_to_drop (set):
            Set of the tables that relationships will be removed.
    """
    relationships = deepcopy(metadata.relationships)
    for relationship in relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        is_parent_to_drop = parent_table in tables_to_drop
        is_child_to_drop = child_table in tables_to_drop
        if is_parent_to_drop or is_child_to_drop:
            metadata.relationships.remove(relationship)
            if is_parent_to_drop and parent_table in metadata.tables:
                del metadata.tables[parent_table]
            elif is_child_to_drop and child_table in metadata.tables:
                del metadata.tables[child_table]


def _simplify_grandchildren(metadata, grandchildren):
    """Simplify the grandchildren of the root table.

    The logic to simplify the grandchildren is to:
     - Drop all modelables columns.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        grandchildren (set):
            Set of the grandchildren of the root table.
    """
    for grandchild in grandchildren:
        key_columns = metadata._get_all_keys(grandchild)
        columns = metadata.tables[grandchild].columns
        columns_to_drop = [
            col_name
            for col_name in columns
            if columns[col_name]['sdtype'] in MODELABLE_SDTYPE
            or (columns[col_name]['sdtype'] == 'id' and col_name not in key_columns)
        ]
        for column in columns_to_drop:
            del columns[column]


def _get_num_column_to_drop(metadata, child_table, max_col_per_relationships):
    """Get the number of columns to drop from the child table.

    Formula to determine how many columns to drop for a child
        - n: number of columns of the table
        - k: num_column_parameter. For beta's distribution k=4
        - m: max_col_per_relationships

        - minimum number of column to drop = n + k - sqrt(k^2 + 1 + 2m)

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        child_table (str):
            Name of the child table.
        max_col_per_relationships (int):
            Maximum number of columns to model per relationship.

    Returns:
        tuple:
            - num_cols_to_drop (int): Number of columns to drop from the child table.
            - modelable_columns (dict): Dictionary that maps the modelable sdtype to their columns.
    """
    default_distribution = HMASynthesizer.DEFAULT_SYNTHESIZER_KWARGS['default_distribution']
    num_column_parameter = HMASynthesizer.DISTRIBUTIONS_TO_NUM_PARAMETER_COLUMNS[
        default_distribution
    ]
    columns = metadata.tables[child_table].columns
    modelable_columns = defaultdict(list)
    for column, column_metadata in columns.items():
        if column_metadata['sdtype'] in MODELABLE_SDTYPE:
            modelable_columns[column_metadata['sdtype']].append(column)

    num_modelable_column = sum([len(value) for value in modelable_columns.values()])
    num_cols_to_drop = math.ceil(
        num_modelable_column
        + num_column_parameter
        - np.sqrt(num_column_parameter**2 + 1 + 2 * max_col_per_relationships)
    )

    return num_cols_to_drop, modelable_columns


def _get_columns_to_drop_child(metadata, child_table, max_col_per_relationships):
    """Get the list of columns to drop from the child table."""
    num_col_to_drop, modelable_columns = _get_num_column_to_drop(
        metadata, child_table, max_col_per_relationships
    )
    num_modelable_column = sum([len(value) for value in modelable_columns.values()])
    if num_col_to_drop >= num_modelable_column:
        return [column for value in modelable_columns.values() for column in value]

    columns_to_drop = []
    sdtypes_frequency = {
        sdtype: len(value) / num_modelable_column for sdtype, value in modelable_columns.items()
    }
    for sdtype, frequency in sdtypes_frequency.items():
        num_col_to_drop_per_sdtype = round(num_col_to_drop * frequency)
        columns_to_drop.extend(
            np.random.choice(modelable_columns[sdtype], num_col_to_drop_per_sdtype, replace=False)
        )

    return columns_to_drop


def _simplify_child(metadata, child_table, max_col_per_relationships):
    """Simplify the child table.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        child_table (str):
            Name of the child table.
        max_col_per_relationships (int):
            Maximum number of columns to model per relationship.
    """
    columns_to_drop = _get_columns_to_drop_child(metadata, child_table, max_col_per_relationships)
    columns = metadata.tables[child_table].columns
    for column in columns_to_drop:
        del columns[column]


def _simplify_children(metadata, children, root_table, num_data_column):
    """Simplify the children of the root table.

    The logic to simplify the children is to:
     - Drop some modelable columns to have at most MAX_NUMBER_OF_COLUMNS columns to model.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        children (set):
            Set of the children of the root table.
        root_table (str):
            Name of the root table.
        num_data_column (dict):
            Dictionary that maps each table name to the number of data columns.
    """
    relationships = metadata.relationships
    max_col_per_relationships = MAX_NUMBER_OF_COLUMNS // len(relationships)
    for child_table in children:
        estimate_column = HMASynthesizer._get_num_extended_columns(
            metadata, child_table, root_table, num_data_column
        )
        num_relationship = len(_get_relationships_for_child(relationships, child_table))
        if estimate_column > max_col_per_relationships * num_relationship:
            _simplify_child(metadata, child_table, max_col_per_relationships)


def _get_total_estimated_columns(metadata):
    """Get the total number of estimated columns from the metadata."""
    estimate_columns_per_table = HMASynthesizer._estimate_num_columns(metadata)
    return sum(estimate_columns_per_table.values())


def _simplify_metadata(metadata):
    """Simplify the metadata of the datasets.

    The logic to simplify the metadata is to:
     - Keep the root table with the most descendants.
     - Drop tables and relationships that are not direct child or grandchild of the root table.
     - Drop all modelables columns of the grandchildren.
     - Drop some modelable columns in the children to have at most 1000 columns to model.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.

    Returns:
        MultiTableMetadata:
            Simplified metadata.
    """
    simplified_metadata = deepcopy(metadata)
    relationships = simplified_metadata.relationships
    descendants_per_root = _get_all_descendant_per_root_at_order_n(relationships, order=2)
    num_descendants_per_root = {
        root: descendants_per_root[root]['num_descendants'] for root in descendants_per_root
    }
    root_to_keep = max(num_descendants_per_root, key=num_descendants_per_root.get)
    children = descendants_per_root[root_to_keep]['order_1']
    grandchildren = descendants_per_root[root_to_keep]['order_2']
    tables_to_keep = set(children) | set(grandchildren) | {root_to_keep}
    table_to_drop = set(simplified_metadata.tables.keys()) - tables_to_keep

    _simplify_relationships_and_tables(simplified_metadata, table_to_drop)
    if grandchildren:
        _simplify_grandchildren(simplified_metadata, grandchildren)

    estimate_columns_per_table = HMASynthesizer._estimate_num_columns(simplified_metadata)
    total_est_column = sum(estimate_columns_per_table.values())
    if total_est_column <= MAX_NUMBER_OF_COLUMNS:
        return simplified_metadata

    num_data_column = HMASynthesizer._get_num_data_columns(simplified_metadata)
    _simplify_children(simplified_metadata, children, root_to_keep, num_data_column)
    simplified_metadata.validate()

    return simplified_metadata


def _simplify_data(data, metadata):
    """Simplify the data to match the simplified metadata.

    Args:
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        metadata (MultiTableMetadata):
            Metadata of the datasets.

    Returns:
        dict:
            Dictionary with the simplified dataframes.
    """
    simplify_data = deepcopy(data)
    tables_to_drop = set(simplify_data.keys()) - set(metadata.tables)
    for table in tables_to_drop:
        del simplify_data[table]

    for table in simplify_data:
        columns_to_drop = set(simplify_data[table].columns) - set(metadata.tables[table].columns)
        simplify_data[table] = simplify_data[table].drop(columns=columns_to_drop, axis=1)

    metadata.validate_data(simplify_data)

    return simplify_data


def _print_simplified_schema_summary(data_before, data_after):
    """Print the summary of the simplified schema."""
    message = ['Success! The schema has been simplified.\n']
    tables = sorted(data_before.keys())
    summary = pd.DataFrame({
        'Table Name': tables,
        '# Columns (Before)': [len(data_before[table].columns) for table in tables],
        '# Columns (After)': [
            len(data_after[table].columns) if table in data_after else 0 for table in tables
        ],
    })
    message.append(summary.to_string(index=False))
    print('\n'.join(message))  # noqa: T201


def _get_rows_to_drop(data, metadata):
    """Get the rows to drop to ensure referential integrity.

    The logic of this function is to start at the root tables, look at invalid references
    and then save the index of the rows to drop. Then, we looked at the relationships that
    we didn't check and repeat the process until there are no more relationships to check.
    This ensures that we preserve the referential integrity between all the relationships.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).

    Returns:
        dict:
            Dictionary with the table names as keys and the indexes of the rows to drop as values.
    """
    table_to_idx_to_drop = defaultdict(set)
    relationships = deepcopy(metadata.relationships)
    while relationships:
        current_roots = _get_root_tables(relationships)
        for root in current_roots:
            parent_table = root
            relationships_parent = _get_relationships_for_parent(relationships, parent_table)
            parent_column = metadata.tables[parent_table].primary_key
            valid_parent_idx = [
                idx
                for idx in data[parent_table].index
                if idx not in table_to_idx_to_drop[parent_table]
            ]
            valid_parent_values = set(data[parent_table].loc[valid_parent_idx, parent_column])
            for relationship in relationships_parent:
                child_table = relationship['child_table_name']
                child_column = relationship['child_foreign_key']

                is_nan = data[child_table][child_column].isna()
                invalid_values = (
                    set(data[child_table].loc[~is_nan, child_column]) - valid_parent_values
                )
                invalid_rows = data[child_table][
                    data[child_table][child_column].isin(invalid_values)
                ]
                idx_to_drop = set(invalid_rows.index)

                if idx_to_drop:
                    table_to_idx_to_drop[child_table] = table_to_idx_to_drop[child_table].union(
                        idx_to_drop
                    )

            relationships = [rel for rel in relationships if rel not in relationships_parent]

    return table_to_idx_to_drop


def _get_nan_fk_indices_table(data, relationships, table):
    """Get the indexes of the rows to drop that have NaN foreign keys."""
    idx_with_nan_foreign_key = set()
    relationships_for_table = _get_relationships_for_child(relationships, table)
    for relationship in relationships_for_table:
        child_column = relationship['child_foreign_key']
        idx_with_nan_foreign_key.update(data[table][data[table][child_column].isna()].index)

    return idx_with_nan_foreign_key


def _drop_rows(data, metadata, drop_missing_values):
    table_to_idx_to_drop = _get_rows_to_drop(data, metadata)
    for table in sorted(metadata.tables):
        idx_to_drop = table_to_idx_to_drop[table]
        data[table] = data[table].drop(idx_to_drop)
        if drop_missing_values:
            idx_with_nan_fk = _get_nan_fk_indices_table(data, metadata.relationships, table)
            data[table] = data[table].drop(idx_with_nan_fk)

        if data[table].empty:
            raise InvalidDataError([
                f"All references in table '{table}' are unknown and must be dropped. "
                'Try providing different data for this table.'
            ])


def _subsample_disconnected_roots(data, metadata, table, ratio_to_keep, drop_missing_values):
    """Subsample the disconnected roots tables and their descendants."""
    relationships = metadata.relationships
    roots = _get_disconnected_roots_from_table(relationships, table)
    for root in roots:
        data[root] = data[root].sample(frac=ratio_to_keep)

    _drop_rows(data, metadata, drop_missing_values)


def _subsample_table_and_descendants(data, metadata, table, num_rows, drop_missing_values):
    """Subsample the table and its descendants.

    The logic is to first subsample all the NaN foreign keys of the table when
    ``drop_missing_values`` is True. We raise an error if we cannot reach referential integrity
    while keeping the number of rows. Then, we drop rows of the descendants to ensure referential
    integrity.

    Args:
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        table (str):
            Name of the table.
        num_rows (int):
            Number of rows to keep in the table.
        drop_missing_values (bool):
            Boolean describing whether or not to also drop foreign keys with missing values
            If True, drop rows with missing values in the foreign keys.
            Defaults to False.
    """
    if drop_missing_values:
        idx_nan_fk = _get_nan_fk_indices_table(data, metadata.relationships, table)
        num_rows_to_drop = len(data[table]) - num_rows
        if len(idx_nan_fk) > num_rows_to_drop:
            raise SamplingError(
                f"Referential integrity cannot be reached for table '{table}' while keeping "
                f'{num_rows} rows. Please try again with a bigger number of rows.'
            )
        else:
            data[table] = data[table].drop(idx_nan_fk)

    data[table] = data[table].sample(num_rows)
    _drop_rows(data, metadata, drop_missing_values)


def _get_primary_keys_referenced(data, metadata):
    """Get the primary keys referenced by the relationships.

    Args:
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        metadata (MultiTableMetadata):
            Metadata of the datasets.

    Returns:
        dict:
            Dictionary that maps the table name to a set of their primary keys referenced.
    """
    relationships = metadata.relationships
    primary_keys_referenced = defaultdict(set)
    for relationship in relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        foreign_key = relationship['child_foreign_key']
        primary_keys_referenced[parent_table].update(set(data[child_table][foreign_key].unique()))

    return primary_keys_referenced


def _subsample_parent(
    parent_table, parent_primary_key, parent_pk_referenced_before, dereferenced_pk_parent
):
    """Subsample the parent table.

    The strategy here is to:
    - Drop the rows that are no longer referenced by the descendants.
    - Drop a proportional amount of never-referenced rows.

    Args:
        parent_table (pandas.DataFrame):
            Parent table to subsample.
        parent_primary_key (str):
            Name of the primary key of the parent table.
        parent_pk_referenced_before (set):
            Set of the primary keys referenced before any subsampling.
        dereferenced_pk_parent (set):
            Set of the primary keys that are no longer referenced by the descendants.

    Returns:
        pandas.DataFrame:
            Subsampled parent table.
    """
    total_referenced = len(parent_pk_referenced_before)
    total_dropped = len(dereferenced_pk_parent)
    drop_proportion = total_dropped / total_referenced

    parent_table = parent_table[~parent_table[parent_primary_key].isin(dereferenced_pk_parent)]
    unreferenced_data = parent_table[
        ~parent_table[parent_primary_key].isin(parent_pk_referenced_before)
    ]

    # Randomly drop a proportional amount of never-referenced rows
    unreferenced_data_to_drop = unreferenced_data.sample(frac=drop_proportion)
    parent_table = parent_table.drop(unreferenced_data_to_drop.index)
    if parent_table.empty:
        raise InvalidDataError([
            f"All references in table '{parent_primary_key}' are unknown and must be dropped. "
            'Try providing different data for this table.'
        ])

    return parent_table


def _subsample_ancestors(data, metadata, table, primary_keys_referenced):
    """Subsample the ancestors of the table.

    The strategy here is to recursively subsample the direct parents of the table until the
    root tables are reached.

    Args:
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        table (str):
            Name of the table.
        primary_keys_referenced (dict):
            Dictionary that maps the table name to a set of their primary keys referenced
            before any subsampling.
    """
    relationships = metadata.relationships
    pk_referenced = _get_primary_keys_referenced(data, metadata)
    direct_relationships = _get_relationships_for_child(relationships, table)
    direct_parents = {rel['parent_table_name'] for rel in direct_relationships}
    for parent in sorted(direct_parents):
        parent_primary_key = metadata.tables[parent].primary_key
        pk_referenced_before = primary_keys_referenced[parent]
        dereferenced_primary_keys = pk_referenced_before - pk_referenced[parent]
        data[parent] = _subsample_parent(
            data[parent], parent_primary_key, pk_referenced_before, dereferenced_primary_keys
        )
        if dereferenced_primary_keys:
            primary_keys_referenced[parent] = pk_referenced[parent]

        _subsample_ancestors(data, metadata, parent, primary_keys_referenced)


def _subsample_data(data, metadata, main_table_name, num_rows, drop_missing_values=False):
    """Subsample multi-table table based on a table and a number of rows.

    The strategy is to:
    - Subsample the disconnected roots tables by keeping a similar proportion of data
      than the main table. Ensure referential integrity.
    - Subsample the main table and its descendants to ensure referential integrity.
    - Subsample the ancestors of the main table by removing primary key rows that are no longer
      referenced by the descendants and some unreferenced rows.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        data (dict):
            Dictionary that maps each table name (string) to the data for that
            table (pandas.DataFrame).
        main_table_name (str):
            Name of the main table.
        num_rows (int):
            Number of rows to keep in the main table.
        drop_missing_values (bool):
            Boolean describing whether or not to also drop foreign keys with missing values
            If True, drop rows with missing values in the foreign keys.
            Defaults to False.

    Returns:
        dict:
            Dictionary with the subsampled dataframes.
    """
    result = deepcopy(data)
    primary_keys_referenced = _get_primary_keys_referenced(result, metadata)
    ratio_to_keep = num_rows / len(result[main_table_name])

    try:
        _subsample_disconnected_roots(
            result, metadata, main_table_name, ratio_to_keep, drop_missing_values
        )
        _subsample_table_and_descendants(
            result, metadata, main_table_name, num_rows, drop_missing_values
        )
        _subsample_ancestors(result, metadata, main_table_name, primary_keys_referenced)
        _drop_rows(result, metadata, drop_missing_values)

    except InvalidDataError as error:
        if 'All references in table' not in str(error.args[0]):
            raise error
        else:
            raise SamplingError(
                f'Subsampling {main_table_name} with {num_rows} rows leads to some empty tables. '
                'Please try again with a bigger number of rows.'
            )

    return result


def _print_subsample_summary(data_before, data_after):
    """Print the summary of the subsampled data."""
    tables = sorted(data_before.keys())
    summary = pd.DataFrame({
        'Table Name': tables,
        '# Rows (Before)': [len(data_before[table]) for table in tables],
        '# Rows (After)': [
            len(data_after[table]) if table in data_after else 0 for table in tables
        ],
    })
    subsample_rows = 100 * (1 - summary['# Rows (After)'].sum() / summary['# Rows (Before)'].sum())
    message = [f'Success! Your subset has {round(subsample_rows)}% less rows than the original.\n']
    message.append(summary.to_string(index=False))
    print('\n'.join(message))  # noqa: T201
