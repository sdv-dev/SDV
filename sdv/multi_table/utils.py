"""Utility functions for the MultiTable models."""
import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from sdv._utils import _get_root_tables
from sdv.multi_table import HMASynthesizer
from sdv.multi_table.hma import MAX_NUMBER_OF_COLUMNS

MODELABLE_SDTYPE = ['categorical', 'numerical', 'datetime', 'boolean']


def _get_relationship_for_child(relationships, child_table):
    return [rel for rel in relationships if rel['child_table_name'] == child_table]


def _get_relationship_for_parent(relationships, parent_table):
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
    """
    descendants = {}
    order_1_descendants = _get_relationship_for_parent(relationships, parent_table)
    descendants['order_1'] = [rel['child_table_name'] for rel in order_1_descendants]
    for i in range(2, order+1):
        descendants[f'order_{i}'] = []
        prov_descendants = []
        for child_table in descendants[f'order_{i-1}']:
            order_i_descendants = _get_relationship_for_parent(relationships, child_table)
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
    """
    root_tables = _get_root_tables(relationships)
    all_descendants = {}
    for root in root_tables:
        order_descendants = _get_n_order_descendants(relationships, root, order)
        all_descendant_root = set()
        for order_key in order_descendants:
            all_descendant_root.update(order_descendants[order_key])

        all_descendants[root] = all_descendant_root

    return all_descendants


def _get_children_and_grandchildren(relationships, root_table, descendants_to_keep):
    """Get the children and grandchildren of the root table.

    Args:
        relationships (list[dict]):
            List of relationships between the tables.
        root_table (str):
            Name of the root table.
        descendants_to_keep (set):
            Set of the descendants of the root table that will be kept.
    """
    children = []
    grandchildren = []
    for relationship in relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        is_parent_root = parent_table == root_table
        is_parent_in_descendants = parent_table in descendants_to_keep
        is_valid_parent = is_parent_root or is_parent_in_descendants
        is_valid_child = child_table in descendants_to_keep
        if is_valid_parent and is_valid_child:
            if is_parent_root:
                children.append(child_table)
            else:
                grandchildren.append(child_table)

    return sorted(set(children)), sorted(set(grandchildren))


def _simplify_relationships(metadata, tables_to_drop):
    """Simplify the relationships of the metadata.

    Removes the relationships that are not direct child or grandchild of the root table.

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

    return metadata


def _remove_tables_metadata(metadata, tables_to_drop):
    """Simplify the tables that are not direct child or grandchild of the root table.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        tables_to_drop (set):
            Set of the tables to remove from the metadata.
    """
    tables = deepcopy(metadata.tables)
    for table in tables:
        if table in tables_to_drop:
            del metadata.tables[table]

    return metadata


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
        columns = metadata.tables[grandchild].columns
        columns_to_drop = [
            col_name for col_name in columns if columns[col_name]['sdtype'] in MODELABLE_SDTYPE
        ]
        for column in columns_to_drop:
            del columns[column]

    return metadata


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
        int:
            Number of columns to drop.
    """
    num_column_parameter = 4  # for beta distribution
    columns = metadata.tables[child_table].columns
    columns_to_sdtypes = {
        column: columns[column]['sdtype'] for column in columns
    }
    sdtypes_to_columns = defaultdict(list)
    for col, sdtype in columns_to_sdtypes.items():
        sdtypes_to_columns[sdtype].append(col)

    modelable_columns = {
        key: value for key, value in sdtypes_to_columns.items() if key in MODELABLE_SDTYPE
    }
    num_modelable_column = sum([len(value) for value in modelable_columns.values()])
    num_cols_to_drop = math.ceil(
        num_modelable_column + num_column_parameter - np.sqrt(
            num_column_parameter ** 2 + 1 + 2 * max_col_per_relationships
        )
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
            np.random.choice(
                modelable_columns[sdtype],
                num_col_to_drop_per_sdtype,
                replace=False
            )
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
    columns_to_drop = _get_columns_to_drop_child(
        metadata, child_table, max_col_per_relationships
    )
    columns = metadata.tables[child_table].columns
    for column in columns_to_drop:
        del columns[column]

    return metadata


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
        num_relationship = len(_get_relationship_for_child(relationships, child_table))
        if estimate_column > max_col_per_relationships * num_relationship:
            metadata = _simplify_child(
                metadata, child_table, max_col_per_relationships
            )

    return metadata


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
    all_descendants_per_root = _get_all_descendant_per_root_at_order_n(relationships, order=2)
    len_all_descendants_per_root = {
        root: len(all_descendants_per_root[root]) for root in all_descendants_per_root
    }
    root_to_keep = max(
        len_all_descendants_per_root, key=len_all_descendants_per_root.get
    )
    all_descendants_to_keep = all_descendants_per_root[root_to_keep]
    tables_to_keep = set(all_descendants_to_keep) | {root_to_keep}
    table_to_drop = set(simplified_metadata.tables.keys()) - tables_to_keep

    simplified_metadata = _simplify_relationships(simplified_metadata, table_to_drop)
    simplified_metadata = _remove_tables_metadata(simplified_metadata, table_to_drop)

    children, grandchildren = _get_children_and_grandchildren(
        relationships, root_to_keep, all_descendants_to_keep
    )
    if grandchildren:
        simplified_metadata = _simplify_grandchildren(simplified_metadata, grandchildren)

    estimate_columns_per_table = HMASynthesizer._estimate_num_columns(simplified_metadata)
    total_est_column = sum(estimate_columns_per_table.values())
    if total_est_column <= MAX_NUMBER_OF_COLUMNS:
        return simplified_metadata

    num_data_column = HMASynthesizer._get_num_data_columns(simplified_metadata)
    simplified_metadata = _simplify_children(
        simplified_metadata, children, root_to_keep, num_data_column
    )
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
        ]
    })
    message.append(summary.to_string(index=False))
    print('\n'.join(message))  # noqa: T001


def _get_rows_to_drop(metadata, data):
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
            relationships_parent = _get_relationship_for_parent(relationships, parent_table)
            parent_column = metadata.tables[parent_table].primary_key
            valid_parent_idx = [
                idx for idx in data[parent_table].index
                if idx not in table_to_idx_to_drop[parent_table]
            ]
            valid_parent_values = set(data[parent_table].loc[valid_parent_idx, parent_column])
            for relationship in relationships_parent:
                child_table = relationship['child_table_name']
                child_column = relationship['child_foreign_key']

                is_nan = data[child_table][child_column].isna()
                invalid_values = set(
                    data[child_table].loc[~is_nan, child_column]
                ) - valid_parent_values
                invalid_rows = data[child_table][
                    data[child_table][child_column].isin(invalid_values)
                ]
                idx_to_drop = set(invalid_rows.index)

                if idx_to_drop:
                    table_to_idx_to_drop[child_table] = table_to_idx_to_drop[
                        child_table
                    ].union(idx_to_drop)

            relationships = [rel for rel in relationships if rel not in relationships_parent]

    return table_to_idx_to_drop
