"""Utility functions for the MultiTable models."""
from collections import defaultdict
from copy import deepcopy

import numpy as np

MODELABLE_SDTYPE = ['categorical', 'numerical', 'datetime', 'boolean']


def _get_root_tables(relationships):
    parent_tables = {rel['parent_table_name'] for rel in relationships}
    child_tables = {rel['child_table_name'] for rel in relationships}
    return parent_tables - child_tables


def _get_relationship_for_child(relationships, child_table):
    return [rel for rel in relationships if rel['child_table_name'] == child_table]


def _get_relationship_for_parent(relationships, parent_table):
    return [rel for rel in relationships if rel['parent_table_name'] == parent_table]


def _get_num_data_columns(metadata):
    """Get the number of data columns, ie colums that are not id, for each table.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
    """
    columns_per_table = {}
    for table_name, table in metadata.tables.items():
        columns_per_table[table_name] = \
            sum([1 for col in table.columns.values() if col['sdtype'] != 'id'])

    return columns_per_table


def _get_num_extended_columns(metadata, table_name, parent_table, columns_per_table):
    """Get the number of columns that will be generated for table_name.

    A table generates, for each foreign key:
        - 1 num_rows column
        - n*(n-1)/2 correlation columns for each data column
        - 4 parameter columns for each data column because we are using a beta distribution
    """
    num_rows_columns = len(metadata._get_foreign_keys(parent_table, table_name))

    # no parameter columns are generated if there are no data columns
    num_data_columns = columns_per_table[table_name]
    if num_data_columns == 0:
        return num_rows_columns

    num_parameters_columns = num_rows_columns * num_data_columns * 4

    num_correlation_columns = num_rows_columns * (num_data_columns - 1) * num_data_columns // 2

    return num_correlation_columns + num_rows_columns + num_parameters_columns


def _estimate_columns_traversal(metadata, table_name, columns_per_table, visited):
    """Given a table, estimate how many columns each parent will model.

    This method recursively models the children of a table all the way to the leaf nodes.

    Args:
        table_name (str):
            Name of the table to estimate the number of columns for.
        columns_per_table (dict):
            Dict that stores the number of data columns + extended columns for each table.
        visited (set):
            Set of table names that have already been visited.
    """
    for child_name in metadata._get_child_map()[table_name]:
        if child_name not in visited:
            _estimate_columns_traversal(metadata, child_name, columns_per_table, visited)

        columns_per_table[table_name] += \
            _get_num_extended_columns(metadata, child_name, table_name, columns_per_table)

    visited.add(table_name)


def _estimate_num_columns(metadata):
    """Estimate the number of columns that will be modeled for each table.

    This method estimates how many extended columns will be generated during the
    `_augment_tables` method, so it traverses the graph in the same way.
    If that method is ever changed, this should be updated to match.

    After running this method, `columns_per_table` will store an estimate of the
    total number of columns that each table has after running `_augment_tables`,
    that is, the number of extended columns generated by the child tables as well
    as the number of data columns in the table itself. `id` columns, like foreign
    and primary keys, are not counted since they are not modeled.

    Returns:
        dict:
            Dictionary of (table_name: int) mappings, indicating the estimated
            number of columns that will be modeled for each table.
    """
    # This dict will store the number of data columns + extended columns for each table
    # Initialize it with the number of data columns per table
    columns_per_table = _get_num_data_columns(metadata)

    # Starting at root tables, recursively estimate the number of columns
    # each table will model
    visited = set()
    for table_name in _get_root_tables(metadata.relationships):
        _estimate_columns_traversal(metadata, table_name, columns_per_table, visited)

    return columns_per_table


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
    descendances = {}
    order_1_descendants = _get_relationship_for_parent(relationships, parent_table)
    descendances['order_1'] = [rel['child_table_name'] for rel in order_1_descendants]
    for i in range(2, order+1):
        descendances[f'order_{i}'] = []
        prov_descendants = []
        for child_table in descendances[f'order_{i-1}']:
            order_i_descendants = _get_relationship_for_parent(relationships, child_table)
            prov_descendants.extend([rel['child_table_name'] for rel in order_i_descendants])

        descendances[f'order_{i}'] = prov_descendants

    return descendances


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
        order_descendances = _get_n_order_descendants(relationships, root, order)
        all_descendant_root = set()
        for order in order_descendances:
            all_descendant_root.update(order_descendances[order])

        all_descendants[root] = all_descendant_root

    return all_descendants


def _simplify_relationships(metadata, root_table, descendants_to_keep):
    """Simplify the relationships of the metadata.

    Removes the relationships that are not direct child or grandchild of the root table.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        root_table (str):
            Name of the root table.
        descendants_to_keep (set):
            Set of the descendants of the root table that will be kept.
    """
    relationships = deepcopy(metadata.relationships)
    childs = []
    grandchilds = []
    for relationship in relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        is_parent_root = parent_table == root_table
        is_parent_in_descendants = parent_table in descendants_to_keep
        is_valid_parent = is_parent_root or is_parent_in_descendants
        is_valid_child = child_table in descendants_to_keep
        if not is_valid_parent or not is_valid_child:
            metadata.remove_relationship(parent_table, child_table)
        else:
            if is_parent_root:
                childs.append(child_table)
            else:
                grandchilds.append(child_table)

    return metadata, set(childs), set(grandchilds)


def _simplify_non_descendants_tables(metadata, descendants_to_keep, root_table):
    """Simplify the tables that are not direct child or grandchild of the root table.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        descendants_to_keep (set):
            Set of the descendants of the root table that will be kept.
        root_table (str):
            Name of the root table.
    """
    tables = deepcopy(metadata.tables)
    for table in tables:
        if table not in descendants_to_keep and table != root_table:
            del metadata.tables[table]

    return metadata


def _simplify_grandchilds(metadata, grandchilds):
    """Simplify the grandchildren of the root table.

    The logic to simplify the grandchildren is to:
     - Drop all modelables columns.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        grandchilds (set):
            Set of the grandchildren of the root table.
    """
    for grandchild in grandchilds:
        columns = metadata.tables[grandchild].columns
        columns_to_drop = [
            col_name for col_name in columns if columns[col_name]['sdtype'] in MODELABLE_SDTYPE
        ]
        for column in columns_to_drop:
            del columns[column]

    return metadata


def _get_num_column_to_drop(
        metadata, child_table, max_col_per_relationships, num_relationship, estimate_column):
    """Get the number of columns to drop from the child table."""
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
    num_modelable_colum = sum([len(value) for value in modelable_columns.values()])
    numerator = estimate_column - max_col_per_relationships * num_relationship
    denomenator = (num_modelable_colum + 3) * num_relationship
    num_cols_to_drop = round(numerator / denomenator) + 1

    return num_cols_to_drop, modelable_columns


def _get_columns_to_drop_child(
        metadata, child_table, max_col_per_relationships, num_relationship, estimate_column):
    """Get the list of columns to drop from the child table."""
    num_col_to_drop, modelable_columns = _get_num_column_to_drop(
        metadata, child_table, max_col_per_relationships, num_relationship, estimate_column
    )
    num_modelable_colum = sum([len(value) for value in modelable_columns.values()])
    if num_col_to_drop >= num_modelable_colum:
        return modelable_columns

    columns_to_drop = []
    sdtypes_frequency = {
        sdtype: len(value) / num_modelable_colum for sdtype, value in modelable_columns.items()
    }
    for sdtype, frequency in sdtypes_frequency.items():
        num_col_to_drop_per_sdtype = round(num_col_to_drop * frequency)
        if num_col_to_drop_per_sdtype < 1:
            continue
        elif num_col_to_drop_per_sdtype >= len(modelable_columns[sdtype]):
            columns_to_drop.extend(modelable_columns[sdtype])
        else:
            columns_to_drop.extend(
                np.random.choice(
                    modelable_columns[sdtype],
                    num_col_to_drop_per_sdtype,
                    replace=False
                )
            )

    return columns_to_drop


def _simplify_child(
        metadata, child_table, max_col_per_relationships, num_relationship, estimate_column):
    """Simplify the child table.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        child_table (str):
            Name of the child table.
        max_col_per_relationships (int):
            Maximum number of columns to model per relationship.
        num_relationship (int):
            Number of relationships of the child table with the root table.
        estimate_column (int):
            Number of columns that will be added to the root table based on the child table.
    """
    columns_to_drop = _get_columns_to_drop_child(
        metadata, child_table, max_col_per_relationships, num_relationship, estimate_column
    )
    columns = metadata.tables[child_table].columns
    for column in columns_to_drop:
        del columns[column]

    return metadata


def _simplify_childs(metadata, childs, root_table, num_data_column):
    """Simplify the children of the root table.

    The logic to simplify the children is to:
     - Drop some modelable columns to have at most 1000 columns to model.

    Args:
        metadata (MultiTableMetadata):
            Metadata of the datasets.
        childs (set):
            Set of the children of the root table.
        root_table (str):
            Name of the root table.
        num_data_column (dict):
            Dictionary that maps each table name to the number of data columns.
    """
    relationships = metadata.relationships
    max_col_per_relationships = 1000 // len(relationships)
    for child_table in childs:
        estimate_column = _get_num_extended_columns(
            metadata, child_table, root_table, num_data_column
        )
        num_relationship = len(_get_relationship_for_child(relationships, child_table))
        if estimate_column > max_col_per_relationships * num_relationship:
            metadata = _simplify_child(
                metadata, child_table, max_col_per_relationships, num_relationship, estimate_column
            )

    return metadata


def _get_total_estimated_columns(metadata):
    """Get the total number of estimated columns from the metadata."""
    estimate_columns_per_table = _estimate_num_columns(metadata)
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
    simplify_metadata = deepcopy(metadata)
    relationships = simplify_metadata.relationships
    all_descendants_per_root = _get_all_descendant_per_root_at_order_n(relationships, 2)
    root_with_max_descendants = max(
        all_descendants_per_root, key=lambda x: len(all_descendants_per_root[x])
    )
    all_descendants_to_keep = all_descendants_per_root[root_with_max_descendants]

    simplify_metadata, childs, grandchilds = _simplify_relationships(
        simplify_metadata, root_with_max_descendants, all_descendants_to_keep
    )

    simplify_metadata = _simplify_non_descendants_tables(
        simplify_metadata, all_descendants_to_keep, root_with_max_descendants
    )

    if grandchilds:
        simplify_metadata = _simplify_grandchilds(simplify_metadata, grandchilds)

    estimate_columns_per_table = _estimate_num_columns(simplify_metadata)
    total_est_column = sum(estimate_columns_per_table.values())
    if total_est_column < 1000:
        return simplify_metadata

    num_data_column = _get_num_data_columns(simplify_metadata)
    simplify_metadata = _simplify_childs(
        simplify_metadata, childs, root_with_max_descendants, num_data_column
    )
    simplify_metadata.validate()

    return simplify_metadata


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
