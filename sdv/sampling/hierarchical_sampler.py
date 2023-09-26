"""Hierarchical Samplers."""
import logging
import math

import pandas as pd

LOGGER = logging.getLogger(__name__)


class BaseHierarchicalSampler():
    """Hierarchical sampler mixin.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi-table metadata representing the data tables that this sampler will be used for.
        table_synthesizers (dict):
            Dictionary mapping each table to a synthesizer. Should be instantiated and passed to
            sampler by the multi-table synthesizer using this sampler.
        table_sizes (dict):
            Dictionary mapping each table to its size. Should be instantiated and passed to the
            sampler by the multi-table synthesizer using this sampler.
    """

    def __init__(self, metadata, table_synthesizers, table_sizes):
        self.metadata = metadata
        self._table_synthesizers = table_synthesizers
        self._table_sizes = table_sizes

    def _recreate_child_synthesizer(self, child_name, parent_name, parent_row):
        """Recreate a child table's synthesizer based on the parent's row.

        Args:
            child_name (str):
                The name of the child table.
            parent_name (str):
                The name of the parent table.
            parent_row (pd.Series):
                The row from the parent table to use for the child synthesizer.
        """
        raise NotImplementedError()

    def _add_foreign_key_columns(self, child_table, parent_table, child_name, parent_name):
        """Add all the foreign keys that connect the child table to the parent table.

        Args:
            child_table (pd.DataFrame):
                The table containing data sampled for the child.
            parent_table (pd.DataFrame):
                The table containing data sampled for the parent.
            parent_name (str):
                The name of the parent table.
            child_name (str):
                The name of the child table.
        """
        raise NotImplementedError()

    def _sample_rows(self, synthesizer, num_rows=None):
        """Sample ``num_rows`` from ``synthesizer``.

        Args:
            synthesizer (copula.multivariate.base):
                The fitted synthesizer for the table.
            num_rows (int or float):
                Number of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled rows, shape (, num_rows)
        """
        num_rows = num_rows or synthesizer._num_rows
        return synthesizer._sample_batch(int(num_rows), keep_extra_columns=True)

    def _get_num_rows_from_parent(self, parent_row, child_name, foreign_key):
        """Get the number of rows to sample for the child from the parent row."""
        num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
        num_rows = 0
        if num_rows_key in parent_row.keys():
            num_rows = parent_row[num_rows_key]
            num_rows = min(
                self._max_child_rows[num_rows_key],
                math.ceil(num_rows)
            )

        return num_rows

    def _add_child_rows(self, child_name, parent_name, parent_row, sampled_data):
        """Sample the child rows that reference the parent row.

        Args:
            child_name (str):
                The name of the child table.
            parent_name (str):
                The name of the parent table.
            parent_row (pd.Series):
                The row from the parent table to sample for from the child table.
            sampled_data (dict):
                A dictionary mapping table names to sampled data (pd.DataFrame).
        """
        # A child table is created based on only one foreign key.
        foreign_key = self.metadata._get_foreign_keys(parent_name, child_name)[0]
        num_rows = self._get_num_rows_from_parent(parent_row, child_name, foreign_key)
        child_synthesizer = self._recreate_child_synthesizer(child_name, parent_name, parent_row)

        sampled_rows = self._sample_rows(child_synthesizer, num_rows)

        if len(sampled_rows):
            parent_key = self.metadata.tables[parent_name].primary_key
            sampled_rows[foreign_key] = parent_row[parent_key]

            previous = sampled_data.get(child_name)
            if previous is None:
                sampled_data[child_name] = sampled_rows
            else:
                sampled_data[child_name] = pd.concat(
                    [previous, sampled_rows]).reset_index(drop=True)

    def _sample_children(self, table_name, sampled_data):
        """Recursively sample the children of a table.

        This method will loop through the children of a table and sample rows for that child for
        every primary key value in the parent. If the child has already been sampled by another
        parent, this method will skip it.

        Args:
            table_name (string):
                Name of the table to sample children for.
            sampled_data (dict):
                A dictionary mapping table names to sampled tables (pd.DataFrame).
        """
        for child_name in self.metadata._get_child_map()[table_name]:
            if child_name not in sampled_data:  # Sample based on only 1 parent
                for _, row in sampled_data[table_name].iterrows():
                    self._add_child_rows(
                        child_name=child_name,
                        parent_name=table_name,
                        parent_row=row,
                        sampled_data=sampled_data
                    )
                self._sample_children(table_name=child_name, sampled_data=sampled_data)

    def _finalize(self, sampled_data):
        """Remove extra columns from sampled tables and apply finishing touches.

        This method reverts the previous transformations to go back
        to values in the original space.

        Args:
            sampled_data (dict):
                Dictionary mapping table names to sampled tables (pd.DataFrame)

        Returns:
            Dictionary mapping table names to their formatted sampled tables.
        """
        final_data = {}
        for table_name, table_rows in sampled_data.items():
            synthesizer = self._table_synthesizers.get(table_name)
            dtypes = synthesizer._data_processor._dtypes
            for name, dtype in dtypes.items():
                table_rows[name] = table_rows[name].dropna().astype(dtype)

            final_data[table_name] = table_rows[list(dtypes.keys())]

        return final_data

    def _sample(self, scale=1.0):
        """Sample the entire dataset.

        Returns a dictionary with all the tables of the dataset. The amount of rows sampled will
        depend from table to table. This is because the children tables are created modelling the
        relation that they have with their parent tables, so its behavior may change from one
        table to another.

        Args:
            scale (float):
                A float representing how much to scale the data by. If scale is set to ``1.0``,
                this does not scale the sizes of the tables. If ``scale`` is greater than ``1.0``
                create more rows than the original data by a factor of ``scale``.
                If ``scale`` is lower than ``1.0`` create fewer rows by the factor of ``scale``
                than the original tables. Defaults to ``1.0``.

        Returns:
            dict:
                A dictionary containing as keys the names of the tables and as values the
                sampled data tables as ``pandas.DataFrame``.
        """
        sampled_data = {}

        # DFS to sample roots and then their children
        non_root_parents = set(self.metadata._get_parent_map().keys())
        root_parents = set(self.metadata.tables.keys()) - non_root_parents
        for table in root_parents:
            num_rows = int(self._table_sizes[table] * scale)
            synthesizer = self._table_synthesizers[table]
            LOGGER.info(f'Sampling {num_rows} rows from table {table}')
            sampled_data[table] = self._sample_rows(synthesizer, num_rows)
            self._sample_children(table_name=table, sampled_data=sampled_data)

        added_relationships = set()
        for relationship in self.metadata.relationships:
            parent_name = relationship['parent_table_name']
            child_name = relationship['child_table_name']
            # When more than one relationship exists between two tables, only the first one
            # is used to recreate the child tables, so the rest can be skipped.
            if (parent_name, child_name) not in added_relationships:
                self._add_foreign_key_columns(
                    sampled_data[child_name],
                    sampled_data[parent_name],
                    child_name,
                    parent_name
                )
                added_relationships.add((parent_name, child_name))

        return self._finalize(sampled_data)
