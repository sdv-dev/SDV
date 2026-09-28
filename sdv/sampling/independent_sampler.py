"""Independent Samplers."""

import logging
import warnings

LOGGER = logging.getLogger(__name__)


class BaseIndependentSampler:
    """Independent sampler mixin.

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

    def _sample_table(self, synthesizer, table_name, num_rows, sampled_data):
        """Sample a single table and all its children.

        Args:
            synthesizer (SingleTableSynthesizer):
                Synthesizer to sample from for the table.
            table_name (string):
                Name of the table to sample.
            num_rows (int):
                Number of rows to sample for the table.
            sampled_data (dict):
                A dictionary mapping table names to sampled tables (pd.DataFrame).
        """
        LOGGER.info(f'Sampling {num_rows} rows from table {table_name}')

        sampled_rows = synthesizer._sample_batch(num_rows, keep_extra_columns=True)
        sampled_data[table_name] = sampled_rows

    def _connect_tables(self, sampled_data):
        """Connect all related tables.

        This method iterates over all tables and adds all foreign keys to connect
        related tables.

        Args:
            sampled_data (dict):
                A dictionary mapping table names to the sampled tables (pd.DataFrame).
        """
        queue = [
            table for table in self.metadata.tables if not self.metadata._get_parent_map()[table]
        ]
        while queue:
            parent = queue.pop(0)
            for child in self.metadata._get_child_map()[parent]:
                self._add_foreign_key_columns(
                    sampled_data[child], sampled_data[parent], child, parent
                )
                if set(self.metadata._get_all_foreign_keys(child)).issubset(
                    set(sampled_data[child].columns)
                ):
                    queue.append(child)

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
            column_names = self.get_metadata().get_column_names(table_name)
            if not synthesizer._fitted:
                final_data[table_name] = table_rows[column_names]
                continue

            dtypes = synthesizer._data_processor._dtypes
            dtypes_to_sdtype = synthesizer._data_processor._DTYPE_TO_SDTYPE
            for name in column_names:
                dtype = dtypes.get(name)
                if dtype is None:
                    continue

                try:
                    table_rows[name] = table_rows[name].dropna().astype(dtype)
                except ValueError as e:
                    sdtype = self.metadata.tables[table_name].columns.get(name).get('sdtype')
                    if sdtype not in dtypes_to_sdtype.values():
                        LOGGER.info(
                            f"The real data in '{table_name}' and column '{name}' was stored as "
                            f"'{dtype}' but the synthetic data could not be cast back to "
                            'this type. If this is a problem, please check your input data '
                            'and metadata settings.'
                        )

                    else:
                        raise ValueError(e)
                except OverflowError:
                    LOGGER.debug(
                        f"The real data in '{table_name}' and column '{name}' was stored as "
                        f"'{dtype}' but the synthetic data overflowed when casting back to "
                        'this type. If this is a problem, please check your input data '
                        'and metadata settings.'
                    )

            final_data[table_name] = table_rows[column_names]

        return final_data

    def _sample(self, scale=1.0):
        """Sample the entire dataset.

        Returns a dictionary with all the tables of the dataset. The amount of rows sampled will
        depend from table to table. This is because the children tables are created modeling the
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
        send_min_sample_warning = False
        for table in self.metadata.tables:
            num_rows = int(self._table_sizes[table] * scale)
            if num_rows <= 0:
                send_min_sample_warning = True
                num_rows = 1
            synthesizer = self._table_synthesizers[table]
            self._sample_table(
                synthesizer=synthesizer,
                table_name=table,
                num_rows=num_rows,
                sampled_data=sampled_data,
            )

        if send_min_sample_warning:
            warn_msg = (
                "The 'scale' parameter is too small. Some tables may have 1 row."
                ' For better quality data, please choose a larger scale.'
            )
            warnings.warn(warn_msg)

        self._connect_tables(sampled_data)
        sampled_data = self._reverse_transform_constraints(sampled_data)
        return self._finalize(sampled_data)
