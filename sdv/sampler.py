import random

import pandas as pd

import exrex


class Sampler:
    """Class to sample data from a model."""

    def __init__(self, data_navigator, modeler):
        """Instantiate a new object."""
        self.dn = data_navigator
        self.modeler = modeler
        self.been_sampled = set()  # table_name -> if already sampled
        self.sampled = {}  # table_name -> [(primary_key, generated_row)]
        self.primary_key = {}

    @staticmethod
    def reset_indices_tables(sampled_tables):
        """Reset the indices of sampled tables.

        Args:
            sampled_tables (dict): All values are dataframes for sampled tables.

        Returns:
            dict: The same dict entered, just reindexed.
        """
        for name, table in sampled_tables.items():
            sampled_tables[name] = table.reset_index(drop=True)

        return sampled_tables

    def transform_synthesized_rows(self, synthesized_rows, table_name, num_rows):
        """Add primary key and reverse transform synthetized data.

        Args:
            synthesized_rows(pandas.DataFrame): Generated data from model
            table_name(str): Name of the table.
            num_rows(int): Number of rows sampled.

        Return:
            pandas.DataFrame: Formatted synthesized data.
        """
        # get primary key column name
        meta = self.dn.tables[table_name].meta
        orig_meta = self._get_table_meta(self.dn.meta, table_name)
        primary_key = meta.get('primary_key')

        if primary_key:
            node = meta['fields'][primary_key]
            regex = node['regex']

            generator = self.primary_key.get(table_name)

            if not generator:
                generator = exrex.generate(regex)

            values = [x for i, x in zip(range(num_rows), generator)]

            self.primary_key[table_name] = generator

            if len(values) != num_rows:
                raise ValueError(
                    'Not enough unique values for primary key of table {} with regex {}'
                    ' to generate {} samples.'.format(table_name, regex, num_rows)
                )

            synthesized_rows[primary_key] = pd.Series(values)

            if (node['type'] == 'number') and (node['subtype'] == 'integer'):
                synthesized_rows[primary_key] = pd.to_numeric(synthesized_rows[primary_key])

        sample_info = (primary_key, synthesized_rows)

        self.sampled = self.update_mapping_list(self.sampled, table_name, sample_info)

        # filter out parameters
        labels = list(self.dn.tables[table_name].data)
        synthesized_rows = self._fill_text_columns(synthesized_rows, labels, table_name)

        # reverse transform data
        reversed_data = self.dn.ht.reverse_transform_table(
            synthesized_rows, orig_meta, missing=False)

        synthesized_rows.update(reversed_data)
        return synthesized_rows[labels]

    def _get_parent_row(self, table_name):
        parents = self.dn.get_parents(table_name)
        if not parents:
            return None

        parent_rows = dict()
        for parent in parents:
            if parent not in self.sampled:
                raise Exception('Parents must be synthesized first')

            parent_rows[parent] = self.sampled[parent]

        random_parent, parent_rows = random.choice(list(parent_rows.items()))
        foreign_key, parent_row = random.choice(parent_rows)

        return random_parent, foreign_key, parent_row

    def sample_rows(self, table_name, num_rows):
        """Sample specified number of rows for specified table.

        Args:
            table_name (str): name of table to synthesize
            num_rows (int): number of rows to synthesize

        Returns:
            pd.DataFrame: synthesized rows.
        """

        parent_row = self._get_parent_row(table_name)

        if parent_row:
            random_parent, fk, parent_row = parent_row

            # Make sure only using one row
            parent_row = parent_row.loc[[0]]

            # get parameters from parent to make model
            model = self._make_model_from_params(
                parent_row, table_name, random_parent)

            # sample from that model
            if model is not None and len(model.distribs) > 0:
                synthesized_rows = model.sample(num_rows)
            else:
                raise ValueError(
                    'There was an error recreating models from parameters. '
                    'Sampling could not continue.'
                )

            # add foreign key value to row
            fk_val = parent_row.loc[0, fk]

            # get foreign key name from current table
            foreign_key = self.dn.foreign_keys[(table_name, random_parent)][1]
            synthesized_rows[foreign_key] = fk_val

            return self.transform_synthesized_rows(synthesized_rows, table_name, num_rows)

        else:    # there is no parent
            model = self.modeler.models[table_name]
            if len(model.distribs):
                synthesized_rows = model.sample(num_rows)
            else:
                raise ValueError(
                    'Modeler hasn\'t been fitted. '
                    'Please call Modeler.model_database() before sampling'
                )

            return self.transform_synthesized_rows(synthesized_rows, table_name, num_rows)

    def sample_table(self, table_name):
        """Sample a table equal to the size of the original.

        Args:
            table_name (str): name of table to synthesize

        Returns:
            pandas.DataFrame: Synthesized table.
        """
        orig_table = self.dn.tables[table_name].data
        num_rows = orig_table.shape[0]
        return self.sample_rows(table_name, num_rows)

    def sample_all(self, num_rows=5):
        """Samples the entire database.

        Args:
            num_rows (int): Number of rows to be sampled on the parent tables.

        Returns:
            dict: Tables sampled.

        `sample_all` returns a dictionary with all the tables of the dataset sampled.
        The amount of rows sampled will depend from table to table, and is only guaranteed
        to match `num_rows` on tables without parents.

        This is this way because the children tables are created modelling the relation
        thet have with their parent tables, so it's behavior may change from one table to another.
        """
        tables = self.dn.tables
        sampled_data = {}

        for table in tables:
            if not self.dn.get_parents(table):
                for _ in range(num_rows):
                    row = self.sample_rows(table, 1)

                    if table in sampled_data:
                        sampled_data[table] = pd.concat([sampled_data[table], row])
                    else:
                        sampled_data[table] = row

                    self._sample_child_rows(table, row, sampled_data)

        return self.reset_indices_tables(sampled_data)

    def _sample_child_rows(self, parent_name, parent_row, sampled_data, num_rows=5):
        """Uses parameters from parent row to synthesize child rows.

        Args:
            parent_name (str): name of parent table
            parent_row (dataframe): synthesized parent row
            sample_data (dict): maps table name to sampled data
            num_rows (int): number of rows to synthesize per parent row

        Returns:
            synthesized children rows
        """
        children = self.dn.get_children(parent_name)
        for child in children:
            rows = self.sample_rows(child, num_rows)

            if child in sampled_data:
                sampled_data[child] = pd.concat([sampled_data[child], rows])
            else:
                sampled_data[child] = rows

            self._sample_child_rows(child, rows.iloc[0:1, :], sampled_data)

    def _make_model_from_params(self, parent_row, table_name, parent_name):
        """ Takes the params from a generated parent row and creates a model from it.

        Args:
            parent_row (dataframe): a generated parent row
            table_name (string): name of table to make model for
            parent_name (string): name of parent table
        """
        # get parameters
        child_range = self.modeler.child_locs.get(parent_name, {}).get(table_name, {})

        if not child_range:
            return None

        param_indices = list(range(child_range[0], child_range[1]))
        params = parent_row.loc[:, param_indices]
        totalcols = params.shape[1]
        num_cols = self.modeler.tables[table_name].shape[1]

        # get labels for dataframe
        labels = list(self.modeler.tables[table_name].columns)

        # parent_meta = self.dn.tables[parent_name].meta
        # fk = parent_meta['primary_key']

        # if fk in labels:
        #     labels.remove(fk)
        #     num_cols -= 1

        cov_size = num_cols ** 2

        # Covariance matrix
        covariance = params.iloc[:, 0:cov_size]
        covariance = covariance.values.reshape((num_cols, num_cols))

        # Distributions
        distributions = {}
        for label_index, i in enumerate(range(cov_size, totalcols, 2)):
            distributions[labels[label_index]] = {
                'std': abs(params.iloc[:, i]),  # Pending for issue
                'mean': params.iloc[:, i + 1],  # https://github.com/HDI-Project/SDV/issues/58
            }

        model_params = {
            'covariance': covariance,
            'distribs': distributions
        }

        return self.modeler.model.from_dict(model_params)

    def _get_table_meta(self, meta, table_name):
        """Return metadata  get table meta for a given table name"""
        for table in meta['tables']:
            if table['name'] == table_name:
                return table

        return None

    def _fill_text_columns(self, row, labels, table_name):
        """Fill in the column values for every non numeric column that isn't the primary key.

        Args:
            row (pandas.Series): row to fill text columns.
            labels (list): Column names.
            table_name (str): Name of the table.

        Returns:
            pd.Series: Series with text values filled.
        """
        fields = self.dn.tables[table_name].meta['fields']
        for label in labels:
            field = fields[label]
            row_columns = list(row)
            if field['type'] == 'id' and field['name'] not in row_columns:
                # check foreign key
                ref = field.get('ref')
                if ref:
                    # generate parent row
                    parent_name = ref['table']
                    parent_row = self.sample_rows(parent_name, 1)
                    # grab value of foreign key
                    val = parent_row[ref['field']]
                    row.loc[:, field['name']] = val
                else:
                    # generate fake id
                    regex = field['regex']
                    row.loc[:, field['name']] = exrex.getone(regex)

            elif field['type'] == 'text':
                # generate fake text
                regex = field['regex']
                row.loc[:, field['name']] = exrex.getone(regex)

        return row

    def update_mapping_list(self, mapping, key, value):
        """Append value on mapping[key] if exists, create it otherwise."""
        item = mapping.get(key)

        if item:
            item.append(value)

        else:
            mapping[key] = [value]

        return mapping
