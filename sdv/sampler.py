"""SDV Sampler."""

import itertools

import exrex
import numpy as np
import pandas as pd


class Sampler:
    """Sampler class.

    Args:
        metadata (Metadata):
            Dataset Metadata.
        models (dict):
            Fitted table models.
        model (SDVModel):
            Model class to use to sample data.
        model_kwargs (dict):
            Additional arguments to create the ``SDVModel``.
        table_sizes (dict):
            Dict indicating the sizes of the tables in the orignal dataset.
    """

    metadata = None
    models = None
    primary_key = None
    remaining_primary_key = None

    def __init__(self, metadata, models, model, model_kwargs, table_sizes):
        self.metadata = metadata
        self.models = models
        self.primary_key = dict()
        self.remaining_primary_key = dict()
        self.model = model
        self.model_kwargs = model_kwargs
        self.table_sizes = table_sizes

    def _reset_primary_keys_generators(self):
        """Reset the primary key generators."""
        self.primary_key = dict()
        self.remaining_primary_key = dict()

    def _finalize(self, sampled_data):
        """Do the final touches to the generated data.

        This method reverts the previous transformations to go back
        to values in the original space and also adds the parent
        keys in case foreign key relationships exist between the tables.

        Args:
            sampled_data (dict):
                Generated data

        Return:
            pandas.DataFrame:
                Formatted synthesized data.
        """
        final_data = dict()
        for table_name, table_rows in sampled_data.items():
            parents = self.metadata.get_parents(table_name)
            if parents:
                for parent_name in parents:
                    foreign_key = self.metadata.get_foreign_key(parent_name, table_name)
                    if foreign_key not in table_rows:
                        parent_ids = self._find_parent_ids(table_name, parent_name, sampled_data)
                        table_rows[foreign_key] = parent_ids

            reversed_data = self.metadata.reverse_transform(table_name, table_rows)

            fields = self.metadata.get_fields(table_name)

            final_data[table_name] = reversed_data[list(fields.keys())]

        return final_data

    def _get_primary_keys(self, table_name, num_rows):
        """Return the primary key and amount of values for the requested table.

        Args:
            table_name (str):
                Name of the table to get the primary keys from.
            num_rows (str):
                Number of ``primary_keys`` to generate.

        Returns:
            tuple (str, pandas.Series):
                primary key name and primary key values. If the table has no primary
                key, ``(None, None)`` is returned.

        Raises:
            ValueError:
                If the ``metadata`` contains invalid types or subtypes, or if
                there are not enough primary keys left on any of the generators.
            NotImplementedError:
                If the primary key subtype is a ``datetime``.
        """
        primary_key = self.metadata.get_primary_key(table_name)
        primary_key_values = None

        if primary_key:
            field = self.metadata.get_fields(table_name)[primary_key]

            generator = self.primary_key.get(table_name)

            if generator is None:
                if field['type'] != 'id':
                    raise ValueError('Only columns with type `id` can be primary keys')

                subtype = field.get('subtype', 'integer')
                if subtype == 'integer':
                    generator = itertools.count()
                    remaining = np.inf
                elif subtype == 'string':
                    regex = field.get('regex', r'^[a-zA-Z]+$')
                    generator = exrex.generate(regex)
                    remaining = exrex.count(regex)
                elif subtype == 'datetime':
                    raise NotImplementedError('Datetime ids are not yet supported')
                else:
                    raise ValueError('Only `integer` or `string` id columns are supported.')

                self.primary_key[table_name] = generator
                self.remaining_primary_key[table_name] = remaining

            else:
                remaining = self.remaining_primary_key[table_name]

            if remaining < num_rows:
                raise ValueError(
                    'Not enough unique values for primary key of table {}'
                    ' to generate {} samples.'.format(table_name, num_rows)
                )

            self.remaining_primary_key[table_name] -= num_rows
            primary_key_values = pd.Series([x for i, x in zip(range(num_rows), generator)])

        return primary_key, primary_key_values

    def _extract_parameters(self, parent_row, table_name):
        """Get the params from a generated parent row.

        Args:
            parent_row (pandas.Series):
                A generated parent row.
            table_name (str):
                Name of the table to make the model for.
        """
        prefix = '__{}__'.format(table_name)
        keys = [key for key in parent_row.keys() if key.startswith(prefix)]
        new_keys = {key: key[len(prefix):] for key in keys}
        flat_parameters = parent_row[keys]
        return flat_parameters.rename(new_keys).to_dict()

    def _sample_rows(self, model, num_rows, table_name):
        """Sample ``num_rows`` from ``model``.

        Args:
            model (copula.multivariate.base):
                Fitted model.
            num_rows (int):
                Number of rows to sample.
            table_name (str):
                Name of the table to sample from.

        Returns:
            pandas.DataFrame:
                Sampled rows, shape (, num_rows)
        """
        primary_key_name, primary_key_values = self._get_primary_keys(table_name, num_rows)

        sampled = model.sample(num_rows)
        if primary_key_name:
            sampled[primary_key_name] = primary_key_values

        return sampled

    def _sample_children(self, table_name, sampled_data, table_rows=None):
        if table_rows is None:
            table_rows = sampled_data[table_name]

        for child_name in self.metadata.get_children(table_name):
            for _, row in table_rows.iterrows():
                self._sample_child_rows(child_name, table_name, row, sampled_data)

    def _sample_child_rows(self, table_name, parent_name, parent_row, sampled_data):
        parameters = self._extract_parameters(parent_row, table_name)

        model = self.model(**self.model_kwargs)
        model.set_parameters(parameters)
        num_rows = max(round(parameters['child_rows']), 0)

        table_rows = self._sample_rows(model, num_rows, table_name)

        parent_key = self.metadata.get_primary_key(parent_name)
        foreign_key = self.metadata.get_foreign_key(parent_name, table_name)
        table_rows[foreign_key] = parent_row[parent_key]

        previous = sampled_data.get(table_name)
        if previous is None:
            sampled_data[table_name] = table_rows
        else:
            sampled_data[table_name] = pd.concat([previous, table_rows]).reset_index(drop=True)

        self._sample_children(table_name, sampled_data, table_rows)

    @staticmethod
    def _find_parent_id(likelihoods, num_rows):
        mean = likelihoods.mean()
        if (likelihoods == 0).all():
            # All rows got 0 likelihood, fallback to num_rows
            likelihoods = num_rows
        elif pd.isnull(mean) or mean == 0:
            # Some rows got singlar matrix error and the rest were 0
            # Fallback to num_rows on the singular matrix rows and
            # keep 0s on the rest.
            likelihoods = likelihoods.fillna(num_rows)
        else:
            # at least one row got a valid likelihood, so fill the
            # rows that got a singular matrix error with the mean
            likelihoods = likelihoods.fillna(mean)

        weights = likelihoods.values / likelihoods.sum()

        return np.random.choice(likelihoods.index, p=weights)

    def _get_likelihoods(self, table_rows, parent_rows, table_name):
        likelihoods = dict()
        for parent_id, row in parent_rows.iterrows():
            parameters = self._extract_parameters(row, table_name)
            model = self.model(**self.model_kwargs)
            model.set_parameters(parameters)
            try:
                likelihoods[parent_id] = model.model.probability_density(table_rows)
            except np.linalg.LinAlgError:
                likelihoods[parent_id] = None

        return pd.DataFrame(likelihoods, index=table_rows.index)

    def _find_parent_ids(self, table_name, parent_name, sampled_data):
        table_rows = sampled_data[table_name]
        if parent_name in sampled_data:
            parent_rows = sampled_data[parent_name]
        else:
            ratio = self.table_sizes[parent_name] / self.table_sizes[table_name]
            num_parent_rows = max(int(round(len(table_rows) * ratio)), 1)
            parent_model = self.models[parent_name]
            parent_rows = self._sample_rows(parent_model, num_parent_rows, parent_name)

        primary_key = self.metadata.get_primary_key(parent_name)
        parent_rows = parent_rows.set_index(primary_key)
        num_rows = parent_rows['__' + table_name + '__child_rows'].clip(0)

        likelihoods = self._get_likelihoods(table_rows, parent_rows, table_name)
        return likelihoods.apply(self._find_parent_id, axis=1, num_rows=num_rows)

    def sample(self, table_name, num_rows=None, reset_primary_keys=False,
               sample_children=True, sample_parents=True):
        """Sample one table.

        Child tables will be sampled when ``sample_children`` is ``True``.
        If ``sampled_data`` is provided then append the sampled child tables there, if not
        create a new dict to fill.

        Args:
            table_name (str):
                Table name to sample.
            num_rows (int):
                Amount of rows to sample. If ``None``, sample the same number of rows
                as there were in the original table.
            reset_primary_keys (bool):
                Whether or not reset the primary keys generators. Defaults to ``False``.
            sample_children (bool):
                Whether or not sample child tables. Defaults to ``True``.
            sample_parents (bool):
                Whether or not sample parent tables. Defaults to ``True``.

        Returns:
            dict or pandas.DataFrame:
                - Returns a ``dict`` when ``sample_children`` is ``True`` with the sampled table
                  and child tables.
                - Returns a ``pandas.DataFrame`` when ``sample_children`` is ``False``.
        """
        if reset_primary_keys:
            self._reset_primary_keys_generators()

        if num_rows is None:
            num_rows = self.table_sizes[table_name]

        model = self.models[table_name]
        table_rows = self._sample_rows(model, num_rows, table_name)

        if sample_children:
            sampled_data = {
                table_name: table_rows
            }

            self._sample_children(table_name, sampled_data)
            return self._finalize(sampled_data)

        else:
            return self._finalize({table_name: table_rows})[table_name]

    def sample_all(self, num_rows=None, reset_primary_keys=False):
        """Sample the entire dataset.

        ``sample_all`` returns a dictionary with all the tables of the dataset sampled.
        The amount of rows sampled will depend from table to table, and is only guaranteed
        to match ``num_rows`` on tables without parents.

        This is because the children tables are created modelling the relation that they have
        with their parent tables, so its behavior may change from one table to another.

        Args:
            num_rows (int):
                Number of rows to be sampled on the first parent tables. If ``None``,
                sample the same number of rows as in the original tables.
            reset_primary_keys (bool):
                Whether or not reset the primary key generators.

        Returns:
            dict:
                A dictionary containing as keys the names of the tables and as values the
                sampled datatables as ``pandas.DataFrame``.
        """
        if reset_primary_keys:
            self._reset_primary_keys_generators()

        sampled_data = dict()
        for table in self.metadata.get_tables():
            if not self.metadata.get_parents(table):
                sampled_data.update(self.sample(table, num_rows))

        return sampled_data
