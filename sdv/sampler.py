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
            Table models.
        model (SDVModel):
            Model class to sample data.
        model_kwargs (dict):
            Additional arguments to create the ``SDVModel``.
    """
    metadata = None
    models = None
    primary_key = None
    remaining_primary_key = None

    def __init__(self, metadata, models, model, model_kwargs):
        self.metadata = metadata
        self.models = models
        self.primary_key = dict()
        self.remaining_primary_key = dict()
        self.model = model
        self.model_kwargs = model_kwargs

    def _reset_primary_keys_generators(self):
        """Reset the primary key generators."""
        self.primary_key = dict()
        self.remaining_primary_key = dict()

    def _transform_synthesized_rows(self, synthesized, table_name):
        """Reverse transform synthetized data.

        Args:
            synthesized (pandas.DataFrame):
                Generated data from model
            table_name (str):
                Name of the table.

        Return:
            pandas.DataFrame:
                Formatted synthesized data.
        """
        reversed_data = self.metadata.reverse_transform(table_name, synthesized)

        fields = self.metadata.get_fields(table_name)

        return reversed_data[list(fields.keys())]

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

    def _get_extension(self, parent_row, table_name):
        """Get the params from a generated parent row.

        Args:
            parent_row (pandas.Series):
                A generated parent row.
            table_name (str):
                Name of the table to make the model for.
        """

        prefix = '__{}__'.format(table_name)
        keys = [key for key in parent_row.keys() if key.startswith(prefix)]
        new_keys = {key: key.replace(prefix, '') for key in keys}
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

    def _sample_children(self, table_name, sampled):
        table_rows = sampled[table_name]
        for child_name in self.metadata.get_children(table_name):
            for _, row in table_rows.iterrows():
                self._sample_table(child_name, table_name, row, sampled)

    def _sample_table(self, table_name, parent_name, parent_row, sampled):
        extension = self._get_extension(parent_row, table_name)

        model = self.model(**self.model_kwargs)
        model.set_parameters(extension)
        num_rows = max(round(extension['child_rows']), 0)

        sampled_rows = self._sample_rows(model, num_rows, table_name)

        parent_key = self.metadata.get_primary_key(parent_name)
        foreign_key = self.metadata.get_foreign_key(parent_name, table_name)
        sampled_rows[foreign_key] = parent_row[parent_key]

        previous = sampled.get(table_name)
        if previous is None:
            sampled[table_name] = sampled_rows
        else:
            sampled[table_name] = pd.concat([previous, sampled_rows]).reset_index(drop=True)

        self._sample_children(table_name, sampled)

    def sample(self, table_name, num_rows, reset_primary_keys=False, sample_children=True):
        """Sample one table.

        Child tables will be sampled when ``sample_children`` is ``True``.
        If ``sampled_data`` is provided then append the sampled child tables there, if not
        create a new dict to fill.

        Args:
            table_name (str):
                Table name to sample.
            num_rows (int):
                Amount of rows to sample.
            reset_primary_keys (bool):
                Whether or not reset the primary keys generators. Defaults to ``False``.
            sample_children (bool):
                Whether or not sample child tables. Defaults to ``True``.

        Returns:
            dict or pandas.DataFrame:
                - Returns a ``dict`` when ``sample_children`` is ``True`` with the sampled table
                  and child tables.
                - Returns a ``pandas.DataFrame`` when ``sample_children`` is ``False``.
        """

        if reset_primary_keys:
            self._reset_primary_keys_generators()

        model = self.models[table_name]

        sampled_rows = self._sample_rows(model, num_rows, table_name)

        parents = self.metadata.get_parents(table_name)
        if parents:
            parent_name = list(parents)[0]
            foreign_key = self.metadata.get_foreign_key(parent_name, table_name)
            parent_id = self._get_primary_keys(parent_name, 1)[1][0]
            sampled_rows[foreign_key] = parent_id

        if sample_children:
            sampled_data = {
                table_name: sampled_rows
            }

            self._sample_children(table_name, sampled_data)

            for table, sampled_rows in sampled_data.items():
                sampled_data[table] = self._transform_synthesized_rows(sampled_rows, table)

            return sampled_data

        else:
            return self._transform_synthesized_rows(sampled_rows, table_name)

    def sample_all(self, num_rows=5, reset_primary_keys=False):
        """Samples the entire dataset.

        ``sample_all`` returns a dictionary with all the tables of the dataset sampled.
        The amount of rows sampled will depend from table to table, and is only guaranteed
        to match ``num_rows`` on tables without parents.

        This is because the children tables are created modelling the relation that they have
        with their parent tables, so its behavior may change from one table to another.

        Args:
            num_rows (int):
                Number of rows to be sampled on the parent tables.
            reset_primary_keys (bool):
                Wheter or not reset the primary key generators.

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
