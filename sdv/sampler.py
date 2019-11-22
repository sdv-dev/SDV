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
    """
    metadata = None
    models = None
    primary_key = None
    remaining_primary_key = None

    def __init__(self, metadata, models):
        self.metadata = metadata
        self.models = models
        self.primary_key = dict()
        self.remaining_primary_key = dict()

    @staticmethod
    def _square_matrix(triangular_matrix):
        """Fill with zeros a triangular matrix to reshape it to a square one.

        Args:
            triangular_matrix (list [list [float]]):
                Array of arrays of

        Returns:
            list:
                Square matrix.
        """
        length = len(triangular_matrix)
        zero = [0.0]

        for item in triangular_matrix:
            item.extend(zero * (length - len(item)))

        return triangular_matrix

    def _check_matrix_symmetric_positive_definite(self, matrix):
        """Check if a matrix is symmetric positive-definite.

        Args:
            matrix (list or numpy.ndarray):
                Matrix to evaluate.

        Returns:
            bool
        """
        try:
            if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
                # Not 2-dimensional or square, so not simmetric.
                return False

            np.linalg.cholesky(matrix)
            return True

        except np.linalg.LinAlgError:
            return False

    def _make_positive_definite(self, matrix):
        """Find the nearest positive-definite matrix to input.

        Args:
            matrix (numpy.ndarray):
                Matrix to transform

        Returns:
            numpy.ndarray:
                Closest symetric positive-definite matrix.
        """
        symetric_matrix = (matrix + matrix.T) / 2
        _, s, V = np.linalg.svd(symetric_matrix)
        symmetric_polar = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (symetric_matrix + symmetric_polar) / 2
        A3 = (A2 + A2.T) / 2

        if self._check_matrix_symmetric_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(matrix))
        identity = np.eye(matrix.shape[0])
        iterations = 1
        while not self._check_matrix_symmetric_positive_definite(A3):
            min_eigenvals = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += identity * (-min_eigenvals * iterations**2 + spacing)
            iterations += 1

        return A3

    def _prepare_sampled_covariance(self, covariance):
        """Prepare a covariance matrix.

        Args:
            covariance (list):
                covariance after unflattening model parameters.

        Result:
            list[list]:
                symmetric Positive semi-definite matrix.
        """
        covariance = np.array(self._square_matrix(covariance))
        covariance = (covariance + covariance.T - (np.identity(covariance.shape[0]) * covariance))

        if not self._check_matrix_symmetric_positive_definite(covariance):
            covariance = self._make_positive_definite(covariance)

        return covariance.tolist()

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

    @staticmethod
    def _key_order(key_value):
        parts = list()
        for part in key_value[0].split('__'):
            if part.isdigit():
                part = int(part)

            parts.append(part)

        return parts

    def _unflatten_dict(self, flat):
        """Transform a flattened dict into its original form.

        Args:
            flat (dict):
                Flattened dict.

        Returns:
            dict:
                Nested dict (if corresponds)
        """
        unflattened = dict()

        for key, value in sorted(flat.items(), key=self._key_order):
            if '__' in key:
                key, subkey = key.split('__', 1)
                subkey, name = subkey.rsplit('__', 1)

                if name.isdigit():
                    column_index = int(name)
                    row_index = int(subkey)

                    array = unflattened.setdefault(key, list())

                    if len(array) == row_index:
                        row = list()
                        array.append(row)
                    elif len(array) == row_index + 1:
                        row = array[row_index]
                    else:
                        # This should never happen
                        raise ValueError('There was an error unflattening the extension.')

                    if len(row) == column_index:
                        row.append(value)
                    else:
                        # This should never happen
                        raise ValueError('There was an error unflattening the extension.')

                else:
                    subdict = unflattened.setdefault(key, dict())
                    if subkey.isdigit():
                        subkey = int(subkey)

                    inner = subdict.setdefault(subkey, dict())
                    inner[name] = value

            else:
                unflattened[key] = value

        return unflattened

    def _unflatten_gaussian_copula(self, model_parameters):
        """Prepare unflattened model params to recreate Gaussian Multivariate instance.

        The preparations consist basically in:

            - Transform sampled negative standard deviations from distributions into positive
              numbers

            - Ensure the covariance matrix is a valid symmetric positive-semidefinite matrix.

            - Add string parameters kept inside the class (as they can't be modelled),
              like ``distribution_type``.

        Args:
            model_parameters (dict):
                Sampled and reestructured model parameters.

        Returns:
            dict:
                Model parameters ready to recreate the model.
        """

        distribution_kwargs = {
            'fitted': True,
            'type': model_parameters['distribution']
        }

        distribs = model_parameters['distribs']
        for distribution in distribs.values():
            distribution.update(distribution_kwargs)
            distribution['std'] = np.exp(distribution['std'])

        covariance = model_parameters['covariance']
        model_parameters['covariance'] = self._prepare_sampled_covariance(covariance)

        return model_parameters

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

    def _get_model(self, extension, table_model):
        """Build a model using the extension parameters."""
        table_model_parameters = table_model.to_dict()

        model_parameters = self._unflatten_dict(extension)
        model_parameters['fitted'] = True
        model_parameters['distribution'] = table_model_parameters['distribution']

        model_parameters = self._unflatten_gaussian_copula(model_parameters)

        return table_model.from_dict(model_parameters)

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

        table_model = self.models[table_name]
        model = self._get_model(extension, table_model)
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
            foreign_key = self.metadata.foreign_keys[(table_name, parent_name)][1]
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
