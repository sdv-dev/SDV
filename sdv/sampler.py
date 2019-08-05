import exrex
import numpy as np
import pandas as pd
from copulas import get_qualified_name
from rdt.transformers.positive_number import PositiveNumberTransformer

GAUSSIAN_COPULA = 'copulas.multivariate.gaussian.GaussianMultivariate'


MODEL_ERROR_MESSAGES = {
    True: (
        'There was an error recreating models from parameters. '
        'Sampling could not continue.'
    ),
    False: (
        'Modeler hasn\'t been fitted. '
        'Please call Modeler.model_database() before sampling'
    )
}


class Sampler:
    """Class to sample data from a model."""

    def __init__(self, data_navigator, modeler):
        """Instantiate a new object."""
        self.dn = data_navigator
        self.modeler = modeler
        self.primary_key = dict()
        self.remaining_primary_key = dict()

    @staticmethod
    def _square_matrix(triangular_matrix):
        """Fill with zeros a triangular matrix to reshape it to a square one.

        Args:
            triangular_matrix (list[list[float]]): Array of arrays of

        Returns:
            list: Square matrix.
        """
        length = len(triangular_matrix)
        zero = [0.0]

        for item in triangular_matrix:
            item.extend(zero * (length - len(item)))

        return triangular_matrix

    def _prepare_sampled_covariance(self, covariance):
        """

        Args:
            covariance (list): covariance after unflattening model parameters.

        Result:
            list[list]: symmetric Positive semi-definite matrix.
        """
        covariance = np.array(self._square_matrix(covariance))
        covariance = (covariance + covariance.T - (np.identity(covariance.shape[0]) * covariance))
        return covariance

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

    def _reset_primary_keys_generators(self):
        """Reset the primary key generators."""
        self.primary_key = dict()
        self.remaining_primary_key = dict()

    def _transform_synthesized_rows(self, synthesized, table_name):
        """Reverse transform synthetized data.

        Args:
            synthesized(pandas.DataFrame): Generated data from model
            table_name(str): Name of the table.

        Return:
            pandas.DataFrame: Formatted synthesized data.
        """
        orig_meta = self.dn.get_meta_data(table_name)

        labels = list(orig_meta['fields'].keys())

        reverse_columns = [
            transformer[1] for transformer in self.dn.ht.transformers
            if table_name in transformer
        ]

        text_filled = self._fill_text_columns(synthesized, labels, table_name)

        # This is here only because RDT expects a different meta format
        # RDT should be fixed to work with the new format.
        fields = list()
        for name, values in orig_meta['fields'].items():
            field = values.copy()
            field['name'] = name
            fields.append(field)

        meta = {
            'fields': fields,
            'name': table_name
        }

        # reverse transform data
        reversed_data = self.dn.ht.reverse_transform_table(text_filled[reverse_columns], meta)

        synthesized.update(reversed_data)

        for name, field in orig_meta['fields'].items():
            subtype = field.get('subtype')
            if subtype == 'integer':
                synthesized[name] = synthesized[name].astype(int)

        return synthesized[labels]

    def _get_primary_keys(self, table_name, num_rows):
        """Return the primary key and amount of values for the requested table.

        Args:
            table_name(str): Name of the table to get the primary keys.
            num_rowd(str): Number of primary_keys to generate.

        Returns:
            tuple(str,pandas.Series): If the table has a primary key.
            tuple(None, None): If the table doesn't have a primary key.

        Raises:
            ValueError: If there aren't enough remaining values to generate.

        """
        meta = self.dn.get_meta_data(table_name)
        primary_key = meta.get('primary_key')
        primary_key_values = None

        if primary_key:
            node = meta['fields'][primary_key]
            regex = node['regex']

            generator = self.primary_key.get(table_name)

            if generator is None:
                generator = exrex.generate(regex)
                self.primary_key[table_name] = generator

                remaining = exrex.count(regex)
                self.remaining_primary_key[table_name] = remaining

            else:
                remaining = self.remaining_primary_key[table_name]

            if remaining < num_rows:
                raise ValueError(
                    'Not enough unique values for primary key of table {} with regex {}'
                    ' to generate {} samples.'.format(table_name, regex, num_rows)
                )

            self.remaining_primary_key[table_name] -= num_rows
            primary_key_values = pd.Series([x for i, x in zip(range(num_rows), generator)])

            if (node['type'] == 'number') and (node['subtype'] == 'integer'):
                primary_key_values = primary_key_values.astype(int)

        return primary_key, primary_key_values

    @staticmethod
    def _setdefault(a_dict, key, a_type):
        if key not in a_dict:
            value = a_type()
            a_dict[key] = value

        else:
            value = a_dict[key]

        return value

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

        Works in exact opposite way that `sdv.Modeler._flatten_dict`.

        Args:
            flat (dict): Flattened dict.

        Returns:
            dict: Nested dict (if corresponds)

        """
        unflattened = dict()

        for key, value in sorted(flat.items(), key=self._key_order):
            if '__' in key:
                key, subkey = key.split('__', 1)
                subkey, name = subkey.rsplit('__', 1)

                if name.isdigit():
                    column_index = int(name)
                    row_index = int(subkey)

                    array = self._setdefault(unflattened, key, list)

                    if len(array) == row_index:
                        row = list()
                        array.append(row)
                    elif len(array) == row_index + 1:
                        row = array[row_index]
                    else:
                        raise ValueError('There was an error unflattening the extension.')

                    if len(row) == column_index:
                        row.append(value)
                    else:
                        raise ValueError('There was an error unflattening the extension.')

                else:
                    subdict = self._setdefault(unflattened, key, dict)
                    if subkey.isdigit():
                        subkey = int(subkey)

                    inner = self._setdefault(subdict, subkey, dict)
                    inner[name] = value

            else:
                unflattened[key] = value

        return unflattened

    def _make_positive_definite(self, matrix):
        """Find the nearest positive-definite matrix to input

        Args:
            matrix (numpy.ndarray): Matrix to transform

        Returns:
            numpy.ndarray: Closest symetric positive-definite matrix.
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

    def _check_matrix_symmetric_positive_definite(self, matrix):
        """Checks if a matrix is symmetric positive-definite.

        Args:
            matrix (list or np.ndarray): Matrix to evaluate.

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

    def _unflatten_gaussian_copula(self, model_parameters):
        """Prepare unflattened model params to recreate Gaussian Multivariate instance.

        The preparations consist basically in:
        - Transform sampled negative standard deviations from distributions into positive numbers
        - Ensure the covariance matrix is a valid symmetric positive-semidefinite matrix.
        - Add string parameters kept inside the class (as they can't be modelled),
          like `distribution_type`.

        Args:
            model_parameters (dict): Sampled and reestructured model parameters.

        Returns:
            dict: Model parameters ready to recreate the model.
        """

        distribution_name = self.modeler.model_kwargs['distribution']
        distribution_kwargs = {
            'fitted': True,
            'type': distribution_name
        }
        model_parameters['distribution'] = distribution_name

        distribs = model_parameters['distribs']
        metadata = {
            'name': 'std',
            'type': 'number'
        }
        transformer = PositiveNumberTransformer(metadata)

        for distribution in distribs.values():
            distribution.update(distribution_kwargs)
            df = pd.DataFrame({'std': [distribution['std']]})
            distribution['std'] = transformer.transform(df).loc[0, 'std']

        covariance = model_parameters['covariance']
        covariance = self._prepare_sampled_covariance(covariance)
        if not self._check_matrix_symmetric_positive_definite(covariance):
            covariance = self._make_positive_definite(covariance)

        model_parameters['covariance'] = covariance.tolist()

        return model_parameters

    def _get_extension(self, parent_row, table_name, parent_name):
        """ Takes the params from a generated parent row.

        Args:
            parent_row (dataframe): a generated parent row
            table_name (string): name of table to make model for
            parent_name (string): name of parent table
        """

        prefix = '__{}__'.format(table_name)
        keys = [key for key in parent_row.keys() if key.startswith(prefix)]
        new_keys = {key: key.replace(prefix, '') for key in keys}
        flat_parameters = parent_row[keys]
        return flat_parameters.rename(new_keys).to_dict()

    def _get_model(self, extension):
        """Build a model using the extension parameters."""
        model_parameters = self._unflatten_dict(extension)
        model_name = get_qualified_name(self.modeler.model)

        model_parameters['fitted'] = True
        model_parameters['type'] = model_name

        if model_name == GAUSSIAN_COPULA:
            model_parameters = self._unflatten_gaussian_copula(model_parameters)

        return self.modeler.model.from_dict(model_parameters)

    @staticmethod
    def _get_missing_valid_rows(synthesized, drop_indices, valid_rows, num_rows):
        """

        Args:
            synthesized (pandas.DataFrame)

        Returns:
            tuple[int, pandas.DataFrame]: Amount of missing values and actual valid rows
        """
        valid_rows = pd.concat([valid_rows, synthesized[~drop_indices].copy()])
        valid_rows = valid_rows.reset_index(drop=True)

        if len(valid_rows) > num_rows:
            valid_rows = valid_rows.iloc[:num_rows]

        missing_rows = num_rows - valid_rows.shape[0]

        return missing_rows, valid_rows

    @staticmethod
    def _sample_model(model, num_rows, columns):
        """Sample from model and format into pandas.DataFrame.

        Args:
            model(copula.multivariate.base): Fitted model.
            num_rows(int): Number of rows to sample.
            columns(Iterable): Column names for the sampled rows.

        Returns:
            pd.DataFrame: Sampled rows.

        """
        if get_qualified_name(model) == 'copulas.multivariate.vine.VineCopula':
            synthesized = [model.sample(num_rows).tolist() for row in range(num_rows)]

        else:
            synthesized = model.sample(num_rows)

        return pd.DataFrame(synthesized, columns=columns)

    def _sample_valid_rows(self, model, num_rows, table_name):
        """Sample using `model` and discard invalid values until having `num_rows`.

        Args:
            model (copula.multivariate.base): Fitted model.
            num_rows (int): Number of rows to sample.
            table_name (str): name of table to synthesize.

        Returns:
            pandas.DataFrame: Sampled rows, shape (, num_rows)
        """

        if model and model.fitted:
            pk_name, pk_values = self._get_primary_keys(table_name, num_rows)

            columns = self.modeler.tables[table_name].columns
            synthesized = self._sample_model(model, num_rows, columns)
            valid_rows = pd.DataFrame(columns=columns)
            drop_indices = pd.Series(False, index=synthesized.index)

            categorical_columns = []
            # table_metadata = self._get_table_meta(self.dn.meta, table_name)
            table_metadata = self.dn.get_meta_data(table_name)

            for column_name, field in table_metadata['fields'].items():
                if field['type'] == 'categorical':
                    categorical_columns.append(column_name)
                    column = synthesized[column_name]
                    filtered_values = ((column < 0) | (column > 1))

                    if filtered_values.any():
                        drop_indices |= filtered_values

            missing_rows, valid_rows = self._get_missing_valid_rows(
                synthesized, drop_indices, valid_rows, num_rows)

            while missing_rows:
                synthesized = self._sample_model(model, missing_rows, columns)
                drop_indices = pd.Series(False, index=synthesized.index)

                for column_name in categorical_columns:
                    column = synthesized[column_name]
                    filtered_values = ((column < 0) | (column > 1))

                    if filtered_values.any():
                        drop_indices |= filtered_values

                missing_rows, valid_rows = self._get_missing_valid_rows(
                    synthesized, drop_indices, valid_rows, num_rows)

            if pk_name:
                valid_rows[pk_name] = pk_values

            return valid_rows

        else:
            parents = bool(self.dn.get_parents(table_name))
            raise ValueError(MODEL_ERROR_MESSAGES[parents])

    def _sample_children(self, table_name, sampled):
        table_rows = sampled[table_name]
        for child_name in self.dn.get_children(table_name):
            for _, row in table_rows.iterrows():
                self._sample(child_name, table_name, row, sampled)

    def _sample(self, table_name, parent_name, parent_row, sampled):
        extension = self._get_extension(parent_row, table_name, parent_name)

        model = self._get_model(extension)
        num_rows = max(round(extension['child_rows']), 0)

        sampled_rows = self._sample_valid_rows(model, num_rows, table_name)

        parent_id, foreign_key = self.dn.foreign_keys[(table_name, parent_name)]
        sampled_rows[foreign_key] = parent_row[parent_id]

        previous = sampled.get(table_name)
        if previous is None:
            sampled[table_name] = sampled_rows
        else:
            sampled[table_name] = pd.concat([previous, sampled_rows]).reset_index(drop=True)

        self._sample_children(table_name, sampled)

    def sample_rows(self, table_name, num_rows, reset_primary_keys=False,
                    sample_children=True, sampled_data=None):

        if reset_primary_keys:
            self._reset_primary_keys_generators()

        model = self.modeler.models[table_name]

        sampled_rows = self._sample_valid_rows(model, num_rows, table_name)

        parents = self.dn.get_parents(table_name)
        if parents:
            parent_name = list(parents)[0]
            foreign_key = self.dn.foreign_keys[(table_name, parent_name)][1]
            parent_id = self._get_primary_keys(parent_name, 1)[1][0]
            sampled_rows[foreign_key] = parent_id

        if sample_children:
            if sampled_data is None:
                sampled_data = dict()

            sampled_data[table_name] = sampled_rows

            self._sample_children(table_name, sampled_data)

            for table, sampled_rows in sampled_data.items():
                sampled_data[table] = self._transform_synthesized_rows(sampled_rows, table)

            return sampled_data

        else:
            return self._transform_synthesized_rows(sampled_rows, table_name)

    def sample_table(self, table_name, reset_primary_keys=False):
        """Sample a table equal to the size of the original.

        Args:
            table_name(str): Name of table to synthesize.
            reset_primary_keys(bool): Wheter or not reset the primary key generators.

        Returns:
            pandas.DataFrame: Synthesized table.
        """
        num_rows = self.dn.tables[table_name].data.shape[0]
        sampled_data = self.sample_rows(
            table_name,
            num_rows,
            sample_children=False,
            reset_primary_keys=reset_primary_keys
        )
        return sampled_data

    def sample_all(self, num_rows=5, reset_primary_keys=False):
        """Samples the entire database.

        Args:
            num_rows(int): Number of rows to be sampled on the parent tables.
            reset_primary_keys(bool): Wheter or not reset the primary key generators.

        Returns:
            dict: Tables sampled.

        `sample_all` returns a dictionary with all the tables of the dataset sampled.
        The amount of rows sampled will depend from table to table, and is only guaranteed
        to match `num_rows` on tables without parents.

        This is this way because the children tables are created modelling the relation
        thet have with their parent tables, so it's behavior may change from one table to another.
        """
        if reset_primary_keys:
            self._reset_primary_keys_generators()

        tables = self.dn.tables

        sampled_data = dict()
        for table in tables:
            if not self.dn.get_parents(table):
                self.sample_rows(table, num_rows, sampled_data=sampled_data)

        return sampled_data
