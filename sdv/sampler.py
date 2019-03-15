import random

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
        self.sampled = {}  # table_name -> [(primary_key, generated_row)]
        self.primary_key = {}

    @staticmethod
    def update_mapping_list(mapping, key, value):
        """Append value on mapping[key] if exists, create it otherwise."""
        item = mapping.get(key)

        if item:
            item.append(value)

        else:
            mapping[key] = [value]

        return mapping

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

    def _get_table_meta(self, metadata, table_name):
        """Return metadata  get table meta for a given table name.

        Args:
            metadata (dict): Metadata for dataset.
            table_name (str): Name of table to get metadata from.

        Returns:
            dict: Metadata for given table.
        """
        for table in metadata['tables']:
            if table['name'] == table_name:
                return table

        return None

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

    def transform_synthesized_rows(self, synthesized, table_name, num_rows):
        """Add primary key and reverse transform synthetized data.

        Args:
            synthesized(pandas.DataFrame): Generated data from model
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

            synthesized[primary_key] = pd.Series(values)

            if (node['type'] == 'number') and (node['subtype'] == 'integer'):
                synthesized[primary_key] = pd.to_numeric(synthesized[primary_key])

        sample_info = (primary_key, synthesized)
        self.sampled = self.update_mapping_list(self.sampled, table_name, sample_info)

        # filter out parameters
        labels = list(self.dn.tables[table_name].data)
        reverse_columns = [
            transformer[1] for transformer in self.dn.ht.transformers
            if table_name in transformer
        ]

        text_filled = self._fill_text_columns(synthesized, labels, table_name)

        # reverse transform data
        reversed_data = self.dn.ht.reverse_transform_table(text_filled[reverse_columns], orig_meta)

        synthesized.update(reversed_data)
        return synthesized[labels]

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

    @staticmethod
    def generate_keys(prefix=''):
        def f(row):
            parts = [str(row[key]) for key in row.keys() if row[key] is not None]
            if prefix:
                parts = [prefix] + parts

            return '__'.join(parts)

        return f

    @classmethod
    def _get_sorted_keys(cls, _dict):
        result = []
        keys = list(_dict.keys())

        if not keys:
            return []

        serie = pd.Series(keys)
        df = pd.DataFrame(serie.str.split('__').values.tolist())
        uniques = df[0].unique()

        for value in uniques:
            index = df[df[0] == value].index
            _slice = df.loc[index, range(1, df.shape[1])].copy()

            try:
                for column in _slice.columns:
                    _slice[column] = _slice[column].astype(int)

            except (ValueError, TypeError):
                pass

            df.drop(index, inplace=True)
            _slice = _slice.sort_values(list(range(1, df.shape[1])))
            result += _slice.apply(cls.generate_keys(value), axis=1).values.tolist()

        df = df.sort_values(list(range(df.shape[1])))
        result += df.apply(cls.generate_keys(), axis=1).values.tolist()

        return result

    def _unflatten_dict(self, flat, table_name=''):
        """Transform a flattened dict into its original form.

        Works in exact opposite way that `sdv.Modeler._flatten_dict`.

        Args:
            flat (dict): Flattened dict.

        """
        result = {}
        children = self.dn.get_children(table_name)
        keys = self._get_sorted_keys(flat)

        for key in keys:
            path = key.split('__')

            if any(['__{}__'.format(child) in key for child in children]):
                path = [
                    path[0],
                    '__'.join(path[1: -1]),
                    path[-1]
                ]

            value = flat[key]
            walked = result
            for step, name in enumerate(path):

                if isinstance(walked, dict) and name in walked:
                    walked = walked[name]
                    continue

                elif isinstance(walked, list) and len(walked) and len(walked) - 1 >= int(name):
                    walked = walked[int(name)]
                    continue

                else:
                    if name.isdigit():
                        name = int(name)

                    if step == len(path) - 1:
                        if isinstance(walked, list):
                            walked.append(value)
                        else:
                            walked[name] = value

                    else:
                        next_step = path[step + 1]
                        if next_step.isdigit():
                            if isinstance(name, int):
                                walked.append([])
                                while len(walked) < name + 1:
                                    walked.append([])

                            else:
                                walked[name] = []

                            walked = walked[name]

                        else:
                            if isinstance(name, int):
                                walked.append({})
                            else:
                                walked[name] = {}

                            walked = walked[name]

        return result

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

    def unflatten_model(self, parent_row, table_name, parent_name):
        """ Takes the params from a generated parent row and creates a model from it.

        Args:
            parent_row (dataframe): a generated parent row
            table_name (string): name of table to make model for
            parent_name (string): name of parent table
        """

        prefix = '__{}__'.format(table_name)
        columns = [column for column in parent_row.columns if column.startswith(prefix)]
        new_columns = {column: column.replace(prefix, '') for column in columns}
        flat_parameters = parent_row.loc[:, columns]
        flat_parameters = flat_parameters.rename(columns=new_columns).to_dict('records')[0]

        model_parameters = self._unflatten_dict(flat_parameters, table_name)
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
            columns = self.modeler.tables[table_name].columns
            synthesized = self._sample_model(model, num_rows, columns)
            valid_rows = pd.DataFrame(columns=columns)
            drop_indices = pd.Series(False, index=synthesized.index)

            categorical_columns = []
            table_metadata = self._get_table_meta(self.dn.meta, table_name)

            for field in table_metadata['fields']:
                if field['type'] == 'categorical':
                    column_name = field['name']
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

            return valid_rows

        else:
            parents = bool(self.dn.get_parents(table_name))
            raise ValueError(MODEL_ERROR_MESSAGES[parents])

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
            model = self.unflatten_model(parent_row, table_name, random_parent)

            synthesized_rows = self._sample_valid_rows(model, num_rows, table_name)

            # add foreign key value to row
            fk_val = parent_row.loc[0, fk]

            # get foreign key name from current table
            foreign_key = self.dn.foreign_keys[(table_name, random_parent)][1]
            synthesized_rows[foreign_key] = fk_val

            return self.transform_synthesized_rows(synthesized_rows, table_name, num_rows)

        else:    # there is no parent
            model = self.modeler.models[table_name]
            synthesized_rows = self._sample_valid_rows(model, num_rows, table_name)

            return self.transform_synthesized_rows(synthesized_rows, table_name, num_rows)

    def sample_table(self, table_name):
        """Sample a table equal to the size of the original.

        Args:
            table_name (str): name of table to synthesize

        Returns:
            pandas.DataFrame: Synthesized table.
        """
        num_rows = self.dn.tables[table_name].data.shape[0]
        return self.sample_rows(table_name, num_rows)

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
