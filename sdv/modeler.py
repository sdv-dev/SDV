import logging
import pickle

import numpy as np
import pandas as pd
from copulas import EPSILON, get_qualified_name
from copulas.multivariate import GaussianMultivariate, TreeTypes
from copulas.univariate import GaussianUnivariate
from rdt.transformers.positive_number import PositiveNumberTransformer

# Configure logger
logger = logging.getLogger(__name__)

DEFAULT_MODEL = GaussianMultivariate
DEFAULT_DISTRIBUTION = GaussianUnivariate
IGNORED_DICT_KEYS = ['fitted', 'distribution', 'type']

MODELLING_ERROR_MESSAGE = (
    'There was an error while trying to model the database. If you are using a custom '
    'distribution or model, please try again using the default ones. If the problem persist, '
    'please report it here:\nhttps://github.com/HDI-Project/SDV/issues.\n'
)


class Modeler:
    """Class responsible for modeling database.

    Args:
        data_navigator (DataNavigator):  object for the dataset.
        model (type): Class of model to use.
        distribution (type): Class of distribution to use. Will be deprecated shortly.
        model_kwargs (dict): Keyword arguments to pass to model.
    """

    DEFAULT_PRIMARY_KEY = 'GENERATED_PRIMARY_KEY'

    def __init__(self, data_navigator, model=DEFAULT_MODEL, distribution=None, model_kwargs=None):
        """Instantiates a modeler object."""
        self.tables = {}
        self.models = {}
        self.child_locs = {}  # maps table->{child: col #}
        self.dn = data_navigator
        self.model = model

        if distribution and model != DEFAULT_MODEL:
            raise ValueError(
                '`distribution` argument is only suported for `GaussianMultivariate` model.')

        if distribution is not None:
            distribution = get_qualified_name(distribution)
        else:
            distribution = get_qualified_name(DEFAULT_DISTRIBUTION)

        if not model_kwargs:
            if model == DEFAULT_MODEL:
                model_kwargs = {'distribution': distribution}

            else:
                model_kwargs = {'vine_type': TreeTypes.REGULAR}

        self.model_kwargs = model_kwargs

    def save(self, file_name):
        """Saves model to file destination.

        Args:
            file_name (string): path to store file
        """
        with open(file_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_name):
        """Load model from filename.

        Args:
            file_name (string): path of file to load
        """
        with open(file_name, 'rb') as input:
            return pickle.load(input)

    def get_pk_value(self, pk, index, mapping):
        if pk == self.DEFAULT_PRIMARY_KEY:
            val = pk + str(index)
        else:
            val = mapping[pk]

        return val

    @classmethod
    def _flatten_array(cls, nested, prefix=''):
        """Return a dictionary with the values of the given nested array.

        Args:
            nested (list, np.array): Iterable to flatten.
            prefix (str): Name to append to the array indices.

        Returns:
            dict
        """
        result = {}
        for index in range(len(nested)):
            prefix_key = '__'.join([prefix, str(index)]) if len(prefix) else str(index)

            if isinstance(nested[index], (list, np.ndarray)):
                result.update(cls._flatten_array(nested[index], prefix=prefix_key))

            else:
                result[prefix_key] = nested[index]

        return result

    @classmethod
    def _flatten_dict(cls, nested, prefix=''):
        """Return a flatten dict from a nested one.

        This method returns a flatten version of a dictionary, concatenating key names with
        double underscores, that is:

        Args:
            nested (dict): Original dictionary to flatten.
            prefix (str): Prefix to append to key name

        Returns:
            dict: Flattened dictionary. That is, all its keys hold a primitive value.
        """
        result = {}

        for key, value in nested.items():
            prefix_key = '__'.join([prefix, str(key)]) if len(prefix) else key

            if key in IGNORED_DICT_KEYS and not isinstance(value, (dict, list)):
                continue

            elif isinstance(value, dict):
                result.update(cls._flatten_dict(value, prefix_key))

            elif isinstance(value, (np.ndarray, list)):
                result.update(cls._flatten_array(value, prefix_key))

            else:
                result[prefix_key] = value

        return result

    def _get_model_dict(self, data):
        """Fit and  serialize  a model and flatten its parameters into an array.

        Args:
            data(pandas.DataFrame): Dataset to fit the model to.

        Returns:
            dict: Flattened parameters for model.

        """
        model = self.fit_model(data)

        if self.model == DEFAULT_MODEL:
            values = []
            triangle = np.tril(model.covariance)

            for index, row in enumerate(triangle.tolist()):
                values.append(row[:index + 1])

            model.covariance = np.array(values)
            if self.model_kwargs['distribution'] == get_qualified_name(DEFAULT_DISTRIBUTION):
                transformer = PositiveNumberTransformer({
                    'name': 'field',
                    'type': 'number'
                })

                for distribution in model.distribs.values():
                    column = pd.DataFrame({'field': [distribution.std]})
                    distribution.std = transformer.reverse_transform(column).loc[0, 'field']

        return self._flatten_dict(model.to_dict())

    def get_foreign_key(self, fields, primary):
        """Get foreign key from primary key.

        Args:
            fields (dict): metadata `fields` key for a given table.
            primary (str): Name of primary key in original table.

        Return:
            str: Name of foreign key in current table.
        """
        for field_key in fields:
            field = fields[field_key]
            ref = field.get('ref')
            if ref and ref['field'] == primary:
                foreign = field['name']
                return foreign

    @staticmethod
    def impute_table(table):
        """Fill in any NaN values in a table.

        Args:
            table(pandas.DataFrame): Table to fill NaN values

        Returns:
            pandas.DataFrame
        """
        values = {}

        for column in table.loc[:, table.isnull().any()].columns:
            if table[column].dtype in [np.float64, np.int64]:
                value = table[column].mean()

            if not pd.isnull(value or np.nan):
                values[column] = value
            else:
                values[column] = 0

        table = table.fillna(values)

        # There is an issue when using KDEUnivariate modeler in tables with childs
        # As the extension columns would have constant values, that make it crash
        # This is a temporary fix while https://github.com/DAI-Lab/Copulas/issues/82 is solved.
        first_index = table.index[0]
        constant_columns = table.loc[:, (table == table.loc[first_index]).all()].columns
        for column in constant_columns:
            table.loc[first_index, column] = table.loc[first_index, column] + EPSILON

        return table

    def fit_model(self, data):
        """Returns an instance of self.model fitted with the given data.

        Args:
            data (pandas.DataFrame): Data to train the model with.

        Returns:
            model: Instance of self.model fitted with data.
        """
        model = self.model(**self.model_kwargs)
        model.fit(data)

        return model

    def _create_extension(self, foreign, transformed_child_table, table_info):
        """Return the flattened model from a dataframe.

        Args:
            foreign(pandas.DataFrame): Object with Index of elements from children table elements
                                       of a given foreign_key.
            transformed_child_table(pandas.DataFrame): Table of data to fil
            table_info (tuple[str, str]): foreign_key and child table names.

        Returns:
            pd.Series or None : Parameter extension if it can be generated, None elsewhere.
            """

        foreign_key, child_name = table_info
        try:
            child_rows = transformed_child_table.loc[foreign.index].copy()
            if foreign_key in child_rows:
                child_rows = child_rows.drop(foreign_key, axis=1)

        except KeyError:
            return None

        num_child_rows = len(child_rows)

        if num_child_rows:
            clean_df = self.impute_table(child_rows)
            extension = self._get_model_dict(clean_df)
            extension['child_rows'] = num_child_rows

            extension = pd.Series(extension)
            extension.index = child_name + '__' + extension.index

            return extension

        return None

    def _get_extensions(self, pk, children):
        """Generate list of extension for child tables.

        Args:
            pk (str): Name of the primary_key column in the parent table.
            children (set[str]): Names of the children.

        Returns: list(pandas.DataFrame)

        Each element of the list is generated for one single children.
        That dataframe should have as index.name the `foreign_key` name, and as index
        it's values.
        The values for a given index is generated by flattening a model fit with the related
        data to that index in the children table.
        """
        extensions = []

        # find children that ref primary key
        for child in children:
            child_table = self.dn.tables[child].data
            child_meta = self.dn.tables[child].meta

            fields = child_meta['fields']
            fk = self.get_foreign_key(fields, pk)

            if not fk:
                continue

            # check if leaf node
            if not self.dn.get_children(child):
                transformed_child_table = self.dn.transformed_data[child]

            else:
                transformed_child_table = self.tables[child]

            table_info = (fk, '__' + child)

            foreign_key_values = child_table[fk].unique()
            parameters = {}

            for foreign_key in foreign_key_values:
                foreign_index = child_table[child_table[fk] == foreign_key]
                parameter = self._create_extension(
                    foreign_index, transformed_child_table, table_info)

                if parameter is not None:
                    parameters[foreign_key] = parameter.to_dict()

            extension = pd.DataFrame(parameters).T
            extension.index.name = pk

            if len(extension):
                extensions.append(extension)

        return extensions

    def CPA(self, table):
        """Run CPA algorithm on a table.

        Conditional Parameter Aggregation. It will take the table's children and generate
        extensions (parameters from modelling the related children for each foreign key)
        and merge them to the original `table`.

        After the extensions are created, `extended_table` is modified in order for the extensions
        to be merged. As the extensions are returned with an index consisting of values of the
        `primary_key` of the parent table, we need to make sure that same values are present in
        `extended_table`. The values couldn't be present in two situations:

        - They weren't numeric, and have been transformed.
        - They weren't transformed, and therefore are not present on `extended_table`

        Args:
            table (string): name of table.

        Returns:
            None
        """
        logger.info('Modeling %s', table)
        # Grab table
        tables = self.dn.tables
        # grab table from self.tables if it is not a leaf
        # o.w. grab from data
        children = self.dn.get_children(table)
        table_meta = tables[table].meta
        # get primary key
        pk = table_meta.get('primary_key', self.DEFAULT_PRIMARY_KEY)

        # start with transformed table
        extended_table = self.dn.transformed_data[table]
        extensions = self._get_extensions(pk, children)

        if extensions:
            original_pk = tables[table].data[pk]
            transformed_pk = None

            if pk in extended_table:
                transformed_pk = extended_table[pk].copy()

            if (pk not in extended_table) or (not extended_table[pk].equals(original_pk)):
                extended_table[pk] = original_pk

            # add extensions
            for extension in extensions:
                extended_table = extended_table.merge(extension.reset_index(), how='left', on=pk)

            if transformed_pk is not None:
                extended_table[pk] = transformed_pk
            else:
                extended_table = extended_table.drop(pk, axis=1)

        self.tables[table] = extended_table

    def RCPA(self, table):
        """Recursively calls CPA starting at table.

        Args:
            table (string): name of table to start from.
        """
        children = self.dn.get_children(table)

        for child in children:
            self.RCPA(child)

        self.CPA(table)

    def model_database(self):
        """Use RCPA and store model for database."""
        for table in self.dn.tables:
            if not self.dn.get_parents(table):
                self.RCPA(table)

        for table in self.tables:
            clean_table = self.impute_table(self.tables[table])
            self.models[table] = self.fit_model(clean_table)

        logger.info('Modeling Complete')
