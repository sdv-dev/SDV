import logging
import pickle

import numpy as np
import pandas as pd
from copulas import get_qualified_name
from copulas.multivariate import GaussianMultivariate, TreeTypes
from copulas.univariate import GaussianUnivariate

# Configure logger
logger = logging.getLogger(__name__)

DEFAULT_MODEL = GaussianMultivariate
DEFAULT_DISTRIBUTION = GaussianUnivariate
IGNORED_DICT_KEYS = ['fitted', 'distribution', 'type']

MODELLING_ERROR_MESSAGE = (
    'There was an error while trying to model the database. If you are using a custom'
    'distribution or model, please try again using the default ones. If the problem persist,'
    'please report it here: https://github.com/HDI-Project/SDV/issues'
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
        """Instantiates a modeler object.

        """
        self.tables = {}
        self.models = {}
        self.child_locs = {}  # maps table->{child: col #}
        self.dn = data_navigator
        self.model = model

        if distribution and model != DEFAULT_MODEL:
            raise ValueError(
                '`distribution` argument is only suported for `GaussianMultivariate` model.')

        if distribution:
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

        for key in nested.keys():
            prefix_key = '__'.join([prefix, str(key)]) if len(prefix) else key

            if key in IGNORED_DICT_KEYS:
                continue

            elif isinstance(nested[key], dict):
                result.update(cls._flatten_dict(nested[key], prefix_key))

            elif isinstance(nested[key], (np.ndarray, list)):
                result.update(cls._flatten_array(nested[key], prefix_key))

            else:
                result[prefix_key] = nested[key]

        return result

    @classmethod
    def flatten_model(cls, model, name=''):
        """Flatten a model's parameters into an array.

        Args:
            model(self.model): Instance of model.
            name (str): Prefix to the parameter name.

        Returns:
            pd.Series: parameters for model
        """

        return pd.Series(cls._flatten_dict(model.to_dict(), name))

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

        for label in table:
            value = table[label].mean()

            if not pd.isnull(value):
                values[label] = value
            else:
                values[label] = 0

        return table.fillna(values)

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
            table_info (tuple(str, str)): foreign_key and child table names.

        Returns:
            pd.Series : Parameter extension
            """

        foreign_key, child_name = table_info
        try:
            conditional_data = transformed_child_table.loc[foreign.index].copy()
            conditional_data = conditional_data.drop(foreign_key, axis=1)

        except KeyError:
            return None

        clean_df = self.impute_table(conditional_data)
        return self.flatten_model(self.fit_model(clean_df), child_name)

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
            extension.index.name = fk

            if len(extension):
                extensions.append(extension)

        return extensions

    def CPA(self, table):
        """Run CPA algorithm on a table.

        Conditional Parameter Aggregation. It will take the table's children and generate
        extensions (parameters from modelling the related children for each foreign key)
        and merge them to the original `table`

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

        # add extensions
        for extension in extensions:
            extended_table = extended_table.merge(extension.reset_index(), how='left', on=pk)

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
        try:
            for table in self.dn.tables:
                if not self.dn.get_parents(table):
                    self.RCPA(table)

            for table in self.tables:
                clean_table = self.impute_table(self.tables[table])
                self.models[table] = self.fit_model(clean_table)

        except (ValueError, np.linalg.linalg.LinAlgError):
            ValueError(MODELLING_ERROR_MESSAGE)

        logger.info('Modeling Complete')
