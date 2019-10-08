import logging
import pickle

import numpy as np
import pandas as pd
from copulas import get_qualified_name
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate

LOGGER = logging.getLogger(__name__)

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
        metadata (DataNavigator):
            Dataset Metadata.
        model (type):
            Class of model to use.
        distribution (type):
            Class of distribution to use. Will be deprecated shortly.
        model_kwargs (dict):
            Keyword arguments to pass to model.
    """

    DEFAULT_PRIMARY_KEY = 'GENERATED_PRIMARY_KEY'

    def __init__(self, metadata, model=DEFAULT_MODEL, distribution=None, model_kwargs=None):
        """Instantiates a modeler object."""
        self.models = dict()
        self.metadata = metadata
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

    def get_primary_key_value(self, primary_key, index, mapping):
        if primary_key == self.DEFAULT_PRIMARY_KEY:
            val = primary_key + str(index)
        else:
            val = mapping[primary_key]

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

                for distribution in model.distribs.values():
                    distribution.std = np.log(distribution.std)

        return self._flatten_dict(model.to_dict())

    def get_foreign_key(self, fields, primary):
        """Get foreign key from primary key.

        Args:
            fields (dict):
                metadata `fields` key for a given table.
            primary (str):
                Name of primary key in original table.

        Return:
            str: Name of foreign key in current table.
        """
        for field in fields.values():
            ref = field.get('ref')
            if ref and ref['field'] == primary:
                foreign = field['name']
                return foreign

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

    def _create_extension(self, foreign, child_table_data, table_info):
        """Return the flattened model from a dataframe.

        Args:
            foreign(pandas.DataFrame): Object with Index of elements from children table elements
                                       of a given foreign_key.
            child_table_data(pandas.DataFrame): Table of data to fil
            table_info (tuple[str, str]): foreign_key and child table names.

        Returns:
            pd.Series or None : Parameter extension if it can be generated, None elsewhere.
            """

        foreign_key, child_name = table_info
        try:
            child_rows = child_table_data.loc[foreign.index].copy()
            if foreign_key in child_rows:
                child_rows = child_rows.drop(foreign_key, axis=1)

        except KeyError:
            return None

        num_child_rows = len(child_rows)

        if num_child_rows:
            extension = self._get_model_dict(child_rows)
            extension['child_rows'] = num_child_rows

            extension = pd.Series(extension)
            extension.index = child_name + '__' + extension.index

            return extension

        return None

    def _get_extensions(self, parent, children, tables):
        """Generate list of extension for child tables.

        Each element of the list is generated for one single children.
        That dataframe should have as index.name the `foreign_key` name, and as index
        it's values.

        The values for a given index are generated by flattening a model fitted with
        the related data to that index in the children table.

        Args:
            parent (str):
                Name of the parent table.
            children (set[str]):
                Names of the children.
            tables (dict):
                Previously processed tables.
        """
        extensions = []

        for child in children:
            child_table = self.metadata.get_table_data(child)

            foreign_key = self.metadata.get_foreign_key(parent, child)

            # check if leaf node
            if not self.metadata.get_children(child):
                child_table_data = self.metadata.get_table_data(child, transform=True)
            else:
                child_table_data = tables[child]

            table_info = (foreign_key, '__' + child)

            foreign_key_values = child_table[foreign_key].unique()
            parameters = {}

            for foreign_key_value in foreign_key_values:
                foreign_index = child_table[child_table[foreign_key] == foreign_key_value]
                parameter = self._create_extension(
                    foreign_index, child_table_data, table_info)

                if parameter is not None:
                    parameters[foreign_key_value] = parameter.to_dict()

            extension = pd.DataFrame(parameters).T

            if len(extension):
                extensions.append(extension)

        return extensions

    def cpa(self, table_name, tables):
        """Run CPA algorithm on a table.

        Conditional Parameter Aggregation. It will take the table children and generate
        extensions (parameters from modelling the related children for each foreign key)
        and merge them into the original table.

        Args:
            table_name (string):
                name of table.
            tables (dict):
                previously processed tables.

        Returns:
            None
        """
        LOGGER.info('Modeling %s', table_name)

        table = self.metadata.get_table_data(table_name, transform=True)

        children = self.metadata.get_children(table_name)
        primary_key = self.metadata.get_primary_key(table_name) or self.DEFAULT_PRIMARY_KEY

        extensions = self._get_extensions(table_name, children, tables)
        if extensions:
            original_primary_key = table[primary_key]
            transformed_primary_key = None

            if primary_key in table:
                transformed_primary_key = table[primary_key].copy()

            if (primary_key not in table) or (not table[primary_key].equals(original_primary_key)):
                table[primary_key] = original_primary_key

            # add extensions
            for extension in extensions:
                table = table.merge(extension, how='left', left_on=primary_key, right_index=True)

            if transformed_primary_key is not None:
                table[primary_key] = transformed_primary_key
            else:
                table = table.drop(primary_key, axis=1)

        return table

    def rcpa(self, table_name, tables):
        """Recursively calls CPA starting at table.

        Args:
            table_name (string):
                name of table to start from.
        """
        children = self.metadata.get_children(table_name)

        for child in children:
            self.rcpa(child, tables)

        table = self.cpa(table_name, tables)
        tables[table_name] = table

    @staticmethod
    def _impute(data):
        for column in data:
            column_data = data[column]
            if column_data.dtype in (np.int, np.float):
                fill_value = column_data.mean()
            else:
                fill_value = column_data.mode()[0]

            data[column] = data[column].fillna(fill_value)

        return data

    def model_database(self):
        """Use RCPA and store model for database."""
        self.tables = dict()
        for table_name in self.metadata.get_table_names():
            if not self.metadata.get_parents(table_name):
                self.rcpa(table_name, self.tables)

        for name, data in self.tables.items():
            data = self._impute(data)
            self.models[name] = self.fit_model(data)

        LOGGER.info('Modeling Complete')
