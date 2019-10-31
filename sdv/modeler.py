import logging
import pickle

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate

LOGGER = logging.getLogger(__name__)

IGNORED_DICT_KEYS = ['fitted', 'distribution', 'type']


class Modeler:
    """Class responsible for modeling database.

    The ``Modeler.model_database`` applies the CPA algorithm to generate the extended table.
    Required before sampling data.

    Args:
        metadata (Metadata):
            Dataset Metadata.
        model (type):
            Class of model to use. Defaults to ``copulas.multivariate.GaussianMultivariate``.
        model_kwargs (dict):
            Keyword arguments to pass to the model. Defaults to ``None``.
    """

    def __init__(self, metadata, model=GaussianMultivariate, model_kwargs=None):
        """Instantiates a modeler object."""
        self.models = dict()
        self.metadata = metadata
        self.model = model
        self.model_kwargs = dict() if model_kwargs is None else model_kwargs

    def save(self, file_name):
        """Saves model to file destination.

        Args:
            file_name (string):
                path where to store the file.
        """
        with open(file_name, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, file_name):
        """Load model from filename.

        Args:
            file_name (str):
                path from where to load.
        """
        with open(file_name, 'rb') as input:
            return pickle.load(input)

    @classmethod
    def _flatten_array(cls, nested, prefix=''):
        """Return a dictionary with the values of the given nested array.

        Args:
            nested (list, numpy.array):
                Iterable to flatten.
            prefix (str):
                Name to append to the array indices. Defaults to ``''``.

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
        double underscores.

        Args:
            nested (dict):
                Original dictionary to flatten.
            prefix (str):
                Prefix to append to key name. Defaults to ``''``.

        Returns:
            dict:
                Flattened dictionary, where all its keys hold a primitive value.
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

    def _fit_model(self, data):
        """Returns an instance of ``self.model`` fitted with the given data.

        Args:
            data (pandas.DataFrame):
                Data to fit the model with.

        Returns:
            model:
                Instance of ``self.model`` fitted with data.
        """
        data = self._impute(data)
        model = self.model(**self.model_kwargs)
        model.fit(data)

        return model

    def _get_model_dict(self, data):
        """Fit and serialize a model and flatten its parameters into an array.

        Args:
            data (pandas.DataFrame):
                Dataset to fit the model to.

        Returns:
            dict:
                Flattened parameters for model.
        """
        model = self._fit_model(data)

        values = []
        triangle = np.tril(model.covariance)

        for index, row in enumerate(triangle.tolist()):
            values.append(row[:index + 1])

        model.covariance = np.array(values)
        for distribution in model.distribs.values():
            if distribution.std is not None:
                distribution.std = np.log(distribution.std)

        return self._flatten_dict(model.to_dict())

    def _get_extension(self, child_name, child_table, foreign_key):
        """Generate list of extension for child tables.

        Each element of the list is generated for one single children.
        That dataframe should have as ``index.name`` the ``foreign_key`` name, and as index
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
        Returns:
            pandas.DataFrame
        """
        extension_rows = list()
        foreign_key_values = child_table[foreign_key].unique()
        child_table = child_table.set_index(foreign_key)
        for foreign_key_value in foreign_key_values:
            child_rows = child_table.loc[[foreign_key_value]]
            num_child_rows = len(child_rows)
            row = self._get_model_dict(child_rows)
            row['child_rows'] = num_child_rows

            row = pd.Series(row)
            row.index = '__' + child_name + '__' + row.index
            extension_rows.append(row)

        return pd.DataFrame(extension_rows, index=foreign_key_values)

    def cpa(self, table_name, tables, foreign_key=None):
        """Run CPA algorithm.

        If ``tables`` is not loaded, load the current table.
        If the table we are processing have childs call ``cpa`` and generate extensions.
        After iterate over the childs, fit the model and return the extended table.

        Args:
            table_name (str):
                Name of table.
            tables (dict):
                Dict of tables to process.
            foreign_key (str):
                Name of the foreign key that references this table. Used only when applying
                CPA on a child table.

        Returns:
            pandas.DataFrame
        """
        LOGGER.info('Modeling %s', table_name)

        if tables:
            table = tables[table_name]
        else:
            table = self.metadata.load_table(table_name)

        extended = self.metadata.transform(table_name, table)

        primary_key = self.metadata.get_primary_key(table_name)
        if primary_key:
            extended.index = table[primary_key]
            for child_name in self.metadata.get_children(table_name):
                child_key = self.metadata.get_foreign_key(table_name, child_name)
                child_table = self.cpa(child_name, tables, child_key)
                extension = self._get_extension(child_name, child_table, child_key)
                extended = extended.merge(extension, how='left',
                                          right_index=True, left_index=True)
                extended['__' + child_name + '__child_rows'].fillna(0, inplace=True)

        self.models[table_name] = self._fit_model(extended)

        if primary_key:
            extended.reset_index(inplace=True)

        if foreign_key:
            extended[foreign_key] = table[foreign_key]

        return extended

    def model_database(self, tables=None):
        """Run CPA algorithm on all tables."""
        for table_name in self.metadata.get_table_names():
            if not self.metadata.get_parents(table_name):
                self.cpa(table_name, tables)

        LOGGER.info('Modeling Complete')
