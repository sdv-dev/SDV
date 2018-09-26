import logging
import pickle

import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate

# Configure logger
logger = logging.getLogger(__name__)

DEFAULT_MODEL = GaussianMultivariate
DEFAULT_DISTRIBUTION = GaussianUnivariate


class Modeler:
    """Class responsible for modeling database."""

    DEFAULT_PRIMARY_KEY = 'GENERATED_PRIMARY_KEY'

    def __init__(self, data_navigator, model=DEFAULT_MODEL, distribution=DEFAULT_DISTRIBUTION):
        """Instantiates a modeler object.

        Args:
            data_navigator (DataNavigator):  object for the dataset.
            transformed_data (dict): transformed tables {table_name:dataframe}.
            model (type): Class of model to use.
            distribution (type): Class of model to use.
        """
        self.tables = {}
        self.models = {}
        self.child_locs = {}  # maps table->{child: col #}
        self.dn = data_navigator
        self.model = model
        self.distribution = distribution

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

    def flatten_model(self, model):
        """Flatten a model's parameters into an array.

        Args:
            model: a model object

        Returns:
            pd.Series: parameters for model
        """
        params = list(model.covariance.flatten())

        for col_model in model.distribs.values():
            params.extend([col_model.std, col_model.mean])

        return pd.Series(params)

    def get_foreign_key(self, fields, primary):
        """Get foreign key from primary key.

        Args:
            fields (dict): metadata's fields key for a given table.
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
            table(pandas.DataFrame):

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
            GaussianMultivariate: Fitted model.
        """
        model = self.model()
        model.fit(data)

        return model

    def _create_extension(self, df, transformed_child_table):
        """Return the flattened model from a dataframe."""
        # remove column of foreign key
        try:
            conditional_data = transformed_child_table.loc[df.index]
        except KeyError:
            return None

        clean_df = self.impute_table(conditional_data)

        return self.flatten_model(self.fit_model(clean_df))

    def _extension_from_group(self, transformed_child_table):
        """Wrapper around _create_extension to use it with pd.DataFrame.apply."""
        def f(group):
            return self._create_extension(group, transformed_child_table)
        return f

    def _get_extensions(self, pk, children, table_name):
        """Generate list of extension for child tables."""
        # keep track of which columns belong to which child
        start = 0
        end = 0
        extensions = []

        # make sure child_locs has value for table name
        self.child_locs[table_name] = self.child_locs.get(table_name, {})

        # find children that ref primary key
        for child in children:
            child_table = self.dn.tables[child].data
            child_meta = self.dn.tables[child].meta

            # check if leaf node
            if not self.dn.get_children(child):
                transformed_child_table = self.dn.transformed_data[child]

            else:
                transformed_child_table = self.tables[child]

            fields = child_meta['fields']
            fk = self.get_foreign_key(fields, pk)

            if not fk:
                continue

            extension = child_table.groupby(fk)
            extension = extension.apply(self._extension_from_group(transformed_child_table))

            if extension is not None:
                # keep track of child column indices
                end = max(end, start + extension.shape[1])

                self.child_locs[table_name][child] = (start, end)

                # rename columns
                extension.columns = range(start, end)
                extensions.append(extension)
                start = end

        return extensions

    def CPA(self, table):
        """Run CPA algorithm on a table.

        Conditional Parameter Aggregation. It will take the tab

        Args:
            table (string): name of table.

        Returns:
            None:
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
        extensions = self._get_extensions(pk, children, table)

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
        for table in self.dn.tables:
            if not self.dn.get_parents(table):
                self.RCPA(table)

        for table in self.tables:
            clean_table = self.impute_table(self.tables[table])
            self.models[table] = self.fit_model(clean_table)

        logger.info('Modeling Complete')
