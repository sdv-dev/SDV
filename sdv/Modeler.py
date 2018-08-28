import logging
import os
import pickle

import pandas as pd
from copulas.multivariate.gaussian import GaussianMultivariate
from copulas.univariate.gaussian import GaussianUnivariate


# Configure logger
logger = logging.getLogger(__name__)


class Modeler:
    """ Class responsible for modeling database """
    DEFAULT_MODEL_PARAMS = ['GaussianUnivariate']
    DEFAULT_PRIMARY_KEY = 'GENERATED_PRIMARY_KEY'
    FILE_SUFFIX = '.pkl'
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, data_navigator, model_type='GaussianMultivariate', model_params=None):
        """ Instantiates a modeler object
        Args:
            data_navigator: A DataNavigator object for the dataset
            transformed_data: tables post tranformation {table_name:dataframe}
            model_type: Name of model class to use
            model_params: list of parameters to use for specified model
        """
        self.tables = {}
        self.models = {}
        self.child_locs = {}  # maps table->{child: col #}
        self.dn = data_navigator
        self.model_type = model_type
        self.model_params = model_params or self.DEFAULT_MODEL_PARAMS

    def CPA(self, table):
        """ Runs CPA algorithm on a table
        Args:
            table (string): name of table
        """
        logger.info('Modeling %s', table)
        # Grab table
        tables = self.dn.tables
        # grab table from self.tables if it is not a leaf
        # o.w. grab from data
        children = self.dn.get_children(table)
        table_df, table_meta = tables[table].data, tables[table].meta
        # get primary key
        pk = table_meta.get('primary_key', self.DEFAULT_PRIMARY_KEY)

        # loop through rows of table
        num_rows = table_df.shape[0]

        # start with transformed table
        extended_table = self.dn.transformed_data[table]
        extensions = self._get_extensions(pk, children, table)
        # add extensions
        for extension in extensions:
            extended_table = pd.concat([extended_table, extension])
        self.tables[table] = extended_table

    def get_pk_value(self, pk, index, mapping):
        if pk == self.DEFAULT_PRIMARY_KEY:
            val = pk + str(index)
        else:
            val = mapping[pk]

        return val

    def RCPA(self, table):
        """ Recursively calls CPA starting at table
        Args:
            table (string): name of table to start from
        """
        children = self.dn.get_children(table)

        for child in children:
            self.RCPA(child)

        self.CPA(table)

    def model_database(self):
        """ Uses RCPA and stores model for database """
        for table in self.dn.tables:
            if not self.dn.get_parents(table):
                self.RCPA(table)

        for table in self.tables:
            table_model = self.get_model()
            clean_table = self.impute_table(self.tables[table])
            table_model.fit(clean_table)
            self.models[table] = table_model
        logger.info('Modeling Complete')

    def flatten_model(self, model, label=''):
        """ Flatten a model's parameters into an array
        Args:
            model: a model object
        Returns:
            pandas Series of parameters for model
        """
        if self.model_type == "GaussianMultivariate":
            params = []
            params = params + list(model.cov_matrix.flatten())
            params = params + model.means

            for key in model.distribs:
                col_model = model.distribs[key]
                params.append(col_model.min)
                params.append(col_model.max)
                params.append(col_model.std)
                params.append(col_model.mean)

            param_series = pd.Series(params)
            return param_series

    def save_model(self, file_destination):
        """ Saves model to file destination
        Args:
            file_destination (string): path to store file
        """
        filename = os.path.join(self.ROOT_DIR, 'models', file_destination + self.FILE_SUFFIX)

        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def get_foreign_key(self, fields, primary):
        for field_key in fields:
            field = fields[field_key]
            ref = field.get('ref')
            if ref and ref['field'] == primary:
                foreign = field['name']
                return foreign

    def _extension_from_group(self, child, transformed_child_table):
        def f(group):
            return self._create_extension(group, child, transformed_child_table)
        return f

    def _get_extensions(self, pk, children, table_name):
        """Loops through child tables and generates list of
        extension dataframes"""
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

            fk = None
            fields = child_meta['fields']
            fk = self.get_foreign_key(fields, pk)

            if not fk:
                continue
            extension = child_table.groupby(fk).apply(self._extension_from_group(child, transformed_child_table))
            if extension is not None:
                # keep track of child column indices
                end = max(end, start + extension.shape[1])

                self.child_locs[table_name][child] = (start, end)
                # rename columns
                extension.columns = range(start, end)
                extensions.append(extension)
                start = end
        return extensions

    def _create_extension(self, df, child, transformed_child_table):
        """Takes a dataframe, models it and returns the flattened model"""
        # remove column of foreign key
        try:
            conditional_data = transformed_child_table.loc[df.index]
        except KeyError:
            return None
        model = self.get_model()
        clean_df = self.impute_table(conditional_data)
        model.fit(clean_df)
        flattened_model = self.flatten_model(model, child)
        return flattened_model

    def get_model(self):
        """ Gets instance of model based on model type """
        return globals()[self.model_type]()

    def get_distribution(self):
        """ Gets instance of model based on model type """
        return globals()[self.model_params[0]]

    def impute_table(self, table):
        """ Fills in any NaN values in a table """
        values = {}

        for label in table:
            if not pd.isnull(table[label].mean()):
                values[label] = table[label].mean()
            else:
                values[label] = 0

        imputed_table = table.fillna(values)
        return imputed_table
