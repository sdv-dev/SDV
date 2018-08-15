import logging
import os
import pickle

import pandas as pd
from copulas.multivariate.GaussianCopula import GaussianCopula  # noqa F401 FIXME
from copulas.univariate.GaussianUnivariate import GaussianUnivariate  # noqa F401

# Configure logger
logger = logging.getLogger(__name__)


class Modeler:
    """ Class responsible for modeling database """
    DEFAULT_MODEL_PARAMS = ['GaussianUnivariate']
    DEFAULT_PRIMARY_KEY = 'GENERATED_PRIMARY_KEY'
    FILE_SUFFIX = '.pkl'
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, data_navigator, model_type='GaussianCopula', model_params=None):
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

        # create dict mapping row id to conditional data
        map_keys = [self.get_pk_value(pk, i, table_df.loc[i, :]) for i in range(num_rows)]
        conditional_data_map = {key: [] for key in map_keys}

        # get conditional data for val
        self._get_conditional_data(conditional_data_map, pk, children, table)
        extended_table = pd.DataFrame()
        # create extended table
        for i in range(num_rows):
            # change to be transformed table
            orig_row = table_df.loc[i, :]
            row = self.dn.transformed_data[table].loc[i, :]

            val = self.get_pk_value(pk, i, orig_row)

            # make sure val isn't none
            if pd.isnull(val):
                continue

            for extension in conditional_data_map[val]:
                row = row.append(extension)

            # make sure row doesn't have nans
            if not row.isnull().values.any():
                extended_table = extended_table.append(row, ignore_index=True)
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
        if self.model_type == "GaussianCopula":
            params = []
            params = params + list(model.cov_matrix.flatten())
            params = params + model.means

            for key in model.distribs:
                col_model = model.distribs[key]
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

    def _get_conditional_data(self, conditional_data_map, pk, children, table):
        """ loops through children looking for rows that
        reference the value
        """
        # keep track of which columns belong to which child
        start = 0
        end = 0
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

            # add model params conditional data
            for val in conditional_data_map:
                # grab the tranformed table columns instead
                extension = transformed_child_table[child_table[fk] == val]

                if extension.empty:
                    continue

                # remove column of foreign key
                model = self.get_model()
                clean_extension = self.impute_table(extension)
                model.fit(clean_extension)
                flattened_extension = self.flatten_model(model, child)
                # keep track of child column indices
                end = max(end, start + len(flattened_extension))

                if table in self.child_locs:
                    self.child_locs[table][child] = (start, end)
                else:
                    self.child_locs[table] = {child: (start, end)}
                # rename columns
                flattened_extension.index = range(start, end)
                conditional_data_map[val].append(flattened_extension)
            start = end

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
