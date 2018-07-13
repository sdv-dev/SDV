import pandas as pd
import pickle
import os.path as op
from copulas.multivariate.GaussianCopula import GaussianCopula
from copulas.univariate.GaussianUnivariate import GaussianUnivariate


class Modeler:
    """ Class responsible for modeling database """
    def __init__(self, data_navigator,
                 model_type='GaussianCopula',
                 model_params=None):
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
        # set default model params if necessary
        if model_params is None:
            self.model_params = ['GaussianUnivariate']
        else:
            self.model_params = model_params

    def CPA(self, table):
        """ Runs CPA algorithm on a table
        Args:
            table (string): name of table
        """
        # Grab table
        data = self.dn.data
        # grab table from self.tables if it is not a leaf
        # o.w. grab from data
        children = self.dn.get_children(table)
        table_df, table_meta = data[table]
        # get primary key
        if 'primary_key' not in table_meta:
            # there are no references to the table
            # have fake primary key
            pk = 'GENERATED_PRIMARY_KEY'
            # return
        else:
            pk = table_meta['primary_key']
        # loop through rows of table
        num_rows = table_df.shape[0]
        # create dict mapping row id to conditional data
        conditional_data_map = {}
        for i in range(num_rows):
            row = table_df.loc[i, :]
            if pk == 'GENERATED_PRIMARY_KEY':
                val = pk + str(i)
            else:
                # get specific value
                val = row[pk]
            conditional_data_map[val] = []
        # get conditional data for val
        self._get_conditional_data(conditional_data_map,
                                   pk, children, table)
        extended_table = pd.DataFrame([])
        # create extended table
        for i in range(num_rows):
            # change to be transformed table
            orig_row = table_df.loc[i, :]
            row = self.dn.transformed_data[table].loc[i, :]
            if pk == 'GENERATED_PRIMARY_KEY':
                val = pk + str(i)
            else:
                # get specific value
                val = orig_row[pk]
            # make sure val isn't none
            if pd.isnull(val):
                continue
            for extension in conditional_data_map[val]:
                row = row.append(extension)
            # make sure row doesn't have nans
            if not row.isnull().values.any():
                extended_table = extended_table.append(row, ignore_index=True)
        self.tables[table] = extended_table

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
        for table in self.dn.data:
            if self.dn.get_parents(table) == []:
                self.RCPA(table)
        for table in self.tables:
            table_model = self._get_model(self.model_type)()
            clean_table = self.dn.ht.impute_table(self.tables[table])
            table_model.fit(clean_table)
            self.models[table] = table_model

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
        suffix = '.pkl'
        filename = op.join('models', file_destination + suffix)
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def _get_conditional_data(self, conditional_data_map, pk, children, table):
        """ loops through children looking for rows that
        reference the value
        """
        # keep track of which columns belong to which child
        start = 0
        end = 0
        # find children that ref primary key
        for child in children:
            child_table, child_meta = self.dn.data[child]
            transformed_child_table = self.dn.transformed_data[child]
            fk = None
            fields = child_meta['fields']
            for field_key in fields:
                field = fields[field_key]
                if 'ref' in field:
                    ref = field['ref']
                    if ref['field'] == pk:
                        fk = field['name']
            # fk should be found by this point
            if fk is None:
                continue
            # add model params conditional data
            for val in conditional_data_map:
                # grab the tranformed table columns instead
                extension = transformed_child_table[child_table[fk] == val]
                if extension.empty:
                    continue
                # remove column of foreign key
                model = self._get_model(self.model_type)()
                clean_extension = self.dn.ht.impute_table(extension)
                model.fit(clean_extension)
                flattened_extension = self.flatten_model(model, child)
                # keep track of child column indices
                end = max(end, start + len(flattened_extension))
                if table in self.child_locs:
                    self.child_locs[table][child] = (start, end)
                else:
                    self.child_locs[table] = {child: (start, end)}
                conditional_data_map[val].append(flattened_extension)
            start = end

    def _get_model(self, model_name):
        """ Gets instance of model from name of model """
        return globals()[model_name]

    def get_model(self):
        """ Gets instance of model based on model type """
        return globals()[self.model_type]

    def get_distribution(self):
        """ Gets instance of model based on model type """
        return globals()[self.model_params[0]]
