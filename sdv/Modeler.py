import pandas as pd
from copulas.multivariate.GaussianCopula import GaussianCopula
from copulas.univariate.GaussianUnivariate import GaussianUnivariate
from rdt.hyper_transformer import HyperTransformer


class Modeler:
    """ Class responsible for modeling database """
    def __init__(self, data_navigator, transformed_data=None,
                 model_type='GaussianCopula',
                 distribution='GaussianUnivariate'):
        """ Instantiates a modeler object
        Args:
            data_navigator: A DataNavigator object for the dataset
            transformed_data: tables post tranformation {table_name:dataframe}
            model_type: Type of Copula to use for modeling
        """
        self.tables = {}
        self.models = {}
        self.child_locs = {}  # maps table->{child: col #}
        self.dn = data_navigator
        self.transformed_data = transformed_data
        self.model_type = model_type
        self.distribution = distribution

    def CPA(self, table):
        """ Runs CPA algorithm on a table
        Args:
            table (string): name of table
        """
        # Grab table
        data = self.dn.data
        # TODO: grab table from self.tables if it is not a leaf
        # ow grab from data
        children = self.dn.get_children(table)
        table_df, table_meta = data[table]
        # get primary key
        if 'primary_key' not in table_meta:
            # there are no references to the table
            return
        else:
            pk = table_meta['primary_key']
        # loop through rows of table
        num_rows = table_df.shape[0]
        sets = {}
        for i in range(num_rows):
            row = table_df.loc[i, :]
            # get specific value
            val = row[pk]
            sets[val] = []
        # get conditional data for val
        self._get_conditional_data(sets, pk, children, table)
        extended_table = pd.DataFrame([])
        # create extended table
        for i in range(num_rows):
            # TODO: change to be transformed table
            orig_row = table_df.loc[i, :]
            row = self.dn.transformed[table].loc[i, :]
            # get specific value
            val = orig_row[pk]
            # make sure val isn't none
            if pd.isnull(val):
                continue
            for extension in sets[val]:
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
        print(self.tables)
        self.CPA(table)

    def model_database(self):
        """ Uses RCPA and stores model for database """
        for table in self.dn.data:
            if self.dn.get_parents(table) == []:
                self.RCPA(table)
        for table in self.tables:
            table_model = self._get_model(self.model_type)()
            table_model.fit(self.tables[table])
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

    def load_model(self, filename):
        """ Loads model from filename
        Args:
            filename (string): path of file to load
        """
        pass

    def save_model(self, file_destination):
        """ Saves model to file destination
        Args:
            file_destination (string): path to store file
        """
        pass

    def _get_conditional_data(self, sets, pk, children, table):
        """ loops through children looking for rows that
        reference the value
        """
        # keep track of which columns belong to which child
        start = 0
        end = 0
        # find children that ref primary key
        for child in children:
            child_table, child_meta = self.dn.data[child]
            # child_table = self.tables[child]
            transformed_child_table = self.dn.transformed[child]
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
            for val in sets:
                # TODO: grab the tranformed table columns instead
                # extension = child_table[child_table[fk] == val]
                extension = transformed_child_table[child_table[fk] == val]
                if extension.empty:
                    continue
                # remove column of foreign key
                # extension = extension.drop(fk, axis=1)
                model = self._get_model(self.model_type)()
                model.fit(extension)
                flattened_extension = self.flatten_model(model, child)
                # keep track of child column indices
                end = max(end, start + len(flattened_extension))
                if table in self.child_locs:
                    self.child_locs[table][child] = (start, end)
                else:
                    self.child_locs[table] = {child: (start, end)}
                sets[val].append(flattened_extension)
            start = end

    def _get_model(self, model_name):
        """ Gets instance of model from name of model """
        return globals()[model_name]

    def get_model(self):
        """ Gets instance of model based on model type """
        return globals()[self.model_type]

    def get_distribution(self):
        """ Gets instance of model based on model type """
        return globals()[self.distribution]
