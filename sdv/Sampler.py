import numpy as np
import pandas as pd
import random


class Sampler:
    """ Class to sample data from a model """
    def __init__(self, data_navigator, modeler):
        """ Instantiates a sampler """
        self.dn = data_navigator
        self.modeler = modeler
        self.been_sampled = set()  # table_name -> if already sampled
        self.sampled = {}  # table_name -> [(primary_key, generated_row)]

    def sample_rows(self, table_name, num_rows):
        """ Sample specified number of rows for specified table
        Args:
            table_name (str): name of table to synthesize
            num_rows (int): number of rows to synthesize
        Returns:
            synthesized rows (dataframe)
        """
        parents = self.dn.get_parents(table_name)
        parent_sampled = False
        primary_key = self.dn.data[table_name][1]['primary_key']
        for parent in parents:
            if parent in self.sampled:
                parent_sampled = True
                break
        if parents == []:
            model = self.modeler.models[table_name]
            synthesized_row = model.sample()
            sample_info = (primary_key, synthesized_row)
            if table_name in self.sampled:
                self.sampled[table_name].append(sample_info)
            else:
                self.sampled[table_name] = [sample_info]
            # filter out parameters
            labels = list(self.dn.data[table_name][0])
            return synthesized_row[labels]
        elif parent_sampled:
            # grab random parent row
            random_parent = random.sample(parents, 1)[0]
            parent_rows = self.sampled[random_parent]
            fk, parent_row = random.sample(parent_rows, 1)[0]
            # get parameters from parent to make model
            model = self._make_model_from_params(parent_row,
                                                 table_name,
                                                 random_parent)
            # sample from that model
            synthesized_row = model.sample()
            # add foreign key value to row
            fk_val = parent_row.loc[0, fk]
            synthesized_row[fk] = fk_val
            sample_info = (primary_key, synthesized_row)
            if table_name in self.sampled:
                self.sampled[table_name].append(sample_info)
            else:
                self.sampled[table_name] = [sample_info]
            # filter out parameters
            labels = list(self.dn.data[table_name][0])
            return synthesized_row[labels]
        else:
            raise Exception('Parents must be synthesized first')

    def sample_table(self, table_name):
        """ Sample a table equal to the size of the original
        Args:
            table_name (str): name of table to synthesize
        Returns:
            Synthesized table (dataframe)
        """
        orig_table = self.dn.data[table_name][0]
        num_rows = orig_table.shape[0]
        return self.sample_rows(table_name, num_rows)

    def sample_all(self, num_rows=5):
        """ Samples the entire database """
        data = self.dn.data
        sampled_data = {}
        for table in data:
            if self.dn.get_parents(table) == []:
                for i in range(num_rows):
                    row = self.sample_rows(table, 1)
                    if table in sampled_data:
                        length = sampled_data[table].shape[0]
                        sampled_data[table].loc[length:, :] = row
                    else:
                        sampled_data[table] = row
                    self._sample_child_rows(table, row, sampled_data)
        return sampled_data

    def _sample_child_rows(self, parent_name, parent_row, sampled_data,
                           num_rows=5):
        """ Uses parameters from parent row to synthesize
        child rows
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
                length = sampled_data[child].shape[0]
                sampled_data[child].loc[length:, :] = rows.iloc[0:1, :]
            else:
                sampled_data[child] = rows
            self._sample_child_rows(child, rows.iloc[0:1, :], sampled_data)

    def _make_model_from_params(self, parent_row, table_name, parent_name):
        """ Takes the params from a generated parent row
        and creates a model from it
        Args:
            parent_row (dataframe): a generated parent row
            table_name (string): name of table to make model for
            parent_name (string): name of parent table
        """
        # get parameters
        child_range = self.modeler.child_locs[parent_name][table_name]
        params = parent_row.iloc[:, child_range[0]:child_range[1]]
        totalcols = params.shape[1]
        # build model
        model = self.modeler.get_model()()
        num_cols = self.modeler.tables[table_name].shape[1]-1
        cov_size = num_cols**2
        # get labels for dataframe
        labels = list(self.modeler.tables[table_name])
        parent_meta = self.dn.data[parent_name][1]
        fk = parent_meta['primary_key']
        labels.remove(fk)
        # get covariance matrix
        cov = params.iloc[:, 0:num_cols**2]
        cov_matrix = cov.as_matrix()
        cov_matrix = cov_matrix.reshape((num_cols, num_cols))
        model.cov_matrix = cov_matrix
        distribs = {}
        # get distributions of columns and means
        means = list(params.iloc[:,
                                 cov_size:cov_size+num_cols].values.flatten())
        model.means = means
        label_index = 0
        for i in range(num_cols**2+num_cols, totalcols, 2):
            distrib = self.modeler.get_distribution()()
            std = params.iloc[:, i]
            mean = params.iloc[:, i+1]
            distrib.mean = mean
            distrib.std = std
            distribs[labels[label_index]] = distrib
            label_index += 1
        model.distribs = distribs
        # create fake data
        # TODO: Change copulas to not need data
        data = pd.DataFrame(np.random.randint(0, 10, size=(2, len(labels))),
                            columns=labels)
        model.data = data
        return model
