import numpy as np
import pandas as pd


class Sampler:
    """ Class to sample data from a model """

    def __init__(self, data_navigator):
        """ Instantiates a sampler """
        self.dn = data_navigator
        self.been_sampled = set()  # table_name -> if already sampled

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
        for parent in parents:
            if parent in self.been_sampled:
                parent_sampled = True
                break
        if parents == []:
            orig_table = self.dn.data[table_name][0]
            num_cols = orig_table.shape[1]
            shape = (num_rows, num_cols)
            col_names = orig_table.columns.values
            sampled_table = pd.DataFrame(np.random.randint(0, 100, size=shape),
                                         columns=col_names)
            self.been_sampled.add(table_name)
        elif parent_sampled:
            # Here we would get params necessary to get model
            orig_table = self.dn.data[table_name][0]
            num_cols = orig_table.shape[1]
            shape = (num_rows, num_cols)
            col_names = orig_table.columns.values
            sampled_table = pd.DataFrame(np.random.randint(0, 100, size=shape),
                                         columns=col_names)
            self.been_sampled.add(table_name)
        else:
            raise Exception('Parents must be synthesized first')
        print(self.been_sampled)
        return sampled_table

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
