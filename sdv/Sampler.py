import numpy as np
import pandas as pd


class Sampler:
    """ Class to sample data from a model """
    def __init__(self, data_navigator):
        """ Instantiates a sampler """
        self.dn = data_navigator

    def sample_rows(self, table_name, num_rows):
        pass

    def sample_table(self, table_name):
        orig_table = self.dn.data[table_name][0]
        shape = orig_table.shape
        col_names = orig_table.columns.values
        sampled_table = pd.DataFrame(np.random.randint(0, 100, size=shape),
                                     columns=col_names)
        return sampled_table

    def sample_all(self):
        pass

    def sample_child_rows(self, table_name):
        pass
