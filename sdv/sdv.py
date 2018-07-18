# -*- coding: utf-8 -*-

"""Main module."""
from sdv.DataNavigator import *
from sdv.Sampler import Sampler
from sdv.Modeler import Modeler
import pickle

DATA_LOADERS = {
    'csv': CSVDataLoader
}


class SDV:
    """ class to do modeling and sampling all in one """
    def __init__(self, meta_file_name, data_loader_type='csv'):
        """ initializes sdv class """
        data_loader = DATA_LOADERS[data_loader_type](meta_file_name)
        self.dn = data_loader.load_data()
        # transform data
        self.dn.transform_data()
        self.modeler = Modeler(self.dn)
        self.sampler = None

    def model_database(self):
        """ has this sdv's modeler model the database """
        self.modeler.model_database()

    def sample_rows(self, table_name, num_rows):
        """ Calls Sampler's sample rows """
        if self.sampler is None:
            self.sampler = Sampler(self.dn, self.modeler)
        return self.sampler.sample_rows(table_name, num_rows)

    def sample_table(self, table_name):
        """ Calls Sampler's sample table """
        if self.sampler is None:
            self.sampler = Sampler(self.dn, self.modeler)
        return self.sampler.sample_table(table_name)

    def sample_all(self, num_rows=5):
        """ Calls Sampler's sample all """
        if self.sampler is None:
            self.sampler = Sampler(self.dn, self.modeler)
        return self.sampler.sample_all(num_rows)

    def load_modeler(self, filename):
        """ Loads model from filename
        Args:
            filename (string): path of file to load
        """
        with open(filename, 'rb') as input:
            self.modeler = pickle.load(input)
