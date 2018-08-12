# -*- coding: utf-8 -*-

"""Main module."""
from sdv.DataNavigator import CSVDataLoader
from sdv.Sampler import Sampler
from sdv.Modeler import Modeler
from sklearn.exceptions import NotFittedError
import os.path as op
import pickle

DATA_LOADERS = {
    'csv': CSVDataLoader
}


class SDV:
    """ class to do modeling and sampling all in one """
    def __init__(self, meta_file_name, data_loader_type='csv'):
        """ initializes sdv class """
        self.data_loader = DATA_LOADERS[data_loader_type](meta_file_name)
        self.sampler = None

    def model_database(self):
        """ has this sdv's modeler model the database """
        self.modeler.model_database()

    def fit(self):
        """ transforms the data and models the database """
        self.dn = self.data_loader.load_data()
        # transform data
        self.dn.transform_data()
        self.modeler = Modeler(self.dn)
        self.sampler = Sampler(self.dn, self.modeler)

    def sample_rows(self, table_name, num_rows):
        """ Calls Sampler's sample rows """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')
        return self.sampler.sample_rows(table_name, num_rows)

    def sample_table(self, table_name):
        """ Calls Sampler's sample table """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')
        return self.sampler.sample_table(table_name)

    def sample_all(self, num_rows=5):
        """ Calls Sampler's sample all """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')
        return self.sampler.sample_all(num_rows)

    def save(self, filename):
        """ Saves SDV instance to file destination
        Args:
            file_destination (string): path to store file
        """
        suffix = '.pkl'
        ROOT_DIR = op.dirname(op.abspath(__file__))
        filename = op.join(ROOT_DIR, 'models', filename + suffix)
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
