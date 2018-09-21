# -*- coding: utf-8 -*-

"""Main module."""
import pickle

from sklearn.exceptions import NotFittedError

from sdv.data_navigator import CSVDataLoader
from sdv.modeler import Modeler
from sdv.sampler import Sampler


class SDV:
    """Class to do modeling and sampling all in one."""

    def __init__(self, meta_file_name, data_loader_type='csv'):
        """Initialize sdv class."""
        self.meta_file_name = meta_file_name
        self.sampler = None

    def fit(self):
        """Transform the data and model the database."""
        data_loader = CSVDataLoader(self.meta_file_name)
        self.dn = data_loader.load_data()
        # transform data
        self.dn.transform_data()
        self.modeler = Modeler(self.dn)
        self.modeler.model_database()
        self.sampler = Sampler(self.dn, self.modeler)

    def sample_rows(self, table_name, num_rows):
        """Wrapper for Sampler.sample_rows."""
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self.sampler.sample_rows(table_name, num_rows)

    def sample_table(self, table_name):
        """Wrapper for Sampler.sample_table."""
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self.sampler.sample_table(table_name)

    def sample_all(self, num_rows=5):
        """Wrapper for Sampler.sample_all."""
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')
        return self.sampler.sample_all(num_rows)

    def save(self, filename):
        """Save SDV instance to file destination.

        Args:
            file_destination (string): path to store file.
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
