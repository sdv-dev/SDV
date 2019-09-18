# -*- coding: utf-8 -*-

"""Main module."""
import pickle

from copulas import NotFittedError

from sdv.data_navigator import CSVDataLoader
from sdv.modeler import DEFAULT_MODEL, Modeler
from sdv.sampler import Sampler


class SDV:
    """Class to do modeling and sampling all in one.

    Args:
        meta_file_name (str): Path to the metadata file.
        data_loader_type (str):
        model (type): Class of model to use.
        distribution (type): Class of distribution to use. Will be deprecated shortly.
        model_kwargs (dict): Keyword arguments to pass to model.

    """

    def __init__(
        self, meta_file_name, data_loader_type='csv', model=DEFAULT_MODEL, distribution=None,
        model_kwargs=None
    ):
        self.meta_file_name = meta_file_name
        self.sampler = None
        self.model = model
        self.distribution = distribution
        self.model_kwargs = model_kwargs

    def _check_unsupported_dataset_structure(self):
        """Checks that no table has two parents."""
        tables = self.dn.tables.keys()
        amount_parents = [len(self.dn.get_parents(table)) <= 1 for table in tables]
        if not all(amount_parents):
            raise ValueError('Some tables have multiple parents, which is not supported yet.')

    def fit(self):
        """Transform the data and model the database.

        Raises:
            ValueError: If the provided dataset has an unsupported structure.
        """
        data_loader = CSVDataLoader(self.meta_file_name)
        self.dn = data_loader.load_data()

        self._check_unsupported_dataset_structure()

        self.dn.transform_data()
        self.modeler = Modeler(
            data_navigator=self.dn,
            model=self.model,
            distribution=self.distribution,
            model_kwargs=self.model_kwargs
        )
        self.modeler.model_database()
        self.sampler = Sampler(self.dn, self.modeler)

    def sample_rows(self, table_name, num_rows, sample_children=True, reset_primary_keys=False):
        """Sample `num_rows` rows from the given table.

        Args:
            table_name(str): Name of the table to sample from.
            num_rows(int): Amount of rows to sample.
            reset_primary_keys(bool): Wheter or not reset the pk generators.
        """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self.sampler.sample_rows(
            table_name, num_rows, sample_children=sample_children,
            reset_primary_keys=reset_primary_keys)

    def sample_table(self, table_name, reset_primary_keys=False):
        """Samples the given table to its original size.

        Args:
            table_name (str): Table to sample.
            reset_primary_keys(bool): Wheter or not reset the pk generators.
        """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self.sampler.sample_table(table_name, reset_primary_keys=reset_primary_keys)

    def sample_all(self, num_rows=5, reset_primary_keys=False):
        """Sample the whole dataset.

        Args:
            num_rows(int): Amount of rows to sample.
            reset_primary_keys(bool): Wheter or not reset the pk generators.
        """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self.sampler.sample_all(num_rows, reset_primary_keys=reset_primary_keys)

    def save(self, filename):
        """Save SDV instance to file destination.

        Args:
            file_destination(str): Path to store file.
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """Load a SDV instance from the given path.

        Args:
            filename(str): Path to load model.

        """
        with open(filename, 'rb') as f:
            instance = pickle.load(f)

        return instance
