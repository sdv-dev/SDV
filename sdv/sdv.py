# -*- coding: utf-8 -*-

"""Main module."""
import pickle

from copulas.multivariate import GaussianMultivariate

from sdv.metadata import Metadata
from sdv.modeler import Modeler
from sdv.sampler import Sampler

DEFAULT_MODEL = GaussianMultivariate
DEFAULT_MODEL_KWARGS = {
    'distribution': 'copulas.univariate.gaussian.GaussianUnivariate'
}


class NotFittedError(Exception):
    pass


class SDV:
    """Automated generative modeling and sampling tool.

    Allows the users to generate synthetic data after creating generative models for their data.

    Args:
        model (type):
            Class of a ``copula`` to use. Defaults to
            ``copulas.multivariate.GaussianMultivariate``.
        model_kwargs (dict):
            Keyword arguments to pass to the model. Defaults to ``None``.
    """

    sampler = None

    def __init__(self, model=DEFAULT_MODEL, model_kwargs=None):
        self.model = model
        if model_kwargs is None:
            self.model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        else:
            self.model_kwargs = model_kwargs

    def _validate_dataset_structure(self):
        """Check if any table has two parents."""
        for table in self.metadata.get_table_names():
            if len(self.metadata.get_parents(table)) > 1:
                raise ValueError('Some tables have multiple parents, which is not supported yet.')

    def fit(self, metadata, tables=None, root_path=None):
        """Fit the ``model`` with to be able to sample data.

        This method creates a ``sdv.Metadata`` object and validates the data structure. Once
        this validation has passed, an instace of ``sdv.Modeler`` which will model the dataset.
        Then a ``sdv.Sampler`` object is being created in order to be used for sampling.

        Args:
            metadata (dict or str):
                Metadata dict or path to the metadata JSON file.
            tables (dict):
                Dictionary with the table name as key and ``pandas.DataFrame`` as value.
                If ``None`` tables will be loaded from the paths indicated in ``metadata``.
                Defaults to ``None``.
            root_path (str or None):
                Path to the dataset directory. If ``None`` and metadata is
                a path, the metadata location is used. If ``None`` and
                metadata is a dict, the current working directory is used.
        """

        self.metadata = Metadata(metadata, root_path)
        self._validate_dataset_structure()

        self.modeler = Modeler(self.metadata, self.model, self.model_kwargs)
        self.modeler.model_database(tables)
        self.sampler = Sampler(self.metadata, self.modeler.models)

    def sample(self, table_name, num_rows, sample_children=True, reset_primary_keys=False):
        """Sample ``num_rows`` rows from the given table.

        Args:
            table_name (str):
                Name of the table to sample from.
            num_rows (int):
                Amount of rows to sample.
            sample_children (bool):
                Whether or not to sample children tables. Defaults to ``True``.
            reset_primary_keys (bool):
                Wheter or not reset the pk generators. Defaults to ``True``.

        Returns:
            pandas.DataFrame:
                Sampled data with the number of rows specified in ``num_rows``.

        Raises:
            NotFittedError:
                A ``NotFittedError`` is raised when the ``SDV`` instance has not been fitted yet.
        """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self.sampler.sample(
            table_name,
            num_rows,
            sample_children=sample_children,
            reset_primary_keys=reset_primary_keys
        )

    def sample_all(self, num_rows=5, reset_primary_keys=False):
        """Sample the entire database.

        Args:
            num_rows (int):
                Amount of rows to sample. Defaults to ``5``.
            reset_primary_keys (bool):
                Wheter or not reset the primary key generators. Defaults to ``False``.

        Returns:
            dict:
                Tables sampled.

        Raises:
            NotFittedError:
                A ``NotFittedError`` is raised when the ``SDV`` instance has not been fitted yet.
        """
        if self.sampler is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self.sampler.sample_all(num_rows, reset_primary_keys=reset_primary_keys)

    def save(self, filename):
        """Save SDV instance to a file destination.

        Args:
            filename (str):
                Path where to store the SDV instance.
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """Load a SDV instance from a given path.

        Args:
            filename (str):
                Path to the saved SDV instance to be loaded.
        """
        with open(filename, 'rb') as f:
            instance = pickle.load(f)

        return instance
