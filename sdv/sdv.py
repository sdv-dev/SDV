# -*- coding: utf-8 -*-

"""Main module."""
import pickle

from copulas.univariate import GaussianUnivariate

from sdv.metadata import Metadata
from sdv.modeler import Modeler
from sdv.models.copulas import GaussianCopula
from sdv.sampler import Sampler

DEFAULT_MODEL = GaussianCopula
DEFAULT_MODEL_KWARGS = {
    'distribution': GaussianUnivariate
}


class NotFittedError(Exception):
    pass


class SDV:
    """Automated generative modeling and sampling tool.

    Allows the users to generate synthetic data after creating generative models for their data.

    Args:
        model (type):
            Class of the ``copula`` to use. Defaults to
            ``sdv.models.copulas.GaussianCopula``.
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

    def fit(self, metadata, tables=None, root_path=None):
        """Fit this SDV instance to the dataset data.

        Args:
            metadata (dict, str or Metadata):
                Metadata dict, path to the metadata JSON file or Metadata instance itself.
            tables (dict):
                Dictionary with the table names as key and ``pandas.DataFrame`` instances as
                values.  If ``None`` is given, the tables will be loaded from the paths
                indicated in ``metadata``. Defaults to ``None``.
            root_path (str or None):
                Path to the dataset directory. If ``None`` and metadata is
                a path, the metadata location is used. If ``None`` and
                metadata is a dict, the current working directory is used.
        """

        if isinstance(metadata, Metadata):
            self.metadata = metadata
        else:
            self.metadata = Metadata(metadata, root_path)

        self.metadata.validate(tables)

        self.modeler = Modeler(self.metadata, self.model, self.model_kwargs)
        self.modeler.model_database(tables)
        self.sampler = Sampler(self.metadata, self.modeler.models, self.model, self.model_kwargs)

    def sample(self, table_name, num_rows, sample_children=True, reset_primary_keys=False):
        """Sample ``num_rows`` rows from the indicated table.

        Args:
            table_name (str):
                Name of the table to sample from.
            num_rows (int):
                Amount of rows to sample.
            sample_children (bool):
                Whether or not to sample children tables. Defaults to ``True``.
            reset_primary_keys (bool):
                Wheter or not reset the primary key generators. Defaults to ``False``.

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
        """Sample the entire dataset.

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

    def save(self, path):
        """Save this SDV instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a SDV instance from a given path.

        Args:
            path (str):
                Path from which to load the SDV instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
