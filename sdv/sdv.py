# -*- coding: utf-8 -*-

"""Main SDV module."""

import pickle
import warnings

from sdv.errors import NotFittedError
from sdv.relational.hma import HMA1
from sdv.tabular.copulas import GaussianCopula
from sdv.utils import get_package_versions, throw_version_mismatch_warning


class SDV:
    """Automated generative modeling and sampling tool.

    Allows the users to generate synthetic data after creating generative models for their data.

    Args:
        model (type):
            Class of the model to use. Defaults to ``sdv.relational.HMA1``.
        model_kwargs (dict):
            Keyword arguments to pass to the model. If no ``model`` is given,
            this defaults to using a ``GaussianCopula`` with ``gaussian`` distribution
            and ``categorical_fuzzy`` categorical transformer.
    """

    _model_instance = None
    DEFAULT_MODEL = HMA1
    DEFAULT_MODEL_KWARGS = {
        'model': GaussianCopula,
        'model_kwargs': {
            'default_distribution': 'gaussian',
            'categorical_transformer': 'categorical_fuzzy',
        }
    }

    def __init__(self, model=None, model_kwargs=None):
        if model is None:
            model = model or self.DEFAULT_MODEL
            if model_kwargs is None:
                model_kwargs = self.DEFAULT_MODEL_KWARGS

        self._model = model
        self._model_kwargs = (model_kwargs or dict()).copy()

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
        self._model_instance = self._model(metadata, root_path, **self._model_kwargs)
        self._model_instance.fit(tables)

    def sample(self, table_name=None, num_rows=None,
               sample_children=True, reset_primary_keys=False):
        """Generate synthetic data for one table or the entire dataset.

        If a ``table_name`` is given and ``sample_children`` is ``False``, a
        ``pandas.DataFrame`` with the values from the indicated table is returned.
        Otherwise, if ``sample_children`` is ``True``, a dictionary containing both
        the table and all its descendant tables is returned.

        If no ``table_name`` is given, the entire dataset is sampled and returned
        in a dictionary.

        If ``num_rows`` is given, the root tables of the dataset will contain the
        indicated number of rows. Otherwise, the number of rows will be the same
        as in the original dataset. Number of rows in the child tables cannot be
        controlled and always will depend on the values from the sampled parent
        tables.

        If ``reset_primary_keys`` is ``True``, the primary key generators will be
        reset.

        Args:
            table_name (str):
                Name of the table to sample from. If not passed, sample the entire
                dataset.
            num_rows (int):
                Amount of rows to sample. If ``None``, sample the same number of rows
                as there were in the original table.
            sample_children (bool):
                Whether or not sample child tables. Used only if ``table_name`` is
                given. Defaults to ``True``.
            reset_primary_keys (bool):
                Whether or not reset the primary keys generators. Defaults to ``False``.

        Returns:
            dict or pandas.DataFrame:
                - Returns a ``dict`` when ``sample_children`` is ``True`` with the sampled table
                  and child tables.
                - Returns a ``pandas.DataFrame`` when ``sample_children`` is ``False``.

        Raises:
            NotFittedError:
                A ``NotFittedError`` is raised when the ``SDV`` instance has not been fitted yet.
        """
        if self._model_instance is None:
            raise NotFittedError('SDV instance has not been fitted')

        return self._model_instance.sample(
            table_name,
            num_rows,
            sample_children=sample_children,
            reset_primary_keys=reset_primary_keys
        )

    def sample_all(self, num_rows=None, reset_primary_keys=False):
        """Sample the entire dataset.

        WARNING: This method is deprecated and will be removed in future relaeses. Please
        use the ``sample`` method instead.

        Args:
            num_rows (int):
                Number of rows to be sampled on the first parent tables. If ``None``,
                sample the same number of rows as in the original tables.
            reset_primary_keys (bool):
                Wheter or not reset the primary key generators. Defaults to ``False``.

        Returns:
            dict:
                Tables sampled.

        Raises:
            NotFittedError:
                A ``NotFittedError`` is raised when the ``SDV`` instance has not been fitted yet.
        """
        warnings.warn('`sample_all` is deprecated and will be removed soon. Please use `sample`',
                      DeprecationWarning)
        return self.sample(num_rows=num_rows, reset_primary_keys=reset_primary_keys)

    def save(self, path):
        """Save this SDV instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        self._package_versions = get_package_versions(getattr(self, '_model', None))

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
            model = pickle.load(f)
            throw_version_mismatch_warning(getattr(model, '_package_versions', None))

            return model
