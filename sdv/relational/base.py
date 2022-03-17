"""Hierarchical Modeling Algorithms."""

import itertools
import logging
import pickle

import numpy as np
import pandas as pd

from sdv.errors import NotFittedError
from sdv.metadata import Metadata, utils
from sdv.utils import get_package_versions, throw_version_mismatch_warning

LOGGER = logging.getLogger(__name__)


class BaseRelationalModel:
    """Base class for all the relational models.

    The ``BaseRelationalModel`` class defines the common API that all the
    relational models need to implement, as well as common functionality.

    Args:
        metadata (dict, str or Metadata):
            Metadata dict, path to the metadata JSON file or Metadata instance itself.
        root_path (str or None):
            Path to the dataset directory. If ``None`` and metadata is
            a path, the metadata location is used. If ``None`` and
            metadata is a dict, the current working directory is used.
    """

    metadata = None

    def __init__(self, metadata, root_path=None):
        if isinstance(metadata, Metadata):
            self.metadata = metadata
        else:
            self.metadata = Metadata(metadata, root_path)

        self._primary_key_generators = dict()
        self._remaining_primary_keys = dict()

    def _fit(self, tables=None):
        """Fit this relational model instance to the dataset data.

        Args:
            tables (dict):
                Dictionary with the table names as key and ``pandas.DataFrame`` instances as
                values.  If ``None`` is given, the tables will be loaded from the paths
                indicated in ``metadata``. Defaults to ``None``.
        """
        raise NotImplementedError()

    def fit(self, tables=None):
        """Fit this relational model instance to the dataset data.

        Args:
            tables (dict):
                Dictionary with the table names as key and ``pandas.DataFrame`` instances as
                values.  If ``None`` is given, the tables will be loaded from the paths
                indicated in ``metadata``. Defaults to ``None``.
        """
        self._fit(tables)
        self.fitted = True

    def _reset_primary_keys_generators(self):
        """Reset the primary key generators."""
        self._primary_key_generators = dict()
        self._remaining_primary_keys = dict()

    def _get_primary_keys(self, table_name, num_rows):
        """Return the primary key and amount of values for the requested table.

        Args:
            table_name (str):
                Name of the table to get the primary keys from.
            num_rows (str):
                Number of ``primary_keys`` to generate.

        Returns:
            tuple (str, pandas.Series):
                primary key name and primary key values. If the table has no primary
                key, ``(None, None)`` is returned.

        Raises:
            ValueError:
                If the ``metadata`` contains invalid types or subtypes, or if
                there are not enough primary keys left on any of the generators.
            NotImplementedError:
                If the primary key subtype is a ``datetime``.
        """
        primary_key = self.metadata.get_primary_key(table_name)

        field = self.metadata.get_fields(table_name)[primary_key]

        generator = self._primary_key_generators.get(table_name)

        if generator is None:
            if field['type'] != 'id':
                raise ValueError('Only columns with type `id` can be primary keys')

            subtype = field.get('subtype', 'integer')
            if subtype == 'integer':
                generator = itertools.count()
                remaining = np.inf
            elif subtype == 'string':
                regex = field.get('regex', r'^[a-zA-Z]+$')
                generator, remaining = utils.strings_from_regex(regex)
            elif subtype == 'datetime':
                raise NotImplementedError('Datetime ids are not yet supported')
            else:
                raise ValueError('Only `integer` or `string` id columns are supported.')

            self._primary_key_generators[table_name] = generator
            self._remaining_primary_keys[table_name] = remaining

        else:
            remaining = self._remaining_primary_keys[table_name]

        if remaining < num_rows:
            raise ValueError(
                'Not enough unique values for primary key of table {}'
                ' to generate {} samples.'.format(table_name, num_rows)
            )

        self._remaining_primary_keys[table_name] -= num_rows
        primary_key_values = pd.Series([x for i, x in zip(range(num_rows), generator)])

        return primary_key_values

    def _sample(self, table_name=None, num_rows=None, sample_children=True):
        """Generate synthetic data for one table or the entire dataset."""
        raise NotImplementedError()

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
                A ``NotFittedError`` is raised when the model has not been fitted yet.
        """
        if not self.fitted:
            raise NotFittedError('SDV instance has not been fitted')

        if reset_primary_keys:
            self._reset_primary_keys_generators()

        return self._sample(table_name, num_rows, sample_children)

    def save(self, path):
        """Save this instance to the given path using pickle.

        Args:
            path (str):
                Path where the instance will be serialized.
        """
        self._package_versions = get_package_versions(getattr(self, '_model', None))

        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a model from a given path.

        Args:
            path (str):
                Path from which to load the instance.
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
            throw_version_mismatch_warning(getattr(model, '_package_versions', None))

            return model
