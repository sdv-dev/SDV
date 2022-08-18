"""Single table data processing."""

import copy
import json

import numpy as np
import pandas as pd
import rdt

from sdv.constraints import Constraint
from sdv.metadata.utils import cast_to_iterable


class DataProcessor:
    """Single table data processor.

    This class handles all pre and post processing that is done to a single table to get it ready
    for modeling and finalize sampling. These processes include formatting, transformations,
    anonymization and constraint handling.

    Args:
        metadata (metadata.SingleTableMetadata):
            The single table metadata instance that will be used to apply constraints and
            transformations to the data.
        learn_rounding_scheme (bool):
            Define rounding scheme for FloatFormatter. If True, the data returned by
            reverse_transform will be rounded to that place. Defaults to True.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by reverse_transform of the numerical
            transformer, FloatFormatter, to the min and max values seen during fit.
            Defaults to True.
        model_kwargs (dict):
            Dictionary specifying the kwargs that need to be used in each tabular
            model when working on this table. This dictionary contains as keys the name of the
            TabularModel class and as values a dictionary containing the keyword arguments to use.
            This argument exists mostly to ensure that the models are fitted using the same
            arguments when the same DataProcessor is used to fit different model instances on
            different slices of the same table.
    """

    _DEFAULT_TRANSFORMERS_BY_SDTYPE = {
        'numerical': rdt.transformers.FloatFormatter(
            learn_rounding_scheme=True,
            enforce_min_max_values=True,
            missing_value_replacement='mean',
            model_missing_values=True,
        ),
        'categorical': rdt.transformers.LabelEncoder(add_noise=True),
        'boolean': rdt.transformers.LabelEncoder(add_noise=True),
        'datetime': rdt.transformers.UnixTimestampEncoder(
            missing_value_replacement='mean',
            model_missing_values=True,
        )
    }

    def _load_constraints(self):
        constraints = self.metadata._constraints or []
        loaded_constraints = [Constraint.from_dict(constraint) for constraint in constraints]
        return loaded_constraints

    def _update_numerical_transformer(self, learn_rounding_scheme, enforce_min_max_values):
        custom_float_formatter = rdt.transformers.FloatFormatter(
            missing_value_replacement='mean',
            model_missing_values=True,
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values
        )
        self._transformers_by_sdtype.update({'numerical': custom_float_formatter})

    def __init__(self, metadata, learn_rounding_scheme=True, enforce_min_max_values=True,
                 model_kwargs=None):
        self.metadata = metadata
        self._model_kwargs = model_kwargs or {}
        self._constraints = self._load_constraints()
        self._constraints_to_reverse = []
        self._transformers_by_sdtype = self._DEFAULT_TRANSFORMERS_BY_SDTYPE.copy()
        self._update_numerical_transformer(learn_rounding_scheme, enforce_min_max_values)

    def filter_valid(self, data):
        """Filter the data using the constraints and return only the valid rows.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Table containing only the valid rows.
        """
        for constraint in self._constraints:
            data = constraint.filter_valid(data)

        return data

    def _get_keys(self):
        """Return a set of all the keys."""
        keys = {self.metadata._sequence_index}
        keys.update(set(cast_to_iterable(self.metadata._primary_key)))
        keys.update(set(cast_to_iterable(self.metadata._sequence_key)))
        if self.metadata._alternate_keys:
            for key in self.metadata._alternate_keys:
                keys.update({key} if isinstance(key, str) else set(key))

        keys.discard(None)
        return keys

    def make_ids_unique(self, data):
        """Repopulate numerical id fields in provided data to guarantee uniqueness.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Table where all id fields are unique.
        """
        data = data.copy()
        keys = self._get_keys()
        for key in keys:
            if not data[key].is_unique:
                field_metadata = self.metadata._columns[key]
                if field_metadata['sdtype'] == 'numerical':
                    new_keys = pd.Series(np.arange(len(data)))
                    new_keys.index = data.index.copy()
                    data[key] = new_keys

        return data

    def to_dict(self):
        """Get a dict representation of this DataProcessor.

        Returns:
            dict:
                Dict representation of this DataProcessor.
        """
        return {
            'metadata': copy.deepcopy(self.metadata.to_dict()),
            'transformers_by_sdtype': copy.deepcopy(self._transformers_by_sdtype),
            'constraints_to_reverse': copy.deepcopy(self._constraints_to_reverse),
            'model_kwargs': copy.deepcopy(self._model_kwargs)
        }

    @classmethod
    def from_dict(cls, metadata_dict):
        """Load a DataProcessor from a metadata dict.

        Args:
            metadata_dict (dict):
                Dict metadata to load.
        """
        instance = cls(
            metadata=metadata_dict['metadata'],
            learn_rounding_scheme=metadata_dict.get('learn_rounding_scheme', True),
            enforce_min_max_values=metadata_dict.get('enforce_min_max_values', True),
            model_kwargs=metadata_dict.get('model_kwargs')
        )
        instance._constraints_to_reverse = metadata_dict.get('constraints_to_reverse', [])
        instance._transformers_by_sdtype = metadata_dict.get(
            'transformers_by_sdtype', instance._transformers_by_sdtype
        )

        return instance

    def to_json(self, path):
        """Dump this DataProcessor into a JSON file.

        Args:
            path (str):
                Path of the JSON file where this metadata will be stored.
        """
        with open(path, 'w') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

    @classmethod
    def from_json(cls, path):
        """Load a DataProcessor from a JSON.

        Args:
            path (str):
                Path of the JSON file to load
        """
        with open(path, 'r') as in_file:
            return cls.from_dict(json.load(in_file))
