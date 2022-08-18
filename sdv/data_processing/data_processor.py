"""Single table data processing."""

import copy
import json

import numpy as np
import pandas as pd
import rdt

from sdv.constraints import Constraint
from ..metadata.single_table import SingleTableMetadata
from sdv.metadata.utils import strings_from_regex


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

    @classmethod
    def _make_ids(cls, field_metadata, length):
        field_subtype = field_metadata.get('subtype', 'integer')
        if field_subtype == 'string':
            regex = field_metadata.get('regex', '[a-zA-Z]+')
            generator, max_size = strings_from_regex(regex)
            if max_size < length:
                raise ValueError((
                    'Unable to generate {} unique values for regex {}, the '
                    'maximum number of unique values is {}.'
                ).format(length, regex, max_size))
            values = [next(generator) for _ in range(length)]

            return pd.Series(list(values)[:length])
        else:
            return pd.Series(np.arange(length))

    def make_ids_unique(self, data):
        """Repopulate any id fields in provided data to guarantee uniqueness.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Table where all id fields are unique.
        """
        data = data.copy()
        for name, field_metadata in self.metadata._columns.items():
            if field_metadata['sdtype'] == 'id' and not data[name].is_unique:
                ids = self._make_ids(field_metadata, len(data))
                ids.index = data.index.copy()
                data[name] = ids

        return data

    def to_dict(self):
        """Get a dict representation of this metadata.

        Returns:
            dict:
                dict representation of this metadata.
        """
        # TODO: why are _transformers_by_sdtype not passed here?
        return {
            'metadata': copy.deepcopy(SingleTableMetadata.to_dict()),
            'model_kwargs': copy.deepcopy(self._model_kwargs),
        }

    def to_json(self, path):
        """Dump this metadata into a JSON file.

        Args:
            path (str):
                Path of the JSON file where this metadata will be stored.
        """
        with open(path, 'w') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

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
        return instance

    @classmethod
    def from_json(cls, path):
        """Load a Table from a JSON.

        Args:
            path (str):
                Path of the JSON file to load
        """
        with open(path, 'r') as in_file:
            return cls.from_dict(json.load(in_file))
