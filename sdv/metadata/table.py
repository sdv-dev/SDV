import copy
import json
import logging

import numpy as np
import pandas as pd
import rdt
from faker import Faker

from sdv.metadata.errors import MetadataError

LOGGER = logging.getLogger(__name__)


class Table:
    """Table Metadata.

    The Metadata class provides a unified layer of abstraction over the metadata
    of a single Table, which includes both the necessary details to load the data
    from the filesystem and to know how to parse and transform it to numerical data.
    """

    _metadata = None
    _hyper_transformer = None
    _anonymization_mappings = None

    _FIELD_TEMPLATES = {
        'i': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'f': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'O': {
            'type': 'categorical',
        },
        'b': {
            'type': 'boolean',
        },
        'M': {
            'type': 'datetime',
        }
    }
    _DTYPES = {
        ('categorical', None): 'object',
        ('boolean', None): 'bool',
        ('numerical', None): 'float',
        ('numerical', 'float'): 'float',
        ('numerical', 'integer'): 'int',
        ('datetime', None): 'datetime64',
        ('id', None): 'int',
        ('id', 'integer'): 'int',
        ('id', 'string'): 'str'
    }

    def _get_faker(self, category):
        """Return the faker object to anonymize data.

        Args:
            category (str or tuple):
                Fake category to use. If a tuple is passed, the first element is
                the category and the rest are additional arguments for the Faker.

        Returns:
            function:
                Faker function to generate new fake data instances.

        Raises:
            ValueError:
                A ``ValueError`` is raised if the faker category we want don't exist.
        """
        if isinstance(category, (tuple, list)):
            category, *args = category
        else:
            args = tuple()

        try:
            faker_method = getattr(Faker(), category)

            if not args:
                return faker_method

            def faker():
                return faker_method(*args)

            return faker

        except AttributeError:
            raise ValueError('Category "{}" couldn\'t be found on faker'.format(category))

    def __init__(self, metadata=None, field_names=None, primary_key=None,
                 field_types=None, anonymize_fields=None, constraints=None,
                 transformer_templates=None):
        # TODO: Validate that the given metadata is a valid dict
        self._metadata = metadata or dict()
        self._hyper_transformer = None

        self._field_names = field_names
        self._primary_key = primary_key
        self._field_types = field_types or {}
        self._anonymize_fields = anonymize_fields
        self._constraints = constraints

        self._transformer_templates = transformer_templates or {}

        self._fakers = {
            name: self._get_faker(category)
            for name, category in (anonymize_fields or {}).items()
        }

    def _get_field_dtype(self, field_name, field_metadata):
        field_type = field_metadata['type']
        field_subtype = field_metadata.get('subtype')
        dtype = self._DTYPES.get((field_type, field_subtype))
        if not dtype:
            raise MetadataError(
                'Invalid type and subtype combination for field {}: ({}, {})'.format(
                    field_name, field_type, field_subtype)
            )

        return dtype

    def get_fields(self):
        """Get fields metadata.

        Returns:
            dict:
                Dictionary of fields metadata for this table.
        """
        return copy.deepcopy(self._metadata['fields'])

    def get_dtypes(self, ids=False):
        """Get a ``dict`` with the ``dtypes`` for each field of the table.

        Args:
            ids (bool):
                Whether or not include the id fields. Defaults to ``False``.

        Returns:
            dict:
                Dictionary that contains the field names and data types.

        Raises:
            ValueError:
                If a field has an invalid type or subtype.
        """
        dtypes = dict()
        for name, field_meta in self._metadata['fields'].items():
            field_type = field_meta['type']

            if ids or (field_type != 'id'):
                dtypes[name] = self._get_field_dtype(name, field_meta)

        return dtypes

    def _build_fields_metadata(self, data):
        """Build all the fields metadata.

        Args:
            data (pandas.DataFrame):
                Data to be analyzed.

        Returns:
            dict:
                Dict of valid fields.

        Raises:
            ValueError:
                If a column from the data analyzed is an unsupported data type
        """
        field_names = self._field_names or data.columns

        fields_metadata = dict()
        for field_name in field_names:
            if field_name not in data:
                raise ValueError('Field {} not found in given data'.format(field_name))

            field_meta = self._field_types.get(field_name)
            if field_meta:
                # Validate the given meta
                self._get_field_dtype(field_name, field_meta)
            else:
                dtype = data[field_name].dtype
                field_template = self._FIELD_TEMPLATES.get(dtype.kind)
                if field_template is None:
                    raise ValueError('Unsupported dtype {} in column {}'.format(dtype, field_name))

                field_meta = copy.deepcopy(field_template)

            fields_metadata[field_name] = field_meta

        return fields_metadata

    def _get_transformers(self, dtypes):
        """Create the transformer instances needed to process the given dtypes.

        Args:
            dtypes (dict):
                mapping of field names and dtypes.

        Returns:
            dict:
                mapping of field names and transformer instances.
        """
        transformer_templates = {
            'i': rdt.transformers.NumericalTransformer(dtype=int),
            'f': rdt.transformers.NumericalTransformer(dtype=float),
            'O': rdt.transformers.CategoricalTransformer,
            'b': rdt.transformers.BooleanTransformer,
            'M': rdt.transformers.DatetimeTransformer,
        }
        transformer_templates.update(self._transformer_templates)

        transformers = dict()
        for name, dtype in dtypes.items():
            transformer_template = transformer_templates[np.dtype(dtype).kind]
            if isinstance(transformer_template, type):
                transformer = transformer_template()
            else:
                transformer = copy.deepcopy(transformer_template)

            LOGGER.info('Loading transformer %s for field %s',
                        transformer.__class__.__name__, name)
            transformers[name] = transformer

        return transformers

    def _fit_hyper_transformer(self, data):
        """Create and return a new ``rdt.HyperTransformer`` instance.

        First get the ``dtypes`` and then use them to build a transformer dictionary
        to be used by the ``HyperTransformer``.

        Returns:
            rdt.HyperTransformer
        """
        dtypes = self.get_dtypes(ids=False)
        transformers_dict = self._get_transformers(dtypes)
        self._hyper_transformer = rdt.HyperTransformer(transformers=transformers_dict)
        self._hyper_transformer.fit(data[list(dtypes.keys())])

    @staticmethod
    def _get_key_subtype(field_meta):
        """Get the appropriate key subtype."""
        field_type = field_meta['type']

        if field_type == 'categorical':
            field_subtype = 'string'

        elif field_type in ('numerical', 'id'):
            field_subtype = field_meta['subtype']
            if field_subtype not in ('integer', 'string'):
                raise ValueError(
                    'Invalid field "subtype" for key field: "{}"'.format(field_subtype)
                )

        else:
            raise ValueError(
                'Invalid field "type" for key field: "{}"'.format(field_type)
            )

        return field_subtype

    def set_primary_key(self, field_name):
        """Set the primary key of this table.

        The field must exist and either be an integer or categorical field.

        Args:
            field_name (str):
                Name of the field to be used as the new primary key.

        Raises:
            ValueError:
                If the table or the field do not exist or if the field has an
                invalid type or subtype.
        """
        if field_name is not None:
            if field_name not in self._metadata['fields']:
                raise ValueError('Field "{}" does not exist in this table'.format(field_name))

            field_metadata = self._metadata['fields'][field_name]
            field_subtype = self._get_key_subtype(field_metadata)

            field_metadata.update({
                'type': 'id',
                'subtype': field_subtype
            })

        self._primary_key = field_name

    def _make_anonymization_mappings(self, data):
        mappings = {}
        for name, faker in self._fakers.items():
            uniques = data[name].unique()
            fake_values = [faker() for x in range(len(uniques))]
            mappings[name] = dict(zip(uniques, fake_values))

        self._anonymization_mappings = mappings

    def _anonymize(self, data):
        if self._anonymization_mappings:
            data = data.copy()
            for name, mapping in self._anonymization_mappings.items():
                data[name] = data[name].map(mapping)

        return data

    def fit(self, data):
        """Fit this metadata to the given data.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
        """
        self._field_names = self._field_names or list(data.columns)
        self._metadata['fields'] = self._build_fields_metadata(data)
        self.set_primary_key(self._primary_key)

        if self._fakers:
            self._make_anonymization_mappings(data)
            data = self._anonymize(data)

        # TODO: Treat/Learn constraints
        self._fit_hyper_transformer(data)

    def transform(self, data):
        """Transform the given data.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        data = self._anonymize(data[self._field_names])
        return self._hyper_transformer.transform(data)

    def reverse_transform(self, data):
        """Reverse the transformed data to the original format.

        Args:
            data (pandas.DataFrame):
                Data to be reverse transformed.

        Returns:
            pandas.DataFrame
        """
        reversed_data = self._hyper_transformer.reverse_transform(data)

        fields = self._metadata['fields']
        for name, dtype in self.get_dtypes(ids=True).items():
            field_type = fields[name]['type']
            if field_type == 'id':
                field_data = pd.Series(np.arange(len(reversed_data)))
            else:
                field_data = reversed_data[name]

            reversed_data[name] = field_data.dropna().astype(dtype)

        return reversed_data[self._field_names]

    # ###################### #
    # Metadata Serialization #
    # ###################### #

    def to_dict(self):
        """Get a dict representation of this metadata.

        Returns:
            dict:
                dict representation of this metadata.
        """
        return copy.deepcopy(self._metadata)

    def to_json(self, path):
        """Dump this metadata into a JSON file.

        Args:
            path (str):
                Path of the JSON file where this metadata will be stored.
        """
        with open(path, 'w') as out_file:
            json.dump(self._metadata, out_file, indent=4)

    @classmethod
    def from_json(cls, path):
        """Load a Table from a JSON

        Args:
            path (str):
                Path of the JSON file to load
        """
        with open(path, 'r') as in_file:
            return cls(json.load(in_file))
