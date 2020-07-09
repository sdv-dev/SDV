import copy
import json
import logging
import os

import numpy as np
import pandas as pd
from faker import Faker
from rdt import HyperTransformer, transformers

from sdv.metadata.error import MetadataError

LOGGER = logging.getLogger(__name__)


class Table:
    """Table Metadata.

    The Metadata class provides a unified layer of abstraction over the metadata
    of a single Table, which includes both the necessary details to load the data
    from the filesystem and to know how to parse and transform it to numerical data.

    Args:
        metadata (str or dict):
            Path to a ``json`` file that contains the metadata or a ``dict`` representation
            of ``metadata`` following the same structure.
        root_path (str):
            The path to which the dataset is located. Defaults to ``None``.
    """

    _metadata = None
    _hyper_transformer = None
    _root_path = None

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

    def __init__(self, metadata=None, root_path=None, field_names=None, primary_key=None,
                 field_types=None, anonymize_fields=None, constraints=None,):
        self._metadata = metadata or dict()
        self._hyper_transformer = None
        self._root_path = None

        self._field_names = field_names
        self._primary_key = primary_key
        self._field_types = field_types or {}
        self._anonymize_fields = anonymize_fields
        self._constraints = constraints

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

    def _get_faker(self):
        """Return the faker object to anonymize data.

        Returns:
            function:
                Faker function to generate new data instances with ``self.anonymize`` arguments.

        Raises:
            ValueError:
                A ``ValueError`` is raised if the faker category we want don't exist.
        """
        if isinstance(self._anonymize, (tuple, list)):
            category, *args = self._anonymize
        else:
            category = self._anonymize
            args = tuple()

        try:
            faker_method = getattr(Faker(), category)

            def faker():
                return faker_method(*args)

            return faker
        except AttributeError:
            raise ValueError('Category "{}" couldn\'t be found on faker'.format(self.anonymize))

    def _anonymize(self, data):
        """Anonymize data and save the anonymization mapping in-memory."""
        # TODO: Do this by column
        faker = self._get_faker()
        uniques = data.unique()
        fake_data = [faker() for x in range(len(uniques))]

        mapping = dict(zip(uniques, fake_data))
        MAPS[id(self)] = mapping

        return data.map(mapping)

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
            if not field_name in data:
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

    def _get_pii_fields(self):
        """Get the ``pii_category`` for each field that contains PII.

        Returns:
            dict:
                pii field names and categories.
        """
        pii_fields = dict()
        for name, field in self._metadata['fields'].items():
            if field['type'] == 'categorical' and field.get('pii', False):
                pii_fields[name] = field['pii_category']

        return pii_fields

    def _get_transformers(self, dtypes):
        """Create the transformer instances needed to process the given dtypes.

        Temporary drop-in replacement of ``HyperTransformer._analyze`` method,
        before RDT catches up.

        Args:
            dtypes (dict):
                mapping of field names and dtypes.
            pii_fields (dict):
                mapping of pii field names and categories.

        Returns:
            dict:
                mapping of field names and transformer instances.
        """
        transformers_dict = dict()
        pii_fields = self._get_pii_fields()
        for name, dtype in dtypes.items():
            dtype = np.dtype(dtype)
            if dtype.kind == 'i':
                transformer = transformers.NumericalTransformer(dtype=int)
            elif dtype.kind == 'f':
                transformer = transformers.NumericalTransformer(dtype=float)
            elif dtype.kind == 'O':
                anonymize = pii_fields.get(name)
                transformer = transformers.CategoricalTransformer(anonymize=anonymize)
            elif dtype.kind == 'b':
                transformer = transformers.BooleanTransformer()
            elif dtype.kind == 'M':
                transformer = transformers.DatetimeTransformer()
            else:
                raise ValueError('Unsupported dtype: {}'.format(dtype))

            LOGGER.info('Loading transformer %s for field %s',
                        transformer.__class__.__name__, name)
            transformers_dict[name] = transformer

        return transformers_dict

    def _fit_hyper_transformer(self, data):
        """Create and return a new ``rdt.HyperTransformer`` instance.

        First get the ``dtypes`` and then use them to build a transformer dictionary
        to be used by the ``HyperTransformer``.

        Returns:
            rdt.HyperTransformer
        """
        dtypes = self.get_dtypes(ids=False)
        transformers_dict = self._get_transformers(dtypes)
        self._hyper_transformer = HyperTransformer(transformers=transformers_dict)
        self._hyper_transformer.fit(data[list(dtypes.keys())])

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
            if field_name not in self.get_fields():
                raise ValueError('Field "{}" does not exist in this table'.format(field_name))

            field_metadata = self._metadata['fields'][field_name]
            field_subtype = self._get_key_subtype(field_metadata)

            field_metadata.update({
                'type': 'id',
                'subtype': field_subtype
            })

        self._primary_key = field_name

    def fit(self, data):
        """Fit this metadata to the given data.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
        """
        self._field_names = self._field_names or list(data.columns)
        self._metadata['fields'] = self._build_fields_metadata(data)
        self.set_primary_key(self._primary_key)

        # TODO: Treat/Learn constraints

        self._fit_hyper_transformer(data)

    def get_fields(self):
        """Get fields metadata.

        Returns:
            dict:
                Mapping of field names and their metadata dicts.
        """
        return self._metadata['fields']

    def transform(self, data):
        """Transform the given data.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """

        # TODO: Do this by column
        # if self.anonymize:
        #     data = data.map(MAPS[id(self)])

        fields = list(self._hyper_transformer.transformers.keys())
        return self._hyper_transformer.transform(data[fields])

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

    def get_children(self):
        """Get tables for which this table is parent.

        Returns:
            set:
                Set of children for this table.
        """
        return self._children

    def get_parents(self):
        """Get tables for with this table is child.

        Returns:
            set:
                Set of parents for this table.
        """
        return self._parents

    def get_field(self, field_name):
        """Get the metadata dict for a field.

        Args:
            field_name (str):
                Name of the field to get data for.

        Returns:
            dict:
                field metadata

        Raises:
            ValueError:
                If the table or the field do not exist in this metadata.
        """
        field_meta = self._metadata['fields'].get(field_name)
        if field_meta is None:
            raise ValueError('Invalid field name "{}"'.format(field_name))

        return copy.deepcopy(field_meta)

    # def _read_csv_dtypes(self):
    #     """Get the dtypes specification that needs to be passed to read_csv."""
    #     dtypes = dict()
    #     for name, field in self._metadata['fields'].items():
    #         field_type = field['type']
    #         if field_type == 'categorical':
    #             dtypes[name] = str
    #         elif field_type == 'id' and field.get('subtype', 'integer') == 'string':
    #             dtypes[name] = str

    #     return dtypes

    # def _parse_dtypes(self, data):
    #     """Convert the data columns to the right dtype after loading the CSV."""
    #     for name, field in self._metadata['fields'].items():
    #         field_type = field['type']
    #         if field_type == 'datetime':
    #             datetime_format = field.get('format')
    #             data[name] = pd.to_datetime(data[name], format=datetime_format, exact=False)
    #         elif field_type == 'numerical' and field.get('subtype') == 'integer':
    #             data[name] = data[name].dropna().astype(int)
    #         elif field_type == 'id' and field.get('subtype', 'integer') == 'integer':
    #             data[name] = data[name].dropna().astype(int)

    #     return data

    # def load(self):
    #     """Load table data.

    #     First load the CSV with the right dtypes and then parse the columns
    #     to the final dtypes.

    #     Returns:
    #         pandas.DataFrame:
    #             DataFrame with the contents of the table.
    #     """
    #     relative_path = os.path.join(self.root_path, self.path)
    #     dtypes = self._read_csv_dtypes()

    #     data = pd.read_csv(relative_path, dtype=dtypes)
    #     data = self._parse_dtypes(data)

    #     return data

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

    def _check_field(self, field, exists=False):
        """Validate the existance of the table and existance (or not) of field."""
        table_fields = self.get_fields(table)
        if exists and (field not in table_fields):
            raise ValueError('Field "{}" does not exist in table "{}"'.format(field, table))

        if not exists and (field in table_fields):
            raise ValueError('Field "{}" already exists in table "{}"'.format(field, table))

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
