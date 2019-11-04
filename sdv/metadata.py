import copy
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rdt import HyperTransformer, transformers

LOGGER = logging.getLogger(__name__)


def _read_csv_dtypes(table_meta):
    """Get the dtypes specification that needs to be passed to read_csv."""
    dtypes = dict()
    for name, field in table_meta['fields'].items():
        field_type = field['type']
        if field_type == 'categorical':
            dtypes[name] = str
        elif field_type == 'id' and field.get('subtype', 'integer') == 'string':
            dtypes[name] = str

    return dtypes


def _parse_dtypes(data, table_meta):
    """Convert the data columns to the right dtype after loading the CSV."""
    for name, field in table_meta['fields'].items():
        field_type = field['type']
        if field_type == 'datetime':
            data[name] = pd.to_datetime(data[name], format=field['format'], exact=False)
        elif field_type == 'numerical' and field.get('subtype') == 'integer':
            data[name] = data[name].dropna().astype(int)
        elif field_type == 'id' and field.get('subtype', 'integer') == 'integer':
            data[name] = data[name].dropna().astype(int)

    return data


def _load_csv(root_path, table_meta):
    """Load a CSV with the right dtypes and then parse the columns."""
    relative_path = os.path.join(root_path, table_meta['path'])
    dtypes = _read_csv_dtypes(table_meta)

    data = pd.read_csv(relative_path, dtype=dtypes)
    data = _parse_dtypes(data, table_meta)

    return data


class Metadata:
    """Dataset Metadata.

    The Metadata class provides a unified layer of abstraction over the dataset
    metadata, which includes both the necessary details to load the data from
    the hdd and to know how to parse and transform it to numerical data.

    Args:
        metadata (str or dict):
            Path to a ``json`` file that contains the metadata or a ``dict`` representation
            of ``metadata`` following the same structure.

        root_path (str):
            The path where the ``metadata.json`` is located. Defaults to ``None``.
    """

    def _get_relationships(self):
        """Exttract information about child-parent relationships.

        Creates the following attributes:
            * ``_child_map``: set of child tables that each table has.
            * ``_parent_map``: set ot parents that each table has.
        """
        self._child_map = defaultdict(set)
        self._parent_map = defaultdict(set)

        for table_meta in self._metadata['tables'].values():
            if table_meta.get('use', True):
                table = table_meta['name']
                for field_meta in table_meta['fields'].values():
                    ref = field_meta.get('ref')
                    if ref:
                        parent = ref['table']
                        self._child_map[parent].add(table)
                        self._parent_map[table].add(parent)

    @staticmethod
    def _dict_metadata(metadata):
        """Get a metadata ``dict`` with SDV format.

        For each table create a dict of fields from a previous list of fields.

        Args:
            metadata (dict):
                Original metadata to format.

        Returns:
            dict:
                Formated metadata dict.
        """
        new_metadata = copy.deepcopy(metadata)
        tables = new_metadata['tables']
        new_tables = dict()

        for table in tables:
            new_tables[table['name']] = table

            fields = table['fields']
            new_fields = dict()
            for field in fields:
                new_fields[field['name']] = field

            table['fields'] = new_fields

        new_metadata['tables'] = new_tables

        return new_metadata

    def __init__(self, metadata, root_path=None):
        if isinstance(metadata, str):
            self.root_path = root_path or os.path.dirname(metadata)
            with open(metadata) as metadata_file:
                metadata = json.load(metadata_file)
        else:
            self.root_path = root_path or '.'

        self._metadata = self._dict_metadata(metadata)
        self._hyper_transformers = dict()
        self._get_relationships()

    def get_children(self, table_name):
        """Get table children.

        Args:
            table_name (str):
                Name of the table from which to get the children.

        Returns:
            set:
                Set of children for the given table.
        """
        return self._child_map[table_name]

    def get_parents(self, table_name):
        """Get table parents.

        Args:
            table_name (str):
                Name of the table from which to get the parents.

        Returns:
            set:
                Set of parents for the given table.
        """
        return self._parent_map[table_name]

    def get_table_meta(self, table_name):
        """Get the metadata  dict for a table.

        Args:
            table_name (str):
                Name of table to get data for.

        Returns:
            dict:
                table metadata
        """
        return self._metadata['tables'][table_name]

    def load_table(self, table_name):
        """Load table data.

        Args:
            table_name (str):
                Name of the table that we want to load.

        Returns:
            pandas.DataFrame:
                DataFrame with the contents of the table.
        """
        LOGGER.info('Loading table %s', table_name)
        table_meta = self.get_table_meta(table_name)
        return _load_csv(self.root_path, table_meta)

    def _get_dtypes(self, table_name, ids=False):
        """Get a ``dict`` with the ``dtypes`` for each field of a given table.

        Args:
            table_name (str):
                Table name for which to retrive the ``dtypes``.
            ids (bool):
                Whether or not include the id fields. Defaults to ``False``.

        Returns:
            dict:
                Dictionary that contains the field names and data types from a table.

        Raises:
            ValueError:
                If a field has an invalid type or subtype.
        """
        dtypes = dict()
        table_meta = self.get_table_meta(table_name)
        for name, field in table_meta['fields'].items():
            field_type = field['type']
            if field_type == 'categorical':
                dtypes[name] = np.object

            elif field_type == 'boolean':
                dtypes[name] = bool

            elif field_type == 'numerical':
                field_subtype = field.get('subtype', 'float')
                if field_subtype == 'integer':
                    dtypes[name] = int
                elif field_subtype == 'float':
                    dtypes[name] = float
                else:
                    raise ValueError('Invalid {} subtype {} - {}'.format(
                        field_type, field_subtype, name))

            elif field_type == 'datetime':
                dtypes[name] = np.datetime64

            elif field_type == 'id':
                if ids:
                    if (name != table_meta.get('primary_key')) and not field.get('ref'):
                        raise ValueError(
                            'id field `{}` is neither a primary or a foreign key'.format(name))

                    field_subtype = field.get('subtype', 'integer')
                    if field_subtype == 'integer':
                        dtypes[name] = int
                    elif field_subtype == 'string':
                        dtypes[name] = str
                    else:
                        raise ValueError('Invalid {} subtype: {} - {}'.format(
                            field_type, field_subtype, name))

            else:
                raise ValueError('Invalid field type: {} - '.format(field_type, name))

        return dtypes

    def _get_pii_fields(self, table_name):
        """Get the ``pii_category`` for each field that contains PII.

        Args:
            table_name (str):
                Table name for which to get the pii fields.

        Returns:
            dict:
                pii field names and categories.
        """
        pii_fields = dict()
        for name, field in self.get_table_meta(table_name)['fields'].items():
            if field['type'] == 'categorical' and field.get('pii', False):
                pii_fields[name] = field['pii_category']

        return pii_fields

    @staticmethod
    def _get_transformers(dtypes, pii_fields):
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

    def _load_hyper_transformer(self, table_name):
        """Create and return a new ``rdt.HyperTransformer`` instance for a table.

        First get the ``dtypes`` and ``pii fields`` from a given table, then use
        those to build a transformer dictionary to be used by the ``HyperTransformer``.

        Args:
            table_name (str):
                Table name for which to load the HyperTransformer.

        Returns:
            rdt.HyperTransformer:
                Instance of ``rdt.HyperTransformer`` for the given table.
        """
        dtypes = self._get_dtypes(table_name)
        pii_fields = self._get_pii_fields(table_name)
        transformers_dict = self._get_transformers(dtypes, pii_fields)
        return HyperTransformer(transformers=transformers_dict)

    def transform(self, table_name, data):
        """Transform data for a given table.

        If the ``HyperTransformer`` for a table is ``None`` it is created.

        Args:
            table_name (str):
                Name of the table that is being transformer.
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        hyper_transformer = self._hyper_transformers.get(table_name)
        if hyper_transformer is None:
            hyper_transformer = self._load_hyper_transformer(table_name)
            fields = list(hyper_transformer.transformers.keys())
            hyper_transformer.fit(data[fields])
            self._hyper_transformers[table_name] = hyper_transformer

        hyper_transformer = self._hyper_transformers.get(table_name)
        fields = list(hyper_transformer.transformers.keys())
        return hyper_transformer.transform(data[fields])

    def get_table_names(self):
        """Get the list of table names.

        Returns:
            list:
                table names.
        """
        return list(self._metadata['tables'].keys())

    def get_tables(self, tables=None):
        """Get a dictionary with data from multiple tables.

        If a ``tables`` list is given, only load the indicated tables.
        Otherwise, load all the tables from this metadata.

        Args:
            tables (list):
                List of table names. Defaults to ``None``.

        Returns:
            dict(str, pandasd.DataFrame):
                mapping of table names and their data loaded as ``pandas.DataFrame`` instances.
        """
        return {
            table_name: self.load_table(table_name)
            for table_name in tables or self.get_table_names()
        }

    def get_fields(self, table_name):
        """Get table fields metadata.

        Args:
            table_name (str):
                Name of the table to get the fields from.

        Returns:
            dict:
                Mapping of field names and their metadata dicts.
        """
        return self.get_table_meta(table_name)['fields']

    def get_primary_key(self, table_name):
        """Get the primary key name of the indicated table.

        Args:
            table_name (str):
                Name of table for which to get the primary key field.

        Returns:
            str or None:
                Primary key field name. ``None`` if the table has no primary key.
        """
        return self.get_table_meta(table_name).get('primary_key')

    def get_foreign_key(self, parent, child):
        """Get table foreign key field name.

        Args:
            parent (str):
                Name of the parent table.
            child (str):
                Name of the child table.

        Returns:
            str or None:
                Foreign key field name.

        Raises:
            ValueError:
                If the relationship does not exist.
        """
        primary = self.get_primary_key(parent)

        for field in self.get_fields(child).values():
            ref = field.get('ref')
            if ref and ref['field'] == primary:
                return field['name']

        raise ValueError('{} is not parent of {}'.format(parent, child))

    def reverse_transform(self, table_name, data):
        """Reverse the transformed data for a given table.

        Args:
            table_name (str):
                Name of the table to reverse transform.
            data (pandas.DataFrame):
                Data to be reversed.

        Returns:
            pandas.DataFrame
        """
        hyper_transformer = self._hyper_transformers[table_name]
        reversed_data = hyper_transformer.reverse_transform(data)

        for name, dtype in self._get_dtypes(table_name, ids=True).items():
            reversed_data[name] = reversed_data[name].dropna().astype(dtype)

        return reversed_data
