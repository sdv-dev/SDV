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
    dtypes = dict()
    for name, field in table_meta['fields'].items():
        field_type = field['type']
        if field_type == 'categorical' and field.get('subtype', 'categorical') == 'categorical':
            dtypes[name] = str
        elif field_type == 'id' and field.get('subtype') == 'string':
            dtypes[name] = str

    return dtypes


def _parse_dtypes(data, table_meta):
    for name, field in table_meta['fields'].items():
        field_type = field['type']
        if field_type == 'datetime':
            data[name] = pd.to_datetime(data[name], format=field['format'], exact=False)
        elif field_type == 'number' and field.get('subtype') == 'integer':
            data[name] = data[name].dropna().astype(int)
        elif field_type == 'id' and field.get('subtype') == 'number':
            data[name] = data[name].dropna().astype(int)

    return data


def load_csv(root_path, table_meta):
    relative_path = os.path.join(root_path, table_meta['path'])
    dtypes = _read_csv_dtypes(table_meta)

    data = pd.read_csv(relative_path, dtype=dtypes)
    data = _parse_dtypes(data, table_meta)

    return data


class Metadata:
    """Navigate through and transform a dataset.

    This class implement two main functionalities:

    - Navigation through the dataset
        Given a table, it allows to move though its relations and acces its data and metadata.

    - Transform data
        Transform the dataset using ``rdt.HyperTransformer`` in a format that is supported
        by ``sdv.Modeler``.

    Args:
        meta_filename (str):
            Path to the metadata file.
        meta (dict):
            Metadata for the dataset.
        tables (dict[str, pandas.DataFrame]):
            Mapping of table names to their values and metadata.
    """

    def _get_relationships(self):
        """Map table name to names of child tables.

        Returns:
            tuple:
                dicts of children, parents and foreign_keys.

        This method is what allow ``DataNavigator`` to be aware of the different tables and the
        relations between them.
        """
        self._child_map = defaultdict(set)
        self._parent_map = defaultdict(set)
        self.foreign_keys = dict()

        for table_meta in self._metadata['tables'].values():
            if table_meta['use']:
                table = table_meta['name']
                for field_meta in table_meta['fields'].values():
                    ref = field_meta.get('ref')
                    if ref:
                        parent = ref['table']
                        parent_pk = ref['field']
                        fk = field_meta['name']

                        self._child_map[parent].add(table)
                        self._parent_map[table].add(parent)
                        self.foreign_keys[(table, parent)] = (parent_pk, fk)

    @staticmethod
    def _dict_metadata(metadata):
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
        """Return set of children of a table.

        Args:
            table_name (str):
                Name of the table from which to get the children.

        Returns:
            set:
                Set of children for the given table.
        """
        return self._child_map[table_name]

    def get_parents(self, table_name):
        """Returns parents of a table.

        Args:
            table_name (str):
                Name of the table from which to get the parents.

        Returns:
            set:
                Set of parents for the given table.
        """
        return self._parent_map[table_name]

    def get_table_meta(self, table_name):
        """Return meta data for a table.

        Args:
            table_name (str): Name of table to get data for.

        Returns:
            dict:
                metadata for table_name
        """
        return self._metadata['tables'][table_name]

    def load_table(self, table_name):
        """Return dataframe for a table.

        Args:
            table_name (str):
                Name of table to get data for.

        Returns:
            pandas.DataFrame:
                DataFrame with the contents of table_name
        """
        LOGGER.info('Loading table %s', table_name)
        table_meta = self.get_table_meta(table_name)
        return load_csv(self.root_path, table_meta)

    def _get_dtypes(self, table_name, ids=False):
        """Get dictionaty with the data type for each table field.

        Args:
            table_name (str):
                Table name to get their fields metadata.
            ids (bool):
                Whether or not include the id fields. Defaults to ``False``.

        Returns:
            dict:
                Dictionary which contains the field names and data types from a table.

        Raises:
            ValueError:
                A ``ValueError`` is raised when a field has an invalid subtype.
        """
        dtypes = dict()
        for name, field in self.get_table_meta(table_name)['fields'].items():
            field_type = field['type']
            if field_type == 'categorical':
                field_subtype = field.get('subtype', 'categorical')
                if field_subtype == 'categorical':
                    dtypes[name] = np.object
                elif field_subtype == 'bool':
                    dtypes[name] = bool
                else:
                    raise ValueError('Invalid {} subtype: {}'.format(field_type, field_subtype))

            elif field_type == 'number':
                field_subtype = field.get('subtype', 'float')
                if field_subtype == 'integer':
                    dtypes[name] = int
                elif field_subtype == 'float':
                    dtypes[name] = float
                else:
                    raise ValueError('Invalid {} subtype: {}'.format(field_type, field_subtype))

            elif field_type == 'datetime':
                dtypes[name] = np.datetime64

            elif ids and field_type == 'id':
                field_subtype = field.get('subtype', 'string')
                if field_subtype == 'number':
                    dtypes[name] = int
                elif field_subtype == 'string':
                    dtypes[name] = str
                else:
                    raise ValueError('Invalid {} subtype: {}'.format(field_type, field_subtype))

        return dtypes

    def _get_pii_fields(self, table_name):
        """Get dictionary with the categorical types for each table field.

        Args:
            table_name (str):
                Table name to get their fields metadata.

        Returns:
            dict:
                Dictionary which contains the field names and categorical types from a table.
        """
        pii_fields = dict()
        for name, field in self.get_table_meta(table_name)['fields'].items():
            if field['type'] == 'categorical' and field.get('pii', False):
                pii_fields[name] = field['pii_category']

        return pii_fields

    @staticmethod
    def _get_transformers(dtypes, pii_fields):
        """Build a ``dict`` with column names and transformers from a given ``pandas.DataFrame``.

        Temporary drop-in replacement of ``HyperTransformer._analyze`` method,
        before RDT catches up.

        Args:
            dtypes (dict):
                Data type dict per field to get their transformers.
            pii_fields (dict):
                Fields to be anonymized with the ``CategoricalTransformer``.

        Returns:
            dict:
                Transformers dictionary.
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

            transformers_dict[name] = transformer

        return transformers_dict

    def _load_hyper_transformer(self, table_name):
        """Create and return a new ``rdt.HyperTransformer`` instance for a table.

        First, get the data types and pii fields from a given table.
        After that, use them to build a transformer dictionary to pass it to the
        ``HyperTransformer``.

        Args:
            table_name (str):
                Table name to get their data types and pii fields.

        Returns:
            rdt.HyperTransformer
        """
        dtypes = self._get_dtypes(table_name)
        pii_fields = self._get_pii_fields(table_name)
        transformers_dict = self._get_transformers(dtypes, pii_fields)
        return HyperTransformer(transformers=transformers_dict)

    def transform(self, table_name, data):
        """Returns transformed data for a table.

        If the ``HyperTransformer`` for a table is ``None``, then its created.

        Args:
            table_name (str):
                Name of the table to transform the data.
            data (pandas.DataFrame):
                Table to be transformed.

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
        """Returns list of table names.

        Returns:
            list:
                List of table names.
        """
        return list(self._metadata['tables'].keys())

    def get_tables(self, tables=None):
        """Get a dictionary with the ``pandas.DataFrame`` for each table.

        Tables can be specified in ``tables`` or it will use the output from
        ``self.get_table_names()``.

        Args:
            tables (list):
                List with the table names to load. Defaults to None.

        Returns:
            dict:
                Dictionary which contains the table names and their corresponding
                ``pandas.DataFrame``.
        """
        return {
            table_name: self.load_table(table_name)
            for table_name in tables or self.get_table_names()
        }

    def get_fields(self, table_name):
        """Return table fields metadata.

        Args:
            table_name (str):
                Name of table to get data for.

        Returns:
            dict:
                table fields metadata
        """
        return self.get_table_meta(table_name)['fields']

    def get_primary_key(self, table_name):
        """Return table primary key field name.

        Args:
            table_name (str):
                Name of table to get data for.

        Returns:
            str or None:
                Primary key field name or ``None`` if the table doesn't have primary key.
        """
        return self.get_table_meta(table_name).get('primary_key')

    def get_foreign_key(self, parent, child):
        """Return table foreign key field name.

        Args:
            parent (str):
                Name of parent table to get the primary key field name.
            child (str):
                Name of child table to get the foreign key field name.

        Returns:
            str or None:
                Foreign key field name of ``None`` if the child table doesn't have foreign key.
        """
        primary = self.get_primary_key(parent)

        for field in self.get_fields(child).values():
            ref = field.get('ref')
            if ref and ref['field'] == primary:
                return field['name']

    def reverse_transform(self, table_name, data):
        """Reverse the transformed data for a given table.

        Call to the ``HyperTransformer.reverse_data()`` for the given table and get the table
        data types to set the original data type.

        Args:
            table_name (str):
                Table name to reverse the transformed data.
            data (pandas.DataFrame):
                Table data to be reversed.

        Returns:
            pandas.DataFrame
        """
        hyper_transformer = self._hyper_transformers[table_name]
        reversed_data = hyper_transformer.reverse_transform(data)

        for name, dtype in self._get_dtypes(table_name, ids=True).items():
            reversed_data[name] = reversed_data[name].dropna().astype(dtype)

        return reversed_data
