import copy
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rdt.hyper_transformer import HyperTransformer

LOGGER = logging.getLogger(__name__)


def _read_csv_dtypes(table_meta):
    dtypes = dict()
    for name, field in table_meta['fields'].items():
        field_type = field['type']
        if field_type == 'categorical':
            if field.get('subtype', 'categorical') == 'categorical':
                dtypes[name] = str

    return dtypes


def _parse_dtypes(data, table_meta):
    for name, field in table_meta['fields'].items():
        field_type = field['type']
        if field_type == 'datetime':
            data[name] = pd.to_datetime(data[name], format=field['format'])
        elif field_type == 'number' and field.get('subtype') == 'integer':
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
        Transform the dataset using `rdt.HyperTransformer` in a format that is supported
        by `sdv.Modeler`.

    Args:
        meta_filename (str): Path to the metadata file.
        meta (dict): Metadata for the dataset.
        tables (dict[str, pd.DataFrame]): Mapping of table names to their values and metadata.

    """

    def _get_relationships(self):
        """Map table name to names of child tables.

        Returns:
            tuple: dicts of children, parents and foreign_keys.

        This method is what allow `DataNavigator` to be aware of the different tables and the
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

    def __init__(self, metadata, root_path='.'):
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
            dict: metadata for table_name
        """
        return self._metadata['tables'][table_name]

    def load_table(self, table_name):
        """Return dataframe for a table.

        Args:
            table_name (str): Name of table to get data for.

        Returns:
            pandas.DataFrame: DataFrame with the contents of table_name
        """
        LOGGER.info('Loading table %s', table_name)
        table_meta = self.get_table_meta(table_name)
        return load_csv(self.root_path, table_meta)

    def _get_dtypes(self, table_name):
        dtypes = list()
        for name, field in self.get_fields(table_name).items():
            field_type = field['type']
            if field_type == 'categorical':
                field_subtype = field.get('subtype', 'categorical')
                if field_subtype == 'categorical':
                    dtypes.append(np.object)
                elif field_subtype == 'bool':
                    dtypes.append(bool)
                else:
                    raise ValueError('Invalid {} subtype: {}'.format(field_type, field_subtype))

            elif field_type == 'number':
                field_subtype = field.get('subtype', 'float')
                if field_subtype == 'integer':
                    dtypes.append(int)
                elif field_subtype == 'float':
                    dtypes.append(float)
                else:
                    raise ValueError('Invalid {} subtype: {}'.format(field_type, field_subtype))

            elif field_type == 'datetime':
                dtypes.append(np.datetime64)

        return dtypes

    def _get_pii_fields(self, table_name):
        pii_fields = dict()
        for name, field in self.get_fields(table_name).items():
            if field['type'] == 'categorical' and field.get('pii', False):
                pii_fields[name] = field['pii_category']

        return pii_fields

    def _load_hyper_transformer(self, table_name):
        dtypes = self._get_dtypes(table_name)
        pii_fields = self._get_pii_fields(table_name)
        return HyperTransformer(anonymize=pii_fields, dtypes=dtypes)

    def get_table_data(self, table_name, transform=False):
        table = self.load_table(table_name)

        hyper_transformer = self._hyper_transformers.get(table_name)
        if hyper_transformer is None:
            hyper_transformer = self._load_hyper_transformer(table_name)
            hyper_transformer.fit(table)
            self._hyper_transformers[table_name] = hyper_transformer

        if transform:
            return hyper_transformer.transform(table)

        return table

    def get_table_names(self):
        return list(self._metadata['tables'].keys())

    def get_tables(self, tables=None):
        return {
            table_name: self.get_table_data(table_name)
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

    def get_field_names(self, table_name):
        """Return table field names.

        Args:
            table_name (str):
                Name of table to get data for.

        Returns:
            list:
                table field names
        """
        return list(self.get_fields(table_name).keys())

    def get_field_meta(self, table_name, field_name):
        """Return field metadata.

        Args:
            table_name (str):
                Name of table to get meta data for.
            field_name (str):
                Name of field to get meta data for.

        Returns:
            dict:
                field metadata
        """
        return self.get_fields(table_name)[field_name]

    def get_primary_key(self, table_name):
        return self.get_table_meta(table_name).get('primary_key')

    def get_foreign_key(self, parent, child):
        primary = self.get_primary_key(parent)

        for field in self.get_fields(child).values():
            ref = field.get('ref')
            if ref and ref['field'] == primary:
                return field['name']

    def reverse_transform(self, table_name, data):
        hyper_transformer = self._hyper_transformers[table_name]
        return hyper_transformer.reverse_transform(data)
