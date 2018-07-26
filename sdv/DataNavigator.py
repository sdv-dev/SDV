import copy
import json
import os.path as op

import pandas as pd
from rdt.hyper_transformer import HyperTransformer


class DataLoader:
    """ Abstract class responsible for loading data and returning a
    DataNavigator """

    def __init__(self, meta_filename):
        """ Instantiates data loader object """
        self.meta_filename = meta_filename
        with open(meta_filename) as f:
            self.meta = json.load(f)

    def load_data(self):
        raise NotImplementedError


class CSVDataLoader(DataLoader):
    """ Data loader class used for loading data from csvs """

    def _format_table_meta(self, table_meta):
        """ reformats table meta to turn fields into dictionary """
        new_fields = {}
        for field in table_meta['fields']:
            field_name = field['name']
            new_fields[field_name] = field
        table_meta['fields'] = new_fields
        return table_meta

    def load_data(self):
        """ loads data from csvs and returns DataNavigator """
        meta = copy.deepcopy(self.meta)
        data = {}
        for table_meta in meta['tables']:
            if table_meta['use']:
                formatted_table_meta = self._format_table_meta(table_meta)
                prefix = op.dirname(self.meta_filename)
                relative_path = op.join(prefix, meta['path'],
                                        table_meta['path'])
                data_table = pd.read_csv(relative_path)
                data[table_meta['name']] = (data_table, formatted_table_meta)
        return DataNavigator(self.meta_filename, self.meta, data)


class DataNavigator:
    """ Class to navigate through data set """

    def __init__(self, meta_filename, meta, data):
        """ Instantiates data navigator object """
        self.meta = meta
        self.data = data
        self.ht = HyperTransformer(meta_filename)
        self.transformed_data = None
        relationships = self._get_relationships(self.data)
        self.child_map = relationships[0]
        self.parent_map = relationships[1]
        self.foreign_keys = relationships[2]

    def get_children(self, table_name):
        """ returns children of a table
        Args:
            table_name (str): name of table to get children of
        """
        return self.child_map.get(table_name, set())

    def get_parents(self, table_name):
        """ returns parents of a table
        Args:
            table_name (str): name of table to get parents of
        """
        return self.parent_map.get(table_name, set())

    def transform_data(self, transformers=None, missing=False):
        """ Applies the specified transformations using
        a hyper transformer and returns the new data
        Args:
            transformers (list): List of transformers to use
            missing (bool): Whether or not to keep track of
            missing variables and create extra columns for them.
        Returns:
            transformed_data (dict): dict with keys that are
            the names of the tables and values that are the
            transformed dataframes.
        """
        if transformers is None:
            transformers = ['NumberTransformer',
                            'DTTransformer',
                            'CatTransformer']
        self.transformed_data = self.ht.hyper_fit_transform(
            transformer_list=transformers, missing=missing)
        return self.transformed_data

    def _get_relationships(self, data):
        """ maps table name to names of child tables """
        child_map = {}
        parent_map = {}
        foreign_keys = {}  # {(child, parent) -> (parent pk, fk)}
        for table in data:
            table_meta = data[table][1]
            for field in table_meta['fields']:
                field_meta = table_meta['fields'][field]
                if 'ref' in field_meta:
                    parent = field_meta['ref']['table']
                    parent_pk = field_meta['ref']['field']
                    fk = field_meta['name']
                    # update child map
                    if parent in child_map:
                        child_map[parent].add(table)
                    else:
                        child_map[parent] = {table}
                    # update parent map
                    if table in parent_map:
                        parent_map[table].add(parent)
                    else:
                        parent_map[table] = {parent}
                    foreign_keys[(table, parent)] = (parent_pk, fk)
        return (child_map, parent_map, foreign_keys)
