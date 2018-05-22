import json
import pandas as pd
import os.path as op
from utils import format_table_meta


class DataNavigator:
    """ Class to navigate through data set """
    def __init__(self, meta_filename):
        """ Instantiates data navigator object """
        with open(meta_filename) as f:
            self.meta = json.load(f)
        self.data = self._parse_data(self.meta, meta_filename)
        self.child_map, self.parent_map = self._get_relationships(self.data)

    def _parse_data(self, meta, meta_filename):
        """ extracts the data from a meta.json object
        and maps tabls name to tuple (dataframe, table meta) """
        data = {}
        for table_meta in meta['tables']:
            if table_meta['use']:
                formatted_table_meta = format_table_meta(table_meta)
                prefix = op.dirname(meta_filename)
                relative_path = op.join(prefix, meta['path'],
                                        table_meta['path'])
                data_table = pd.read_csv(relative_path)
                data[table_meta['name']] = (data_table, formatted_table_meta)
        return data

    def _get_relationships(self, data):
        """ maps table name to names of child tables """
        child_map = {}
        parent_map = {}
        for table in data:
            table_meta = data[table][1]
            for field in table_meta['fields']:
                field_meta = table_meta['fields'][field]
                if 'ref' in field_meta:
                    parent = field_meta['ref']['table']
                    # update child map
                    if parent in child_map:
                        child_map[parent].add(table)
                    else:
                        child_map[parent] = {table}
                    # update parent map
                    if table in parent_map:
                        parent_map[table].add(parent)
                    else:
                        parent_map[table] = {table}
        return (child_map, parent_map)
