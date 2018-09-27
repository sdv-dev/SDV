import copy
import json
import os

import pandas as pd
from rdt.hyper_transformer import HyperTransformer


class Table:
    """Class that represents a table object."""

    def __init__(self, data, meta):
        self.data = data
        self.meta = meta


class DataLoader:
    """Abstract class responsible for loading data and returning a DataNavigator."""

    def __init__(self, meta_filename):
        """Instantiate data loader object."""
        self.meta_filename = meta_filename

        with open(meta_filename) as f:
            self.meta = json.load(f)

    def load_data(self):
        raise NotImplementedError


class CSVDataLoader(DataLoader):
    """Data loader class used for loading data from csvs."""

    def _format_table_meta(self, table_meta):
        """Format table meta to turn fields into dictionary."""
        new_fields = {}

        for field in table_meta['fields']:
            field_name = field['name']
            new_fields[field_name] = field

        table_meta['fields'] = new_fields
        return table_meta

    def load_data(self):
        """Load data from csvs and returns DataNavigator."""
        meta = copy.deepcopy(self.meta)
        tables = {}
        prefix = os.path.dirname(self.meta_filename)

        for table_meta in meta['tables']:
            if table_meta['use']:
                formatted_table_meta = self._format_table_meta(table_meta)
                relative_path = os.path.join(prefix, meta['path'], table_meta['path'])
                data_table = pd.read_csv(relative_path)
                tables[table_meta['name']] = Table(data_table, formatted_table_meta)

        return DataNavigator(self.meta_filename, self.meta, tables)


class DataNavigator:
    """Class to navigate through data set."""

    DEFAULT_TRANSFORMERS = ['NumberTransformer', 'DTTransformer', 'CatTransformer']

    def __init__(self, meta_filename, meta, tables):
        """Instantiate data navigator object."""
        self.meta = meta
        self.tables = tables
        self.ht = HyperTransformer(meta_filename)
        self.transformed_data = None
        self.child_map, self.parent_map, self.foreign_keys = self._get_relationships(self.tables)

    def get_children(self, table_name):
        """Return set of children of a table.

        Args:
            table_name (str): Name of table to get children of.

        Returns:
            set: Set of children for the given table.
        """
        return self.child_map.get(table_name, set())

    def get_parents(self, table_name):
        """Returns parents of a table.

        Args:
            table_name (str): Name of table to get parents of.

        Returns:
            set: Set of parents for the given table.
        """
        return self.parent_map.get(table_name, set())

    def get_data(self, table_name):
        """Return dataframe for a table.

        Args:
            table_name (str): Name of table to get data for.

        Returns:
            pandas.DataFrame: DataFrame with the contents of table_name
        """
        return self.tables[table_name].data

    def get_meta_data(self, table_name):
        """Return meta data for a table.

        Args:
            table_name (str): Name of table to get data for.

        Returns:
            dict: metadata for table_name
        """
        return self.tables[table_name].meta

    def transform_data(self, transformers=None, missing=False):
        """Applies the specified transformations using an HyperTransformer and returns the new data

        Args:
            transformers (list): List of transformers to use.
            missing (bool): Whether or not to keep track of missing variables
                            and create extra columns for them.

        Returns:
            dict: dict with the transformed dataframes.
        """
        transformers = transformers or self.DEFAULT_TRANSFORMERS
        self.transformed_data = self.ht.fit_transform(
            transformer_list=transformers, missing=missing)

        return self.transformed_data

    def update_mapping(self, mapping, key, value):
        """Safely updates a dict of sets.

        Args:
            mapping (dict): Dictionary to be updated.
            key(string): Key to update on `mapping`.
            value: Value to add.

        Returns:
            dict: Updated mapping.

        If mapping[key] exists then value will be added to it.
        If not, it will be created as a single-element set containing `value`.
        """
        item = mapping.get(key)

        if item:
            item.add(value)

        else:
            mapping[key] = {value}

        return mapping

    def _get_relationships(self, tables):
        """Map table name to names of child tables.

        Arguments:
            tables (dict): table_name -> Table.

        Returns:
            tuple: dicts of children, parents and foreign_keys.

        This method is what allow `DataNavigator` to be aware of the different tables and the
        relations between them.
        """
        child_map = {}
        parent_map = {}
        foreign_keys = {}  # {(child, parent) -> (parent pk, fk)}

        for table in tables:
            table_meta = tables[table].meta
            for field_meta in table_meta['fields'].values():
                ref = field_meta.get('ref')
                if ref:
                    parent = ref['table']
                    parent_pk = ref['field']
                    fk = field_meta['name']

                    # update child map
                    child_map = self.update_mapping(child_map, parent, table)

                    # update parent map
                    parent_map = self.update_mapping(parent_map, table, parent)

                    foreign_keys[(table, parent)] = (parent_pk, fk)

        return (child_map, parent_map, foreign_keys)
