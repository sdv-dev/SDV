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

    def _analyze(data, columns=None):
        """Get a dictionary with the metadata analyzed from a dictionary.

        Analyze a ``pandas.DataFrame`` to build a ``dict`` with the name of the column, and
        their data type and subtype. If ``columns`` are provided, only those columns will be
        analyzed.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
            columns(list):
                List of columns used to specify which fields analyze from the data.

        Returns:
            dict:
                Generated metadata from a ``pandas.DataFrame``.

        Raises:
            ValueError:
                A ``ValueError`` is raised when a column from the data analyzed is an unsupported
                data type.
        """
        fields = dict()
        for column in columns or data.columns:
            dtype = data[column].dtype
            subtype = None

            if dtype.kind == 'i':
                type = 'numerical'
                subtype = 'integer'

            elif dtype.kind == 'f':
                type = 'numerical'
                subtype = 'float'

            elif dtype.kind == 'O':
                type = 'categorical'

            elif dtype.kind == 'b':
                type = 'boolean'

            elif dtype.kind == 'M':
                type = 'datetime'

            else:
                raise ValueError('Unsupported dtype: {} in column {}'.format(dtype, column))

            fields[column] = {
                'name': column,
                'type': type
            }

            if subtype:
                fields[column]['subtype'] = subtype

        return fields

    def _validate_field(self, field):
        dtype = field['type']
        if dtype == 'categorical':
            pass

        elif dtype == 'id':
            pass

        elif dtype == 'numerical':
            subtype = field.get('subtype')
            if subtype and subtype != 'integer' and subtype != 'float':
                raise ValueError()

        elif dtype == 'boolean':
            pass

        elif dtype == 'datetime':
            pass

        else:
            raise ValueError('Type {} is not supported.'.format(dtype))

    def _validate_circular_relationships(self, parent, children=None):
        if children is None:
            children = self.get_children(parent)

        if parent in children:
            raise ValueError('Circular relationship not supported')

        for grandchild in children:
            self._validate_circular_relationships(parent, self.get_children(grandchild))

    def add_field(self, table, field, type, subtype=None, properties=dict()):
        """Add a new field into a given table.

        Before add the field validate the ``table`` already exists, the ``field`` does not exist,
        ``type`` and ``subtype`` supporteds, and ``properties`` valid properties.

        The error message displayed, when the ``ValueError`` is raised because the ``fields``
        already exists in the table, recommends you to use the ``update_fields`` instead.

        Args:
            table (str):
                Table name to add the new field, it must exist.
            field (str):
                Field name to be added, it must not exist.
            type (str):
                Data type of field to be added. Required.
            subtype (str):
                Data subtype of field to be added. Not required. Defaults to ``None``.
            properties (dict):
                Extra properties of field like: ref, format, min, max, etc. Not required.
                Defaults to ``dict()``.

        Raises:
            ValueError:
                A ``ValueError`` is raised when the ``table`` doesn't exists or the ``field``
                exists in the table.
        """
        if table not in self.get_table_names():
            raise ValueError('Table "{}" doesn\'t exists.'.format(table))

        fields = self.get_fields(table)
        if field in fields:
            raise ValueError(
                'Table {}, field {} already exists. Use "update_field()" to modify it.'
                .format(table, field)
            )

        field_details = {
            'name': field,
            'type': type
        }

        if subtype:
            field_details['subtype'] = subtype

        if properties:
            for property_name, property_value in properties.items():
                field_details[property_name] = property_value

        self._validate_field(field_details)

        fields[field] = field_details

    def add_primary_key(self, table, field):
        """Add a primary key into a given table.

        First, assert that the ``table`` exists and the ``field`` does not.
        Then, assert that the ``table`` doesn't have primary key. Finally, add primary key.

        Args:
            table (str):
                Table name to add the new primary key, it must exist.
            field (str):
                Field name to be the new primary key, it must not exist.

        Raises:
            ValueError:
                A ``ValueError`` is raised when the table not exist,
                the field already exist or the primary key already exist.
        """
        if table not in self.get_table_names():
            raise ValueError('Table "{}" doesn\'t exists.'.format(table))

        if field in self.get_fields(table).keys():
            raise ValueError('Table {}, field {} already exists.'.format(table, field))

        if self.get_primary_key(table):
            raise ValueError('Table {} already have primary key.'.format(table))

        self.get_table_meta(table)['primary_key'] = field
        self.add_field(table, field, 'id', None, None)

    def add_relationship(self, table, parent, foreign_key=None):
        """Add a new relationship between a 2 tables.

        By a given ``table`` and ``parent`` add a new relationship using ``foreign_key``
        to reference the parent field, when ``foreign_key`` is ``None`` use the primary key
        field name as foreign key.

        First, assert that the ``table`` and ``parent`` exists.
        Then, assert if already exists a relationship between both tables.
        If not, add their relationships, in ``table`` add a new parent ``parent``
        and in ``parent`` add a new children ``table``.

        Args:
            table (str):
                Table name to add a new relationship with a parent table.
            parent (str):
                Table name to add a new relationship with a children table.
            foreign_key (str):
                Field name from the parent table to create the reference in the children table.
                Defaults to ``None``.

        Raises:
            ValueError:
                A ``ValueError`` is raised when ``table`` or ``parent`` don't exist,
                the relationship already exists, ``parent`` is child of ``table`` of
                ``parent`` table does not have primary key.
        """
        if table not in self.get_table_names():
            raise ValueError('Table "{}" doesn\'t exists.'.format(table))

        if parent not in self.get_table_names():
            raise ValueError('Table "{}" doesn\'t exists.'.format(parent))

        if parent in self.get_parents(table):
            raise ValueError('Table {} is the parent table of {}.'.format(parent, table))

        if parent in self.get_children(table):
            raise ValueError('Table {} is the children table of {}.'.format(parent, table))

        primary_key = self.get_primary_key(parent)
        if not primary_key:
            raise ValueError('Parent table {} have not primary key.'.format(primary_key))

        self._validate_circular_relationships(parent, self.get_children(table))

        properties = {'ref': {'field': primary_key, 'table': parent}}
        self.add_field(table, foreign_key or primary_key, 'id', None, properties)

        self._get_relationships()

    def _add_table_load_data(self, data):
        data = data if os.path.exists(data) else os.path.join(self.root_path, data)
        return pd.read_csv(data)

    def add_table(self, name, primary_key=None, fields=None, data=None, parent=None,
                  foreign_key=None):
        """Add a new table to the metadata.

        First, validate that the ``name`` table already exists.

        When ``fields`` is a ``dict``, ignore ``data`` and validate those fields.
        When ``fields`` is a ``list`` and ``data`` is ``None``, raise a ``ValueError``.
        When ``fields`` is a ``list`` and ``data`` is not ``None``, analyze data columns in list.
        When ``fields`` is ``None`` and ``data`` is not ``None``, analyze all columns.

        Use the ``fields`` to create the table and add primary key or relationship if needed.

        Args:
            name (str):
                Table name to be created, it must not exists.
            primary_key (str):
                Field name to add as primary key, it must not exists. Defaults to ``None``.
            fields (dict or list):
                If it's a ``dict``, data is ignored.
                If it's a ``list``, indicate which columns will be analized.
                Defaults to ``None``.
            data (str or pandas.DataFrame):
                Table to be analyzed or path to the csv file.
                If it's a relative path, use ``root_path`` to find the file.
                Only used if fields is a ``list`` or ``None``.
                Defaults to ``None``.
            parent (str):
                Table name to refere a foreign key field. Defaults to ``None``.
            foreign_key (str):
                Foreing key field name to ``parent`` table primary key. Defaults to ``None``.

        Raises:
            ValueError:
                A ``ValueError`` is raised when the table ``name`` already exists
                or ``fields`` is a ``list`` and ``data`` is ``None``
        """
        if name in self.get_table_names():
            raise ValueError('Table "{}" already exists.'.format(name))

        if isinstance(fields, dict):
            for field_key, field_value in fields.items():
                field_value['name'] = field_key
                self._validate_field(field_value)

        if isinstance(fields, list) and data is None:
            raise ValueError()

        if isinstance(fields, list) and data is not None:
            if isinstance(data, str):
                data = self._add_table_load_data(data)

            fields = Metadata._analyze(data, columns=fields)

        if not fields and data is not None:
            if isinstance(data, str):
                data = self._add_table_load_data(data)

            fields = Metadata._analyze(data)

        table = {'name': name, 'fields': fields or dict()}
        self._metadata['tables'][name] = table

        if primary_key:
            self.add_primary_key(name, primary_key)
            # table['primary_key'] = primary_key

        # Add relationship
        if parent:
            self.add_relationship(name, parent, foreign_key)

    def to_dict(self):
        return copy.deepcopy(self._metadata)

    def to_json(self):
        return json.dumps(self._metadata, indent=4)
