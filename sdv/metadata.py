import copy
import json
import logging
import os
from collections import defaultdict

import graphviz
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
            datetime_format = field.get('format')
            data[name] = pd.to_datetime(data[name], format=datetime_format, exact=False)
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


class MetadataError(Exception):
    pass


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

    def _analyze_relationships(self):
        """Extract information about child-parent relationships.

        Creates the following attributes:
            * ``_child_map``: set of child tables that each table has.
            * ``_parent_map``: set ot parents that each table has.
        """
        self._child_map = defaultdict(set)
        self._parent_map = defaultdict(set)

        for table, table_meta in self._metadata['tables'].items():
            if table_meta.get('use', True):
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
        if isinstance(tables, dict):
            new_metadata['tables'] = {
                table: meta
                for table, meta in tables.items()
                if meta.pop('use', True)
            }
            return new_metadata

        new_tables = dict()
        for table in tables:
            if table.pop('use', True):
                new_tables[table.pop('name')] = table

                fields = table['fields']
                new_fields = dict()
                for field in fields:
                    new_fields[field.pop('name')] = field

                table['fields'] = new_fields

        new_metadata['tables'] = new_tables

        return new_metadata

    def __init__(self, metadata=None, root_path=None):
        if isinstance(metadata, str):
            self.root_path = root_path or os.path.dirname(metadata)
            with open(metadata) as metadata_file:
                metadata = json.load(metadata_file)
        else:
            self.root_path = root_path or '.'

        if metadata is not None:
            self._metadata = self._dict_metadata(metadata)
        else:
            self._metadata = {'tables': {}}

        self._hyper_transformers = dict()
        self._analyze_relationships()

    def get_children(self, table_name):
        """Get tables for which the given table is parent.

        Args:
            table_name (str):
                Name of the table from which to get the children.

        Returns:
            set:
                Set of children for the given table.
        """
        return self._child_map[table_name]

    def get_parents(self, table_name):
        """Get tables for with the given table is child.

        Args:
            table_name (str):
                Name of the table from which to get the parents.

        Returns:
            set:
                Set of parents for the given table.
        """
        return self._parent_map[table_name]

    def get_table_meta(self, table_name):
        """Get the metadata dict for a table.

        Args:
            table_name (str):
                Name of table to get data for.

        Returns:
            dict:
                table metadata

        Raises:
            ValueError:
                If table does not exist in this metadata.
        """
        table = self._metadata['tables'].get(table_name)
        if table is None:
            raise ValueError('Table "{}" does not exist'.format(table_name))

        return copy.deepcopy(table)

    def get_tables(self):
        """Get the list of table names.

        Returns:
            list:
                table names.
        """
        return list(self._metadata['tables'].keys())

    def get_fields(self, table_name):
        """Get table fields metadata.

        Args:
            table_name (str):
                Name of the table to get the fields from.

        Returns:
            dict:
                Mapping of field names and their metadata dicts.

        Raises:
            ValueError:
                If table does not exist in this metadata.
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

        Raises:
            ValueError:
                If table does not exist in this metadata.
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

        for name, field in self.get_fields(child).items():
            ref = field.get('ref')
            if ref and ref['field'] == primary:
                return name

        raise ValueError('{} is not parent of {}'.format(parent, child))

    def load_table(self, table_name):
        """Load table data.

        Args:
            table_name (str):
                Name of the table to load.

        Returns:
            pandas.DataFrame:
                DataFrame with the contents of the table.

        Raises:
            ValueError:
                If table does not exist in this metadata.
        """
        LOGGER.info('Loading table %s', table_name)
        table_meta = self.get_table_meta(table_name)
        return _load_csv(self.root_path, table_meta)

    def load_tables(self, tables=None):
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
            for table_name in tables or self.get_tables()
        }

    def get_dtypes(self, table_name, ids=False):
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
                If a field has an invalid type or subtype or if the table does not
                exist in this metadata.
        """
        dtypes = dict()
        table_meta = self.get_table_meta(table_name)
        for name, field in table_meta['fields'].items():
            field_type = field['type']
            field_subtype = field.get('subtype')
            dtype = self._DTYPES.get((field_type, field_subtype))
            if not dtype:
                raise MetadataError(
                    'Invalid type and subtype combination for field {}: ({}, {})'.format(
                        name, field_type, field_subtype)
                )

            if ids and field_type == 'id':
                if (name != table_meta.get('primary_key')) and not field.get('ref'):
                    raise MetadataError(
                        'id field `{}` is neither a primary or a foreign key'.format(name))

            if ids or (field_type != 'id'):
                dtypes[name] = dtype

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
        dtypes = self.get_dtypes(table_name)
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

        for name, dtype in self.get_dtypes(table_name, ids=True).items():
            reversed_data[name] = reversed_data[name].dropna().astype(dtype)

        return reversed_data

    # ################### #
    # Metadata Validation #
    # ################### #

    def _validate_table(self, table_name, table_meta, table_data=None):
        """Validate table metadata.

        Validate the type and subtype combination for each field in ``table_meta``.
        If a field has type ``id``, validate that it either is the ``primary_key`` or
        has a ``ref`` entry.

        If the table has ``primary_key``, make sure that the corresponding field exists
        and its type is ``id``.

        If ``table_data`` is provided, also check that the list of columns corresponds
        to the ones indicated in the metadata and that all the dtypes are valid.

        Args:
            table_name (str):
                Name of the table to validate.
            table_meta (dict):
                Metadata of the table to validate.
            table_data (pandas.DataFrame):
                If provided, make sure that the data matches the one described
                on the metadata.

        Raises:
            MetadataError:
                If there is any error in the metadata or the data does not
                match the metadata description.
        """
        dtypes = self.get_dtypes(table_name, ids=True)

        # Primary key field exists and its type is 'id'
        primary_key = table_meta.get('primary_key')
        if primary_key:
            pk_field = table_meta['fields'].get(primary_key)

            if not pk_field:
                raise MetadataError('Primary key is not an existing field.')

            if pk_field['type'] != 'id':
                raise MetadataError('Primary key is not of type `id`.')

        if table_data is not None:
            for column in table_data:
                try:
                    dtype = dtypes.pop(column)
                    table_data[column].dropna().astype(dtype)
                except KeyError:
                    message = 'Unexpected column in table `{}`: `{}`'.format(table_name, column)
                    raise MetadataError(message) from None
                except ValueError as ve:
                    message = 'Invalid values found in column `{}` of table `{}`: `{}`'.format(
                        column, table_name, ve)
                    raise MetadataError(message) from None

            # assert all dtypes are in data
            if dtypes:
                raise MetadataError(
                    'Missing columns on table {}: {}.'.format(table_name, list(dtypes.keys()))
                )

    def _validate_circular_relationships(self, parent, children=None):
        """Validate that there is no circular relatioship in the metadata."""
        if children is None:
            children = self.get_children(parent)

        if parent in children:
            raise MetadataError('Circular relationship found for table "{}"'.format(parent))

        for child in children:
            self._validate_circular_relationships(parent, self.get_children(child))

    def _validate_parents(self, table_name):
        """Make sure that the table has only one parent."""
        if len(self.get_parents(table_name)) > 1:
            raise MetadataError('Table {} has more than one parent.'.format(table_name))

    def validate(self, tables=None):
        """Validate this metadata.

        For each table from in metadata ``tables`` entry:
            * Validate the table metadata is correct.

        * If ``tables`` are provided or they have been loaded, check
          that all the metadata tables exists in the ``tables`` dictionary.
        * Validate the type/subtype combination for each field and
          if a field of type ``id`` exists it must be the ``primary_key``
          or must have a ``ref`` entry.
        * If ``primary_key`` entry exists, check that it's an existing
          field and its type is ``id``.
        * If ``tables`` are provided or they have been loaded, check
          all the data types for the table correspond to each column and
          all the data types exists on the table.
        * Validate that there is no circular relatioship in the metadata.
        * Check that all the tables have at most one parent.

        Args:
            tables (bool, dict):
                If a dict of table is passed, validate that the columns and
                dtypes match the metadata. If ``True`` is passed, load the
                tables from the Metadata instead. If ``None``, omit the data
                validation. Defaults to ``None``.
        """
        tables_meta = self._metadata.get('tables')
        if not tables_meta:
            raise MetadataError('"tables" entry not found in Metadata.')

        if tables and not isinstance(tables, dict):
            tables = self.load_tables()

        for table_name, table_meta in tables_meta.items():
            if tables:
                table = tables.get(table_name)
                if table is None:
                    raise MetadataError('Table `{}` not found in tables'.format(table_name))

            else:
                table = None

            self._validate_table(table_name, table_meta, table)
            self._validate_circular_relationships(table_name)
            self._validate_parents(table_name)

    def _check_field(self, table, field, exists=False):
        """Validate the existance of the table and existance (or not) of field."""
        table_fields = self.get_fields(table)
        if exists and (field not in table_fields):
            raise ValueError('Field "{}" does not exist in table "{}"'.format(field, table))

        if not exists and (field in table_fields):
            raise ValueError('Field "{}" already exists in table "{}"'.format(field, table))

    # ################# #
    # Metadata Creation #
    # ################# #

    def add_field(self, table, field, field_type, field_subtype=None, properties=None):
        """Add a new field to the indicated table.

        Args:
            table (str):
                Table name to add the new field, it must exist.
            field (str):
                Field name to be added, it must not exist.
            field_type (str):
                Data type of field to be added. Required.
            field_subtype (str):
                Data subtype of field to be added. Optional.
                Defaults to ``None``.
            properties (dict):
                Extra properties of field like: ref, format, min, max, etc. Optional.
                Defaults to ``None``.

        Raises:
            ValueError:
                If the table does not exist or it already contains the field.
        """
        self._check_field(table, field, exists=False)

        field_details = {
            'type': field_type
        }

        if field_subtype:
            field_details['subtype'] = field_subtype

        if properties:
            field_details.update(properties)

        self._metadata['tables'][table]['fields'][field] = field_details

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

    def set_primary_key(self, table, field):
        """Set the primary key field of the indicated table.

        The field must exist and either be an integer or categorical field.

        Args:
            table (str):
                Name of the table where the primary key will be set.
            field (str):
                Field to be used as the new primary key.

        Raises:
            ValueError:
                If the table or the field do not exist or if the field has an
                invalid type or subtype.
        """
        self._check_field(table, field, exists=True)

        field_meta = self.get_fields(table).get(field)
        field_subtype = self._get_key_subtype(field_meta)

        table_meta = self._metadata['tables'][table]
        table_meta['fields'][field] = {
            'type': 'id',
            'subtype': field_subtype
        }
        table_meta['primary_key'] = field

    def add_relationship(self, parent, child, foreign_key=None):
        """Add a new relationship between the parent and child tables.

        The relationship is created by adding a reference (``ref``) on the ``foreign_key``
        field of the ``child`` table pointing at the ``parent`` primary key.

        Args:
            parent (str):
                Name of the parent table.
            child (str):
                Name of the child table.
            foreign_key (str):
                Field in the child table through which the relationship is created.
                If ``None``, use the parent primary key name.

        Raises:
            ValueError:
                If any of the following happens:
                    * The parent table does not exist.
                    * The child table does not exist.
                    * The parent table does not have a primary key.
                    * The foreign_key field already exists in the child table.
                    * The child table already has a parent.
                    * The new relationship closes a relationship circle.
        """
        # Validate table and field names
        primary_key = self.get_primary_key(parent)
        if not primary_key:
            raise ValueError('Parent table "{}" does not have a primary key'.format(parent))

        if foreign_key is None:
            foreign_key = primary_key

        # Validate relationships
        if self.get_parents(child):
            raise ValueError('Table "{}" already has a parent'.format(child))

        grandchildren = self.get_children(child)
        if grandchildren:
            self._validate_circular_relationships(parent, grandchildren)

        # Copy primary key details over to the foreign key
        foreign_key_details = copy.deepcopy(self.get_fields(parent)[primary_key])
        foreign_key_details['ref'] = {
            'table': parent,
            'field': primary_key
        }

        # Make sure that key subtypes are the same
        foreign_meta = self.get_fields(child).get(foreign_key)
        if foreign_meta:
            foreign_subtype = self._get_key_subtype(foreign_meta)
            if foreign_subtype != foreign_key_details['subtype']:
                raise ValueError('Primary and Foreign key subtypes mismatch')

        self._metadata['tables'][child]['fields'][foreign_key] = foreign_key_details

        # Re-analyze the relationships
        self._analyze_relationships()

    def _get_field_details(self, data, fields):
        """Get or build all the fields metadata.

        Analyze a ``pandas.DataFrame`` to build a ``dict`` with the name of the column, and
        their data type and subtype. If ``columns`` are provided, only those columns will be
        analyzed.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
            fields (set):
                Set of field names or field specifications.

        Returns:
            dict:
                Dict of valid fields.

        Raises:
            TypeError:
                If a field specification is not a str or a dict.
            ValueError:
                If a column from the data analyzed is an unsupported data type or
        """
        fields_metadata = dict()
        for field in fields:
            dtype = data[field].dtype
            field_template = self._FIELD_TEMPLATES.get(dtype.kind)
            if not field_template:
                raise ValueError('Unsupported dtype {} in column {}'.format(dtype, field))

            field_details = copy.deepcopy(field_template)
            fields_metadata[field] = field_details

        return fields_metadata

    def add_table(self, name, data=None, fields=None, fields_metadata=None,
                  primary_key=None, parent=None, foreign_key=None):
        """Add a new table to this metadata.

        ``fields`` list can be a mixture of field names, which will be build automatically
        from the data, or dictionaries specifying the field details. If a field needs to be
        analyzed, data has to be also passed.

        If ``parent`` is given, a relationship will be established between this table
        and the specified parent.

        Args:
            name (str):
                Name of the new table.
            data (str or pandas.DataFrame):
                Table to be analyzed or path to the csv file.
                If it's a relative path, use ``root_path`` to find the file.
                Only used if fields is not ``None``.
                Defaults to ``None``.
            fields (list):
                List of field names to build. If ``None`` is given, all the fields
                found in the data will be used.
                Defaults to ``None``.
            fields_metadata (dict):
                Metadata to be used when creating fields. This will overwrite the
                metadata built from the fields found in data.
                Defaults to ``None``.
            primary_key (str):
                Field name to add as primary key, it must not exists. Defaults to ``None``.
            parent (str):
                Table name to refere a foreign key field. Defaults to ``None``.
            foreign_key (str):
                Foreing key field name to ``parent`` table primary key. Defaults to ``None``.

        Raises:
            ValueError:
                If the table ``name`` already exists or ``data`` is not passed and
                fields need to be built from it.
        """
        if name in self.get_tables():
            raise ValueError('Table "{}" already exists.'.format(name))

        path = None
        if data is not None:
            if isinstance(data, str):
                path = data
                if not os.path.isabs(data):
                    data = os.path.join(self.root_path, data)

                data = pd.read_csv(data)

            fields = set(fields or data.columns)
            if fields_metadata:
                fields = fields - set(fields_metadata.keys())
            else:
                fields_metadata = dict()

            fields_metadata.update(self._get_field_details(data, fields))

        elif fields_metadata is None:
            fields_metadata = dict()

        table_metadata = {'fields': fields_metadata}
        if path:
            table_metadata['path'] = path

        self._metadata['tables'][name] = table_metadata

        try:
            if primary_key:
                self.set_primary_key(name, primary_key)

            if parent:
                self.add_relationship(parent, name, foreign_key)

        except ValueError:
            # Cleanup
            del self._metadata['tables'][name]
            raise

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

    @staticmethod
    def _get_graphviz_extension(path):
        if path:
            path_splitted = path.split('.')
            if len(path_splitted) == 1:
                raise ValueError('Path without graphviz extansion.')

            graphviz_extension = path_splitted[-1]

            if graphviz_extension not in graphviz.backend.FORMATS:
                raise ValueError('"{}" not a valid graphviz extension format.')

            return ''.join(path_splitted[:-1]), graphviz_extension

        return None, None

    def visualize(self, path=None):
        """Plot metadata usign graphviz.

        Try to generate a plot using graphviz.
        If a ``path`` is provided save the output into a file.

        Args:
            path (str):
                Output file path to save the plot, it requires a graphviz
                supported extension. If ``None`` do not save the plot.
                Defaults to ``None``.
        """
        try:
            graphviz.Digraph().pipe()
        except graphviz.backend.ExecutableNotFound:
            raise SystemError(
                'Missing graphviz executable. Please take a look at: '
                'https://graphviz.gitlab.io/download/'
            )

        filename, graphviz_extension = self._get_graphviz_extension(path)
        plot = graphviz.Digraph(
            'Metadata',
            format=graphviz_extension,
            graph_attr={"rankdir": "BT"},
            node_attr={"shape": "Mrecord"},
        )

        for table in self.get_tables():
            fields = r'\l'.join([
                '{} : {}'.format(name, value['type'])
                for name, value in self.get_fields(table).items()
            ])
            title = r'{%s|%s\l}' % (table, fields)
            plot.node(table, label=title)

        for table in self.get_tables():
            for parent in list(self.get_parents(table)):
                plot.edge(
                    table,
                    parent,
                    label='{}.{} -> {}.{}'.format(
                        table, self.get_foreign_key(parent, table),
                        parent, self.get_primary_key(parent)
                    )
                )

        if filename:
            plot.render(filename=filename, cleanup=True, format=graphviz_extension)

        return plot

    def __str__(self):
        tables = self.get_tables()
        relationships = [
            '    {}.{} -> {}.{}'.format(
                table, self.get_foreign_key(parent, table),
                parent, self.get_primary_key(parent)
            )
            for table in tables
            for parent in list(self.get_parents(table))
        ]

        return (
            "Metadata\n"
            "  root_path: {}\n"
            "  tables: {}\n"
            "  relationships:\n"
            "{}"
        ).format(
            os.path.abspath(self.root_path),
            tables,
            '\n'.join(relationships)
        )
