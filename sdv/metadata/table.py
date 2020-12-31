"""Metadata for a single table."""

import copy
import json
import logging

import numpy as np
import pandas as pd
import rdt
from faker import Faker
from xeger import Xeger

from sdv.constraints.base import Constraint
from sdv.metadata.errors import MetadataError, MetadataNotFittedError

LOGGER = logging.getLogger(__name__)


class Table:
    """Table Metadata.

    The Metadata class provides a unified layer of abstraction over the metadata
    of a single Table, which includes all the necessary details to handle the
    table of this data, including the data types, the fields with pii information
    and the constraints that affect this data.

    Args:
        name (str):
            Name of this table. Optional.
        field_names (list[str]):
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types (dict[str, dict]):
            Dictinary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        field_transformers (dict[str, str]):
            Dictinary specifying which transformers to use for each field.
            Available transformers are:

                * ``integer``: Uses a ``NumericalTransformer`` of dtype ``int``.
                * ``float``: Uses a ``NumericalTransformer`` of dtype ``float``.
                * ``categorical``: Uses a ``CategoricalTransformer`` without gaussian noise.
                * ``categorical_fuzzy``: Uses a ``CategoricalTransformer`` adding gaussian noise.
                * ``one_hot_encoding``: Uses a ``OneHotEncodingTransformer``.
                * ``label_encoding``: Uses a ``LabelEncodingTransformer``.
                * ``boolean``: Uses a ``BooleanTransformer``.
                * ``datetime``: Uses a ``DatetimeTransformer``.

        anonymize_fields (dict[str, str]):
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key (str):
            Name of the field which is the primary key of the table.
        constraints (list[Constraint, dict]):
            List of Constraint objects or dicts.
        dtype_transformers (dict):
            Dictionary of transformer templates to be used for the
            different data types. The keys must be any of the `dtype.kind`
            values, `i`, `f`, `O`, `b` or `M`, and the values must be
            either RDT Transformer classes or RDT Transformer instances.
        model_kwargs (dict):
            Dictionary specifiying the kwargs that need to be used in
            each tabular model when working on this table. This dictionary
            contains as keys the name of the TabularModel class and as
            values a dictionary containing the keyword arguments to use.
            This argument exists mostly to ensure that the models are
            fitted using the same arguments when the same Table is used
            to fit different model instances on different slices of the
            same table.
        sequence_index (str):
            Name of the column that acts as the order index of each
            sequence. The sequence index column can be of any type that can
            be sorted, such as integer values or datetimes.
        entity_columns (list[str]):
            Names of the columns which identify different time series
            sequences. These will be used to group the data in separated
            training examples.
        context_columns (list[str]):
            The columns in the dataframe which are constant within each
            group/entity. These columns will be provided at sampling time
            (i.e. the samples will be conditioned on the context variables).
    """

    _hyper_transformer = None
    _fakers = None
    _constraint_instances = None
    _fields_metadata = None
    fitted = False

    _ANONYMIZATION_MAPPINGS = dict()
    _TRANSFORMER_TEMPLATES = {
        'integer': rdt.transformers.NumericalTransformer(dtype=int),
        'float': rdt.transformers.NumericalTransformer(dtype=float),
        'categorical': rdt.transformers.CategoricalTransformer,
        'categorical_fuzzy': rdt.transformers.CategoricalTransformer(fuzzy=True),
        'one_hot_encoding': rdt.transformers.OneHotEncodingTransformer,
        'label_encoding': rdt.transformers.LabelEncodingTransformer,
        'boolean': rdt.transformers.BooleanTransformer,
        'datetime': rdt.transformers.DatetimeTransformer(strip_constant=True),
    }
    _DTYPE_TRANSFORMERS = {
        'i': 'integer',
        'f': 'float',
        'O': 'one_hot_encoding',
        'b': 'boolean',
        'M': 'datetime',
    }
    _DTYPES_TO_TYPES = {
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
    _TYPES_TO_DTYPES = {
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

    def __init__(self, name=None, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None,
                 dtype_transformers=None, model_kwargs=None, sequence_index=None,
                 entity_columns=None, context_columns=None):
        self.name = name
        self._field_names = field_names
        self._field_types = field_types or {}
        self._field_transformers = field_transformers or {}
        self._anonymize_fields = anonymize_fields or {}
        self._model_kwargs = model_kwargs or {}

        self._primary_key = primary_key
        self._sequence_index = sequence_index
        self._entity_columns = entity_columns or []
        self._context_columns = context_columns or []
        self._constraints = constraints or []
        self._dtype_transformers = self._DTYPE_TRANSFORMERS.copy()
        if dtype_transformers:
            self._dtype_transformers.update(dtype_transformers)

    def __repr__(self):
        return 'Table(name={}, field_names={})'.format(self.name, self._field_names)

    def get_model_kwargs(self, model_name):
        """Return the required model kwargs for the indicated model.

        Args:
            model_name (str):
                Qualified Name of the model for which model kwargs
                are needed.

        Returns:
            dict:
                Keyword arguments to use on the indicated model.
        """
        return copy.deepcopy(self._model_kwargs.get(model_name))

    def set_model_kwargs(self, model_name, model_kwargs):
        """Set the model kwargs used for the indicated model."""
        self._model_kwargs[model_name] = model_kwargs

    def _get_field_dtype(self, field_name, field_metadata):
        field_type = field_metadata['type']
        field_subtype = field_metadata.get('subtype')
        dtype = self._TYPES_TO_DTYPES.get((field_type, field_subtype))
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
        return copy.deepcopy(self._fields_metadata)

    def get_dtypes(self, ids=False):
        """Get a ``dict`` with the ``dtypes`` for each field of the table.

        Args:
            ids (bool):
                Whether or not to include the id fields. Defaults to ``False``.

        Returns:
            dict:
                Dictionary that contains the field names and data types.
        """
        dtypes = dict()
        for name, field_meta in self._fields_metadata.items():
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
        fields_metadata = dict()
        for field_name in self._field_names:
            if field_name not in data:
                raise ValueError('Field {} not found in given data'.format(field_name))

            field_meta = self._field_types.get(field_name)
            if field_meta:
                dtype = self._get_field_dtype(field_name, field_meta)
            else:
                dtype = data[field_name].dtype
                field_template = self._DTYPES_TO_TYPES.get(dtype.kind)
                if field_template is None:
                    msg = 'Unsupported dtype {} in column {}'.format(dtype, field_name)
                    raise ValueError(msg)

                field_meta = copy.deepcopy(field_template)

            field_transformer = self._field_transformers.get(field_name)
            if field_transformer:
                field_meta['transformer'] = field_transformer
            else:
                field_meta['transformer'] = self._dtype_transformers.get(np.dtype(dtype).kind)

            anonymize_category = self._anonymize_fields.get(field_name)
            if anonymize_category:
                field_meta['pii'] = True
                field_meta['pii_category'] = anonymize_category

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
        transformers = dict()
        for name, dtype in dtypes.items():
            field_metadata = self._fields_metadata.get(name, {})
            transformer_template = field_metadata.get('transformer')
            if transformer_template is None:
                transformer_template = self._dtype_transformers[np.dtype(dtype).kind]
                if transformer_template is None:
                    # Skip this dtype
                    continue

                field_metadata['transformer'] = transformer_template

            if isinstance(transformer_template, str):
                transformer_template = self._TRANSFORMER_TEMPLATES[transformer_template]

            if isinstance(transformer_template, type):
                transformer = transformer_template()
            else:
                transformer = copy.deepcopy(transformer_template)

            LOGGER.debug('Loading transformer %s for field %s',
                         transformer.__class__.__name__, name)
            transformers[name] = transformer

        return transformers

    def _fit_transform_constraints(self, data):
        for idx, constraint in enumerate(self._constraints):
            if isinstance(constraint, type):
                constraint = constraint().to_dict()
            elif isinstance(constraint, Constraint):
                constraint = constraint.to_dict()

            constraint = Constraint.from_dict(constraint)
            self._constraints[idx] = constraint

            data = constraint.fit_transform(data)

        return data

    def _fit_hyper_transformer(self, data, extra_columns):
        """Create and return a new ``rdt.HyperTransformer`` instance.

        First get the ``dtypes`` and then use them to build a transformer dictionary
        to be used by the ``HyperTransformer``.

        Args:
            data (pandas.DataFrame):
                Data to transform.
            extra_columns (set):
                Names of columns that are not in the metadata but that should also
                be transformed. In most cases, these are the fields that were added
                by previous transformations which the data underwent.

        Returns:
            rdt.HyperTransformer
        """
        meta_dtypes = self.get_dtypes(ids=False)
        dtypes = {}
        for column in data.columns:
            if column in meta_dtypes:
                dtypes[column] = meta_dtypes[column]
            elif column in extra_columns:
                dtypes[column] = data[column].dtype.kind

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

    def set_primary_key(self, primary_key):
        """Set the primary key of this table.

        The field must exist and either be an integer or categorical field.

        Args:
            primary_key (str or list):
                Name of the field(s) to be used as the new primary key.

        Raises:
            ValueError:
                If the table or the field do not exist or if the field has an
                invalid type or subtype.
        """
        if primary_key is not None:
            fields = primary_key if isinstance(primary_key, list) else [primary_key]
            for field_name in fields:
                if field_name not in self._fields_metadata:
                    raise ValueError('Field "{}" does not exist in this table'.format(field_name))

                field_metadata = self._fields_metadata[field_name]
                if field_metadata['type'] != 'id':
                    field_subtype = self._get_key_subtype(field_metadata)

                    field_metadata.update({
                        'type': 'id',
                        'subtype': field_subtype
                    })

        self._primary_key = primary_key

    def _make_anonymization_mappings(self, data):
        mappings = {}
        for name, field_metadata in self._fields_metadata.items():
            if field_metadata['type'] != 'id' and field_metadata.get('pii'):
                faker = self._get_faker(field_metadata['pii_category'])

                uniques = data[name].unique()
                fake_values = [faker() for _ in range(len(uniques))]
                mappings[name] = dict(zip(uniques, fake_values))

        self._ANONYMIZATION_MAPPINGS[id(self)] = mappings

        return mappings

    def _anonymize(self, data):
        anonymization_mappings = self._ANONYMIZATION_MAPPINGS.get(id(self))
        if anonymization_mappings:
            data = data.copy()
            for name, mapping in anonymization_mappings.items():
                data[name] = data[name].map(mapping)

        return data

    def fit(self, data):
        """Fit this metadata to the given data.

        Args:
            data (pandas.DataFrame):
                Table to be analyzed.
        """
        LOGGER.info('Fitting table %s metadata', self.name)
        if not self._field_names:
            self._field_names = list(data.columns)
        elif isinstance(self._field_names, set):
            self._field_names = [field for field in data.columns if field in self._field_names]

        self._dtypes = data[self._field_names].dtypes

        if not self._fields_metadata:
            self._fields_metadata = self._build_fields_metadata(data)

        # Re-set the primary key to validate its name and type
        self.set_primary_key(self._primary_key)

        self._make_anonymization_mappings(data)
        LOGGER.info('Anonymizing table %s', self.name)
        data = self._anonymize(data)

        LOGGER.info('Fitting constraints for table %s', self.name)
        constrained = self._fit_transform_constraints(data)
        extra_columns = set(constrained.columns) - set(data.columns)

        LOGGER.info('Fitting HyperTransformer for table %s', self.name)
        self._fit_hyper_transformer(constrained, extra_columns)
        self.fitted = True

    def transform(self, data):
        """Transform the given data.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        if not self.fitted:
            raise MetadataNotFittedError()

        fields = self.get_dtypes(ids=False)
        LOGGER.debug('Anonymizing table %s', self.name)
        data = self._anonymize(data[fields])

        LOGGER.debug('Transforming constraints for table %s', self.name)
        for constraint in self._constraints:
            data = constraint.transform(data)

        LOGGER.debug('Transforming table %s', self.name)
        return self._hyper_transformer.transform(data)

    def _make_ids(self, name, field_metadata, length):
        field_subtype = field_metadata.get('subtype', 'integer')
        if field_subtype == 'string':
            regex = field_metadata.get('regex', '[a-z][A-Z]+')
            xeger = Xeger(limit=10)
            values = {xeger.xeger(regex) for _ in range(length)}
            for _ in range(10):
                done = len(values)
                if done >= length:
                    break

                remaining = length - done
                generate = int(round((remaining / done) * length))
                values.update({xeger.xeger(regex) for _ in range(generate)})

            else:
                msg = 'Unable to generate {} unique values for field {} regex "{}"'.format(
                    length, name, regex
                )
                raise ValueError(msg)

            return pd.Series(list(values)[:length])
        else:
            return pd.Series(np.arange(length))

    def reverse_transform(self, data):
        """Reverse the transformed data to the original format.

        Args:
            data (pandas.DataFrame):
                Data to be reverse transformed.

        Returns:
            pandas.DataFrame
        """
        if not self.fitted:
            raise MetadataNotFittedError()

        reversed_data = self._hyper_transformer.reverse_transform(data)

        for constraint in self._constraints:
            reversed_data = constraint.reverse_transform(reversed_data)

        for name, field_metadata in self._fields_metadata.items():
            field_type = field_metadata['type']
            if field_type == 'id' and name not in reversed_data:
                field_data = self._make_ids(name, field_metadata, len(reversed_data))
            elif field_metadata.get('pii', False):
                faker = self._get_faker(field_metadata['pii_category'])
                field_data = pd.Series([faker() for _ in range(len(reversed_data))])
            else:
                field_data = reversed_data[name]

            reversed_data[name] = field_data[field_data.notnull()].astype(self._dtypes[name])

        return reversed_data[self._field_names]

    def filter_valid(self, data):
        """Filter the data using the constraints and return only the valid rows.

        Args:
            data (pandas.DataFrame):
                Table data.

        Returns:
            pandas.DataFrame:
                Table containing only the valid rows.
        """
        for constraint in self._constraints:
            data = constraint.filter_valid(data)

        return data

    # ###################### #
    # Metadata Serialization #
    # ###################### #

    def to_dict(self):
        """Get a dict representation of this metadata.

        Returns:
            dict:
                dict representation of this metadata.
        """
        return {
            'fields': copy.deepcopy(self._fields_metadata),
            'constraints': [
                constraint if isinstance(constraint, dict) else constraint.to_dict()
                for constraint in self._constraints
            ],
            'model_kwargs': copy.deepcopy(self._model_kwargs),
            'name': self.name,
            'primary_key': self._primary_key,
            'sequence_index': self._sequence_index,
            'entity_columns': self._entity_columns,
            'context_columns': self._context_columns,
        }

    def to_json(self, path):
        """Dump this metadata into a JSON file.

        Args:
            path (str):
                Path of the JSON file where this metadata will be stored.
        """
        with open(path, 'w') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

    @classmethod
    def from_dict(cls, metadata_dict, dtype_transformers=None):
        """Load a Table from a metadata dict.

        Args:
            metadata_dict (dict):
                Dict metadata to load.
            dtype_transformers (dict):
                If passed, set the dtype_transformers on the new instance.
        """
        metadata_dict = copy.deepcopy(metadata_dict)
        fields = metadata_dict['fields'] or {}
        instance = cls(
            name=metadata_dict.get('name'),
            field_names=set(fields.keys()),
            field_types=fields,
            constraints=metadata_dict.get('constraints') or [],
            model_kwargs=metadata_dict.get('model_kwargs') or {},
            primary_key=metadata_dict.get('primary_key'),
            sequence_index=metadata_dict.get('sequence_index'),
            entity_columns=metadata_dict.get('entity_columns') or [],
            context_columns=metadata_dict.get('context_columns') or [],
            dtype_transformers=dtype_transformers,
        )
        instance._fields_metadata = fields
        return instance

    @classmethod
    def from_json(cls, path):
        """Load a Table from a JSON.

        Args:
            path (str):
                Path of the JSON file to load
        """
        with open(path, 'r') as in_file:
            return cls.from_dict(json.load(in_file))
