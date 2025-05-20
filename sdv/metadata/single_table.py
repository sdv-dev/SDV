"""Single Table Metadata."""

import json
import logging
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import pandas as pd
from rdt.transformers._validators import AddressValidator, GPSValidator
from rdt.transformers.pii.anonymization import SDTYPE_ANONYMIZERS, is_faker_function

from sdv._utils import (
    _cast_to_iterable,
    _format_invalid_values_string,
    _get_datetime_format,
    _is_boolean_type,
    _is_datetime_type,
    _is_numerical_type,
    _load_data_from_csv,
    _validate_datetime_format,
    get_possible_chars,
)
from sdv.errors import InvalidDataError
from sdv.logging import get_sdv_logger
from sdv.metadata.errors import InvalidMetadataError
from sdv.metadata.metadata_upgrader import convert_metadata
from sdv.metadata.utils import _validate_file_mode, read_json, validate_file_does_not_exist
from sdv.metadata.visualization import (
    create_columns_node,
    create_summarized_columns_node,
    visualize_graph,
)

LOGGER = logging.getLogger(__name__)
SINGLETABLEMETADATA_LOGGER = get_sdv_logger('SingleTableMetadata')
INT_REGEX_ZERO_ERROR_MESSAGE = (
    'is stored as an int but the Regex allows it to start with "0". Please remove the Regex '
    'or update it to correspond to valid ints.'
)


class SingleTableMetadata:
    """Single Table Metadata class."""

    _SDTYPE_KWARGS = {
        'numerical': frozenset(['computer_representation']),
        'datetime': frozenset(['datetime_format']),
        'categorical': frozenset(['order', 'order_by']),
        'boolean': frozenset([]),
        'id': frozenset(['regex_format']),
        'unknown': frozenset(['pii']),
    }

    _DTYPES_TO_SDTYPES = {
        'b': 'categorical',
        'M': 'datetime',
    }

    _NUMERICAL_REPRESENTATIONS = frozenset([
        'Float32',
        'Float64',
        'Float',
        'Int64',
        'Int32',
        'Int16',
        'Int8',
        'UInt64',
        'UInt32',
        'UInt16',
        'UInt8',
    ])
    _KEYS = frozenset([
        'columns',
        'primary_key',
        'alternate_keys',
        'sequence_key',
        'sequence_index',
        'column_relationships',
        'METADATA_SPEC_VERSION',
    ])

    _REFERENCE_TO_SDTYPE = {
        'phonenumber': 'phone_number',
        'email': 'email',
        'ssn': 'ssn',
        'firstname': 'first_name',
        'lastname': 'last_name',
        'countrycode': 'country_code',
        'administativeunit': 'administrative_unit',
        'state': 'administrative_unit',
        'province': 'administrative_unit',
        'stateabbr': 'state_abbr',
        'city': 'city',
        'postalcode': 'postcode',
        'zipcode': 'postcode',
        'postcode': 'postcode',
        'streetaddress': 'street_address',
        'line1': 'street_address',
        'secondaryaddress': 'secondary_address',
        'line2': 'secondary_address',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'ipv4': 'ipv4_address',
        'ipv4address': 'ipv4_address',
        'ipv6': 'ipv6_address',
        'ipv6address': 'ipv6_address',
        'ipaddress': 'ipv6_address',
        'macaddress': 'mac_address',
        'useragent': 'user_agent_string',
        'useragentstring': 'user_agent_string',
        'iban': 'iban',
        'swift': 'swift11',
        'swift11': 'swift11',
        'swift8': 'swift8',
        'creditcardnumber': 'credit_card_number',
        'vin': 'vin',
        'licenseplate': 'license_plate',
        'license': 'license_plate',
    }

    _SDTYPES_WITHOUT_SUBSTRINGS = {
        reference: sdtype
        for reference, sdtype in _REFERENCE_TO_SDTYPE.items()
        if sdtype not in {'ssn', 'administrative_unit', 'city', 'vin'}
    }

    _SDTYPES_WITH_SUBSTRINGS = dict(
        set(_REFERENCE_TO_SDTYPE.items()) - set(_SDTYPES_WITHOUT_SUBSTRINGS.items())
    )

    _COLUMN_RELATIONSHIP_TYPES = {
        'address': AddressValidator.validate,
        'gps': GPSValidator.validate,
    }

    METADATA_SPEC_VERSION = 'SINGLE_TABLE_V1'
    _DEFAULT_SDTYPES = list(_SDTYPE_KWARGS) + list(SDTYPE_ANONYMIZERS)
    _MIN_ROWS_FOR_PREDICTION = 5
    _NUMERICAL_DTYPES = frozenset(['i', 'f', 'u'])

    def _validate_numerical(self, column_name, **kwargs):
        representation = kwargs.get('computer_representation')
        if representation and representation not in self._NUMERICAL_REPRESENTATIONS:
            raise InvalidMetadataError(
                f"Invalid value for 'computer_representation' '{representation}'"
                f" for column '{column_name}'."
            )

    @staticmethod
    def _validate_datetime(column_name, **kwargs):
        datetime_format = kwargs.get('datetime_format')
        if datetime_format is not None:
            try:
                formated_date = datetime.now().strftime(datetime_format)
            except Exception as exception:
                raise InvalidMetadataError(
                    f"Invalid datetime format string '{datetime_format}' "
                    f"for datetime column '{column_name}'."
                ) from exception

            matches = re.findall('(%.)|(%)', formated_date)
            if matches:
                raise InvalidMetadataError(
                    f"Invalid datetime format string '{datetime_format}' "
                    f"for datetime column '{column_name}'."
                )

    @staticmethod
    def _validate_categorical(column_name, **kwargs):
        order = kwargs.get('order')
        order_by = kwargs.get('order_by')
        if order is not None and order_by is not None:
            raise InvalidMetadataError(
                f"Categorical column '{column_name}' has both an 'order' and 'order_by' "
                'attribute. Only 1 is allowed.'
            )
        if order_by is not None and order_by not in ('numerical_value', 'alphabetical'):
            raise InvalidMetadataError(
                f"Unknown ordering method '{order_by}' provided for categorical column "
                f"'{column_name}'. Ordering method must be 'numerical_value' or 'alphabetical'."
            )
        if (isinstance(order, list) and not len(order)) or (
            not isinstance(order, list) and order is not None
        ):
            raise InvalidMetadataError(
                f"Invalid order value provided for categorical column '{column_name}'. "
                "The 'order' must be a list with 1 or more elements."
            )

    @staticmethod
    def _validate_id(column_name, **kwargs):
        regex = kwargs.get('regex_format', '')
        try:
            re.compile(regex)
        except Exception as exception:
            raise InvalidMetadataError(
                f"Invalid regex format string '{regex}' for id column '{column_name}'."
            ) from exception

    @staticmethod
    def _validate_pii(column_name, **kwargs):
        pii_value = kwargs['pii']
        if not isinstance(pii_value, bool):
            raise InvalidMetadataError(
                f"Parameter 'pii' is set to an invalid attribute ('{pii_value}') for column "
                f"'{column_name}'. Expected a value of True or False."
            )

    def __init__(self):
        self.columns = {}
        self.primary_key = None
        self.alternate_keys = []
        self.sequence_key = None
        self.sequence_index = None
        self.column_relationships = []
        self._version = self.METADATA_SPEC_VERSION
        self._updated = False

    def _get_unexpected_kwargs(self, sdtype, **kwargs):
        expected_kwargs = self._SDTYPE_KWARGS.get(sdtype, ['pii'])
        unexpected_kwargs = set(kwargs) - set(expected_kwargs)
        if unexpected_kwargs:
            unexpected_kwargs = sorted(unexpected_kwargs)
            unexpected_kwargs = ', '.join(unexpected_kwargs)

        return unexpected_kwargs

    def _validate_unexpected_kwargs(self, column_name, sdtype, **kwargs):
        unexpected_kwargs = self._get_unexpected_kwargs(sdtype, **kwargs)
        if unexpected_kwargs:
            raise InvalidMetadataError(
                f"Invalid values '({unexpected_kwargs})' for {sdtype} column '{column_name}'."
            )

    def _validate_sdtype(self, sdtype):
        if not isinstance(sdtype, str):
            raise InvalidMetadataError(
                f'Invalid sdtype: {sdtype} is not a string. Please use one of the '
                'supported SDV sdtypes.'
            )

        if sdtype in self._DEFAULT_SDTYPES:
            return

        if not is_faker_function(sdtype):
            raise InvalidMetadataError(
                f"Invalid sdtype: '{sdtype}' is not recognized. Please use one of the "
                'supported SDV sdtypes.'
            )

    def _validate_column_args(self, column_name, sdtype, **kwargs):
        self._validate_sdtype(sdtype)
        self._validate_unexpected_kwargs(column_name, sdtype, **kwargs)
        if sdtype == 'categorical':
            self._validate_categorical(column_name, **kwargs)
        elif sdtype == 'numerical':
            self._validate_numerical(column_name, **kwargs)
        elif sdtype == 'datetime':
            self._validate_datetime(column_name, **kwargs)
        elif sdtype == 'id':
            self._validate_id(column_name, **kwargs)
        elif 'pii' in kwargs:
            self._validate_pii(column_name, **kwargs)

    def add_column(self, column_name, **kwargs):
        """Add a column to the ``SingleTableMetadata``.

        Args:
            column_name (str):
                The column name to be added.
            kwargs (type):
                Any additional key word arguments for the column, where ``sdtype`` is required.

        Raises:
            - ``InvalidMetadataError`` if the column already exists.
            - ``InvalidMetadataError`` if the ``kwargs`` do not contain ``sdtype``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              given ``sdtype``.
            - ``InvalidMetadataError`` if the ``pii`` value is not ``True`` or ``False`` when
               present.
        """
        if column_name in self.columns:
            raise InvalidMetadataError(
                f"Column name '{column_name}' already exists. Use 'update_column' "
                'to update an existing column.'
            )

        sdtype = kwargs.get('sdtype')
        if sdtype is None:
            raise InvalidMetadataError(f"Please provide a 'sdtype' for column '{column_name}'.")

        self._validate_column_args(column_name, **kwargs)
        column_kwargs = deepcopy(kwargs)
        if sdtype not in self._SDTYPE_KWARGS:
            pii = column_kwargs.get('pii', True)
            column_kwargs['pii'] = pii

        self._updated = True
        self.columns[column_name] = column_kwargs

    def _validate_column_exists(self, column_name):
        if column_name not in self.columns:
            raise InvalidMetadataError(
                f"Column name ('{column_name}') does not exist in the table. "
                "Use 'add_column' to add new column."
            )

    def _validate_update_column(self, column_name, **kwargs):
        self._validate_column_exists(column_name)
        sdtype = kwargs.get('sdtype', self.columns[column_name]['sdtype'])
        kwargs_without_sdtype = {key: value for key, value in kwargs.items() if key != 'sdtype'}
        self._validate_column_args(column_name, sdtype, **kwargs_without_sdtype)

    def update_column(self, column_name, **kwargs):
        """Update an existing column in the ``SingleTableMetadata``.

        Args:
            column_name (str):
                The column name to be updated.
            **kwargs (type):
                Any key word arguments that describe metadata for the column.

        Raises:
            - ``InvalidMetadataError`` if the column doesn't already exist in the
              ``SingleTableMetadata``.
            - ``InvalidMetadataError`` if the column has unexpected values or ``kwargs`` for the
              current
              ``sdtype``.
            - ``InvalidMetadataError`` if the ``pii`` value is not ``True`` or ``False`` when
               present.
        """
        self._validate_update_column(column_name, **kwargs)
        if 'sdtype' not in kwargs:
            kwargs['sdtype'] = self.columns[column_name]['sdtype']

        self.columns[column_name] = kwargs
        self._updated = True

    def update_columns(self, column_names, **kwargs):
        """Update multiple columns with the same metadata kwargs.

        Args:
            column_names (list[str]):
                A list of column names to be updated.
            **kwargs (type):
                Any key word arguments that describe metadata for the column.
        """
        errors = []
        has_sdtype_key = 'sdtype' in kwargs
        if has_sdtype_key:
            kwargs_without_sdtype = {key: value for key, value in kwargs.items() if key != 'sdtype'}
            unexpected_kwargs = self._get_unexpected_kwargs(
                kwargs['sdtype'], **kwargs_without_sdtype
            )
            if unexpected_kwargs:
                raise InvalidMetadataError(
                    f"Invalid values '({unexpected_kwargs})' for '{kwargs['sdtype']}' sdtype."
                )

        for column_name in column_names:
            try:
                self._validate_update_column(column_name, **kwargs)
            except InvalidMetadataError as e:
                errors.append(e)

        if errors:
            raise InvalidMetadataError(
                'The following errors were found when updating columns:\n\n'
                + '\n'.join([str(e) for e in errors])
            )

        for column_name in column_names:
            column_metadata = deepcopy(kwargs)
            if not has_sdtype_key:
                column_metadata['sdtype'] = self.columns[column_name]['sdtype']

            self.columns[column_name] = column_metadata

        self._updated = True

    def update_columns_metadata(self, column_metadata):
        """Update the metadata for multiple columns using metadata from the input dictionary.

        Args:
            column_metadata (dict):
                A dictionary of column names and their metadata to be updated.
        """
        errors = []
        for column_name, kwargs in column_metadata.items():
            try:
                self._validate_update_column(column_name, **kwargs)
            except InvalidMetadataError as e:
                errors.append(e)

        if errors:
            raise InvalidMetadataError(
                'The following errors were found when updating columns:\n\n'
                + '\n'.join([str(e) for e in errors])
            )

        for column_name, kwargs in column_metadata.items():
            if 'sdtype' not in kwargs:
                kwargs['sdtype'] = self.columns[column_name]['sdtype']

            self.columns[column_name] = kwargs

        self._updated = True

    def get_column_names(self, **kwargs):
        """Return a list of column names that match the given metadata keyword arguments.

        Args:
            **kwargs:
                Column metadata keyword arguments to filter on, for example sdtype='id'
                or pii=True.

        Returns:
            list:
                The list of columns that match the metadata kwargs.
        """
        if not kwargs:
            return list(self.columns.keys())

        matches = []
        for col, col_metadata in self.columns.items():
            if kwargs.items() <= col_metadata.items():
                matches.append(col)

        return matches

    def to_dict(self):
        """Return a python ``dict`` representation of the ``SingleTableMetadata``."""
        metadata = {}
        for key in self._KEYS:
            not_version = key != 'METADATA_SPEC_VERSION'
            value = getattr(self, f'{key}', None) if not_version else self._version
            if value:
                metadata[key] = value

        return deepcopy(metadata)

    def _tokenize_column_name(self, column_name):
        """Tokenize a column name.

        Args:
            column_name (str):
                The column name to be tokenized.
        """
        tokens = column_name.replace(' ', '_').replace('-', '_').split('_')
        if len(tokens) == 1:
            tokens = []
            if column_name.upper() != column_name and column_name[1:].lower() != column_name[1:]:
                tokens = re.findall('[A-Z][^A-Z]*', column_name)

        tokens = tokens if tokens else [column_name]
        tokens = [token.lower() for token in tokens]

        return tokens

    def _detect_pii_column(self, column_name):
        """Detect PII columns.

        Args:
            column_name (str):
                The column name to be analyzed.
        """
        # Subset of sdtypes which are unambiguous, ie aren't a substring of another word
        # For such cases just check if the word is in the column name
        cleaned_name = re.sub(r'[^a-zA-Z0-9]', '', column_name.lower())
        for reference, sdtype in self._SDTYPES_WITHOUT_SUBSTRINGS.items():
            if reference in cleaned_name:
                return sdtype

        # To handle the cases where the sdtype could be a substring of another word,
        # tokenize the column name based on (1) symbols and (2) camelCase
        tokens = self._tokenize_column_name(column_name)

        return next(
            (
                sdtype
                for reference, sdtype in self._SDTYPES_WITH_SUBSTRINGS.items()
                if reference in tokens
            ),
            None,
        )

    def _detect_id_column(self, column_name):
        """Detect if the column has id in one of the words.

        Args:
            column_name (str):
                The column name to be analyzed.
        """
        tokens = self._tokenize_column_name(column_name)
        for token in tokens:
            if token == 'id':
                return 'id'

        return None

    def _determine_sdtype_for_numbers(self, data, valid_potential_primary_key):
        """Determine the sdtype for a numerical column.

        Args:
            data (pandas.Series):
                The data to be analyzed.
            valid_potential_primary_key(bool):
                If the column is unique and doesn't have NaNs.
        """
        sdtype = 'numerical'
        pk_candidate = False
        if len(data) > self._MIN_ROWS_FOR_PREDICTION:
            is_not_null = ~data.isna()
            clean_data = (data == data.round()).loc[is_not_null]
            if clean_data.empty:
                return sdtype, pk_candidate

            whole_values = clean_data.all()
            positive_values = (data >= 0).loc[is_not_null].all()

            unique_values = data.nunique()
            unique_lt_categorical_threshold = unique_values <= min(round(len(data) / 10), 10)

            if whole_values and positive_values and unique_lt_categorical_threshold:
                sdtype = 'categorical'

            pk_candidate = valid_potential_primary_key and whole_values and positive_values

        return sdtype, pk_candidate

    def _determine_sdtype_for_objects(self, data, valid_potential_primary_key):
        """Determine the sdtype for an object column.

        Args:
            data (pandas.Series):
                The data to be analyzed.
            valid_potential_primary_key(bool):
                If the column is unique and doesn't have NaNs.
        """
        sdtype = 'categorical'
        pk_candidate = False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            data_test = data.sample(10000) if len(data) > 10000 else data

            try:
                datetime_format = _get_datetime_format(data_test)
                if datetime_format:
                    pd.to_datetime(data_test, format=datetime_format, errors='raise')
                    sdtype = 'datetime'

            except Exception:
                pass

        enough_data = len(data) > self._MIN_ROWS_FOR_PREDICTION
        if enough_data and sdtype == 'categorical' and valid_potential_primary_key:
            pk_candidate = True

        return sdtype, pk_candidate

    def _handle_detection_error(self, error, column, table_name=None):
        """Add user friendly error message with column name to underlying exception.

        Args:
            error (Exception):
                The exception to handle.
            column (str):
                The name of the column being detected.
            table_name (str):
                The name of the table name being detected.
        """
        error_type = type(error).__name__
        table_str = f"table '{table_name}' " if table_name else ''
        error_message = (
            f"Unable to detect metadata for {table_str}column '{column}'. This may be because "
            f'the data type is not supported.\n {error_type}: {error}'
        )
        raise InvalidMetadataError(error_message) from error

    def _detect_sdtype_and_primary_key(
        self, column_data, column_name, pii_pk_candidates, pk_candidates, table_name
    ):
        """Detect the sdtype for a column and whether or not it could be a primary key.

        Args:
            column_data (pandas.DataFrame):
                The data to detect.
            column_name (str):
                The name of the column to detect.
            pii_pk_candidates (list):
                A list of column names that are detected as pii columns.
            pk_candidates (list):
                A list of column names that are potential primary keys.
            table_name (str):
                The name of the data table.
        """
        try:
            clean_data = column_data.dropna()
            dtype = clean_data.infer_objects().dtype.kind
            pk_candidate = False
            has_nan = column_data.isna().any()
            valid_potential_primary_key = column_data.is_unique and not has_nan
            sdtype = self._detect_pii_column(column_name) or self._detect_id_column(column_name)

            if sdtype is None:
                if dtype in self._DTYPES_TO_SDTYPES:
                    sdtype = self._DTYPES_TO_SDTYPES[dtype]
                elif dtype in self._NUMERICAL_DTYPES:
                    sdtype, pk_candidate = self._determine_sdtype_for_numbers(
                        column_data, valid_potential_primary_key
                    )

                elif dtype == 'O':
                    sdtype, pk_candidate = self._determine_sdtype_for_objects(
                        column_data, valid_potential_primary_key
                    )

                if sdtype is None:
                    table_str = f"table '{table_name}' " if table_name else ''
                    error_message = (
                        f"Unsupported data type for {table_str}column '{column_name}' "
                        f"(kind: {dtype}). The valid data types are: 'object', "
                        "'int', 'float', 'datetime', 'bool'."
                    )
                    raise InvalidMetadataError(error_message)
            elif valid_potential_primary_key:
                if sdtype == 'id':
                    pk_candidate = True
                else:
                    pii_pk_candidates.append(column_name)

            if pk_candidate:
                pk_candidates.append(column_name)

            return sdtype, dtype

        except InvalidMetadataError as e:
            raise e
        except Exception as e:
            self._handle_detection_error(e, column_name, table_name)

    def _select_primary_key(self, infer_sdtypes, pk_candidates, pii_pk_candidates):
        """Select the primary key from a list of candidates.

        If there are any non-pii candidates, we select the first one. Otherwise, we select the
        first pii_candidate. If we select a non-pii candidate or we select a pii candidate and
        ``infer_sdtypes`` is False, we set the sdtype to 'id' and delete the 'pii' field if it
        exists.

        Args:
            infer_sdtypes (bool):
                Whether or not the sdtypes were inferred.
            pk_candidates (list):
                A list of primary key candidates that aren't pii.
            pii_pk_candidates (list):
                A list of primary key candidates that are pii.
        """
        if pk_candidates:
            selected_pk = pk_candidates[0]
            self.primary_key = selected_pk
            self.columns[self.primary_key]['sdtype'] = 'id'

        elif pii_pk_candidates:
            self.primary_key = pii_pk_candidates[0]
            if not infer_sdtypes:
                self.columns[self.primary_key]['sdtype'] = 'id'

        if self.primary_key and self.columns[self.primary_key].get('sdtype') == 'id':
            if self.columns[self.primary_key].get('pii') is not None:
                del self.columns[self.primary_key]['pii']

    def _detect_columns(self, data, table_name=None, infer_sdtypes=True, infer_keys='primary_only'):
        """Detect metadata information for each column in the data.

        Args:
            data (pandas.DataFrame):
                The data to be analyzed.
            table_name (str):
                The name of the table to be analyzed. Defaults to ``None``.
            infer_sdtypes (bool):
                A boolean describing whether to infer the sdtypes of each column.
                If True it infers the sdtypes based on the data.
                If False it does not infer the sdtypes and all columns are marked as unknown.
                Defaults to True.
            infer_keys (str):
                A string describing whether to infer the primary keys. Options are:
                    - 'primary_only': Infer the primary keys.
                    - None: Do not infer any keys.
                Defaults to 'primary_only'.
        """
        old_columns = data.columns
        data.columns = data.columns.astype(str)
        pk_candidates = []
        pii_pk_candidates = []
        for field in data:
            column_data = data[field]
            if infer_sdtypes or infer_keys == 'primary_only':
                sdtype, dtype = self._detect_sdtype_and_primary_key(
                    column_data=column_data,
                    column_name=field,
                    pii_pk_candidates=pii_pk_candidates,
                    pk_candidates=pk_candidates,
                    table_name=table_name,
                )
            column_dict = {}
            if infer_sdtypes:
                sdtype_in_reference = sdtype in self._REFERENCE_TO_SDTYPE.values()
                if sdtype_in_reference and sdtype != 'id':
                    column_dict['pii'] = True

                if sdtype == 'datetime' and dtype == 'O':
                    datetime_format = _get_datetime_format(column_data.iloc[:100])
                    column_dict['datetime_format'] = datetime_format
            else:
                sdtype = 'unknown'
                column_dict['pii'] = True

            column_dict['sdtype'] = sdtype
            self.columns[field] = deepcopy(column_dict)

        if infer_keys == 'primary_only':
            self._select_primary_key(
                infer_sdtypes=infer_sdtypes,
                pk_candidates=pk_candidates,
                pii_pk_candidates=pii_pk_candidates,
            )

        self._updated = True
        data.columns = old_columns

    def detect_from_dataframe(self, data):
        """Detect the metadata from a ``pd.DataFrame`` object.

        This method automatically detects the ``sdtypes`` for the given ``pandas.DataFrame``.
        All data column names are converted to strings.

        Args:
            data (pandas.DataFrame):
                ``pandas.DataFrame`` to detect the metadata from.
        """
        if self.columns:
            raise InvalidMetadataError(
                'Metadata already exists. Create a new ``SingleTableMetadata`` '
                'object to detect from other data sources.'
            )

        self._detect_columns(data)

        LOGGER.info('Detected metadata:')
        LOGGER.info(json.dumps(self.to_dict(), indent=4))

    def detect_from_csv(self, filepath, read_csv_parameters=None):
        """Detect the metadata from a ``csv`` file.

        This method automatically detects the ``sdtypes`` for a given ``csv`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``csv`` file.
            read_csv_parameters (dict):
                A python dictionary of with string and value accepted by ``pandas.read_csv``
                function. Defaults to ``None``.
        """
        if self.columns:
            raise InvalidMetadataError(
                'Metadata already exists. Create a new ``SingleTableMetadata`` '
                'object to detect from other data sources.'
            )

        data = _load_data_from_csv(filepath, read_csv_parameters)
        self.detect_from_dataframe(data)

    @staticmethod
    def _validate_key_datatype(column_name):
        """Check whether column_name is a string."""
        return isinstance(column_name, str)

    def _validate_keys_sdtype(self, keys, key_type):
        """Validate that each key is of type 'id' or a valid Faker function."""
        bad_keys = set()
        for key in keys:
            if not (
                self.columns[key]['sdtype'] == 'id'
                or is_faker_function(self.columns[key]['sdtype'])
            ):
                bad_keys.add(key)
        if bad_keys:
            raise InvalidMetadataError(
                f"The {key_type}_keys {sorted(bad_keys)} must be type 'id' or another PII type."
            )

    def _validate_key(self, column_name, key_type):
        """Validate the primary and sequence keys."""
        if column_name is not None:
            if not self._validate_key_datatype(column_name):
                raise InvalidMetadataError(f"'{key_type}_key' must be a string.")

            keys = {column_name} if isinstance(column_name, str) else set(column_name)
            setting_sequence_as_primary = key_type == 'primary' and column_name == self.sequence_key
            setting_primary_as_sequence = key_type == 'sequence' and column_name == self.primary_key
            if setting_sequence_as_primary or setting_primary_as_sequence:
                raise InvalidMetadataError(
                    f'The column ({column_name}) cannot be set as {key_type}_key as it is already '
                    f'set as the {"sequence" if key_type == "primary" else "primary"}_key.'
                )

            invalid_ids = keys - set(self.columns)
            if invalid_ids:
                raise InvalidMetadataError(
                    f'Unknown {key_type} key values {invalid_ids}.'
                    ' Keys should be columns that exist in the table.'
                )

            self._validate_keys_sdtype(keys, key_type)

    def set_primary_key(self, column_name):
        """Set the metadata primary key.

        Args:
            column_name (str):
                Name of the primary key column(s).
        """
        self._validate_key(column_name, 'primary')
        if column_name in self.alternate_keys:
            warnings.warn(
                f"'{column_name}' is currently set as an alternate key and will be removed from "
                'that list.'
            )
            self.alternate_keys.remove(column_name)

        if self.primary_key is not None:
            warnings.warn(
                f"There is an existing primary key '{self.primary_key}'. This key will be removed."
            )

        self._updated = True
        self.primary_key = column_name

    def remove_primary_key(self):
        """Remove the metadata primary key."""
        if self.primary_key is None:
            warnings.warn('No primary key exists to remove.')

        self._updated = True
        self.primary_key = None

    def set_sequence_key(self, column_name):
        """Set the metadata sequence key.

        Args:
            column_name (str):
                Name of the sequence key column(s).
        """
        self._validate_key(column_name, 'sequence')
        if self.sequence_key is not None:
            warnings.warn(
                f"There is an existing sequence key '{self.sequence_key}'."
                ' This key will be removed.'
            )

        self._updated = True
        self.sequence_key = column_name

    def _validate_alternate_keys(self, column_names):
        if not isinstance(column_names, list) or not all(
            self._validate_key_datatype(column_name) for column_name in column_names
        ):
            raise InvalidMetadataError("'alternate_keys' must be a list of strings.")

        keys = set()
        for column_name in column_names:
            keys.update({column_name} if isinstance(column_name, str) else set(column_name))

        invalid_ids = keys - set(self.columns)
        if invalid_ids:
            raise InvalidMetadataError(
                f'Unknown alternate key values {invalid_ids}.'
                ' Keys should be columns that exist in the table.'
            )

        if self.primary_key in column_names:
            raise InvalidMetadataError(
                f"Invalid alternate key '{self.primary_key}'. The key is "
                'already specified as a primary key.'
            )

        self._validate_keys_sdtype(keys, 'alternate')

    def add_alternate_keys(self, column_names):
        """Set the metadata alternate keys.

        Args:
            column_names (list[str]):
                List of names of the alternate key columns.
        """
        self._validate_alternate_keys(column_names)
        for column in column_names:
            if column in self.alternate_keys:
                warnings.warn(f'{column} is already an alternate key.')
            else:
                self.alternate_keys.append(column)

        self._updated = True

    def _validate_sequence_index(self, column_name):
        if not isinstance(column_name, str):
            raise InvalidMetadataError("'sequence_index' must be a string.")

        if column_name not in self.columns:
            column_name = {column_name}
            raise InvalidMetadataError(
                f'Unknown sequence index value {column_name}.'
                ' Keys should be columns that exist in the table.'
            )

        sdtype = self.columns[column_name].get('sdtype')
        if sdtype not in ['datetime', 'numerical']:
            raise InvalidMetadataError(
                "The sequence_index must be of type 'datetime' or 'numerical'."
            )

    def set_sequence_index(self, column_name):
        """Set the metadata sequence index.

        Args:
            column_name (str):
                Name of the sequence index column.
        """
        self._validate_sequence_index(column_name)
        self.sequence_index = column_name
        self._updated = True

    def remove_sequence_index(self):
        """Remove the sequence index."""
        if self.sequence_index is None:
            warnings.warn('No sequence index exists to remove.')

        self.sequence_index = None
        self._updated = True

    def _validate_sequence_index_not_in_sequence_key(self):
        """Check that ``_sequence_index`` and ``_sequence_key`` don't overlap."""
        seq_key = self.sequence_key
        sequence_key = set(_cast_to_iterable(seq_key))
        if self.sequence_index in sequence_key or seq_key is None:
            index = {self.sequence_index}
            raise InvalidMetadataError(
                f"'sequence_index' and 'sequence_key' have the same value {index}."
                ' These columns must be different.'
            )

    def _append_error(self, errors, method, *args, **kwargs):
        """Inplace, append the produced error to the passed ``errors`` list."""
        try:
            method(*args, **kwargs)
        except InvalidMetadataError as e:
            errors.append(e)

    def _validate_column_relationship(self, relationship):
        """Validate a column relationship.

        Verify that a column relationship has a valid relationship type, has
        columns that are present in the metadata, and that those columns have
        valid sdtypes for the relationship type.

        Args:
            relationship (dict):
                Column relationship to validate.

        Raises:
            - ``InvalidMetadataError`` if relationship is invalid
        """
        relationship_type = relationship['type']
        column_names = relationship['column_names']
        if relationship_type not in self._COLUMN_RELATIONSHIP_TYPES:
            raise InvalidMetadataError(
                f"Unknown column relationship type '{relationship_type}'. "
                f'Must be one of {list(self._COLUMN_RELATIONSHIP_TYPES.keys())}.'
            )

        errors = []
        for column in column_names:
            if column not in self.columns:
                errors.append(f"Column '{column}' not in metadata.")
            elif self.primary_key == column:
                errors.append(f"Cannot use primary key '{column}' in column relationship.")

        columns_to_sdtypes = {
            column: self.columns.get(column, {}).get('sdtype') for column in column_names
        }
        try:
            self._COLUMN_RELATIONSHIP_TYPES[relationship_type](columns_to_sdtypes)

        except ImportError:
            warnings.warn(
                f"The metadata contains a column relationship of type '{relationship_type}' "
                f'which requires the {relationship_type} add-on. '
                'This relationship will be ignored. For higher quality data in this'
                ' relationship, please inquire about the SDV Enterprise tier.'
            )
            raise ImportError

        except Exception as e:
            errors.append(str(e))

        if errors:
            raise InvalidMetadataError('\n'.join(errors))

    def _validate_column_relationship_with_others(self, column_relationship, other_relationships):
        """Validate a column relationship with others.

        Verify that the columns in the relationship are not used in more than one
        column relationship.

        Args:
            column_relationship (dict):
                Column relationship to validate.
            other_relationships (list[dict]):
                List of other column relationships to compare against.
        """
        for other_relationship in other_relationships:
            repeated_columns = set(other_relationship.get('column_names', [])) & set(
                column_relationship['column_names']
            )
            if repeated_columns:
                repeated_columns = "', '".join(repeated_columns)
                raise InvalidMetadataError(
                    f"Columns '{repeated_columns}' is already part of a relationship of type"
                    f" '{other_relationship['type']}'. Columns cannot be part of multiple"
                    ' relationships.'
                )

    def _validate_all_column_relationships(self, column_relationships):
        """Validate all column relationships.

        Validates that all column relationships are well formed and that
        columns are not used in more than one column relationship.

        Args:
            column_relationships (list[dict]):
                List of column relationships to validate.

        Raises:
            - ``InvalidMetadataError`` if the relationships are invalid.
        """
        # Validate relationship keys
        valid_relationship_keys = {'type', 'column_names'}
        for idx, relationship in enumerate(column_relationships):
            if set(relationship.keys()) != valid_relationship_keys:
                unknown_keys = set(relationship.keys()).difference(valid_relationship_keys)
                raise InvalidMetadataError(f'Relationship has invalid keys {unknown_keys}.')

            self._validate_column_relationship_with_others(
                relationship, column_relationships[idx + 1 :]
            )

        # Validate each individual relationship
        errors = []
        self._valid_column_relationships = deepcopy(column_relationships)
        invalid_indexes = []
        for idx, relationship in enumerate(column_relationships):
            try:
                self._append_error(
                    errors,
                    self._validate_column_relationship,
                    relationship,
                )
            except ImportError:
                invalid_indexes.append(idx)

        for idx in reversed(invalid_indexes):
            del self._valid_column_relationships[idx]

        if errors:
            raise InvalidMetadataError(
                'Column relationships have following errors:\n'
                + '\n'.join([str(e) for e in errors])
            )

    def add_column_relationship(self, relationship_type, column_names):
        """Add a column relationship to the metadata.

        Args:
            relationship_type (str):
                Type of column relationship.
            column_names (list[str]):
                List of column names in the relationship.
        """
        relationship = {'type': relationship_type, 'column_names': column_names}
        to_check = [relationship] + self.column_relationships
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self._validate_all_column_relationships(to_check)

        self.column_relationships.append(relationship)
        self._updated = True

    def validate(self):
        """Validate the metadata.

        Raises:
            - ``InvalidMetadataError`` if the metadata is invalid.
        """
        errors = []

        # Validate keys
        self._append_error(errors, self._validate_key, self.primary_key, 'primary')
        self._append_error(errors, self._validate_key, self.sequence_key, 'sequence')
        if self.sequence_index:
            self._append_error(errors, self._validate_sequence_index, self.sequence_index)
            self._append_error(errors, self._validate_sequence_index_not_in_sequence_key)

        self._append_error(errors, self._validate_alternate_keys, self.alternate_keys)

        # Validate columns
        for column, kwargs in self.columns.items():
            self._append_error(errors, self._validate_column_args, column, **kwargs)

        # Validate column relationships
        self._append_error(
            errors, self._validate_all_column_relationships, self.column_relationships
        )

        if errors:
            raise InvalidMetadataError(
                'The following errors were found in the metadata:\n\n'
                + '\n'.join([str(e) for e in errors])
            )

    def _validate_metadata_matches_data(self, columns):
        errors = []
        metadata_columns = self.columns or {}
        missing_data_columns = set(columns).difference(metadata_columns)
        if missing_data_columns:
            errors.append(
                f'The columns {sorted(missing_data_columns)} are not present in the metadata.'
            )

        missing_metadata_columns = set(metadata_columns).difference(columns)
        if missing_metadata_columns:
            errors.append(
                f'The metadata columns {sorted(missing_metadata_columns)} '
                'are not present in the data.'
            )

        if errors:
            raise InvalidDataError(errors)

    def _get_primary_and_alternate_keys(self):
        """Get set of primary and alternate keys.

        Returns:
            set:
                Set of keys.
        """
        keys = set(self.alternate_keys)
        if self.primary_key:
            keys.update({self.primary_key})

        return keys

    def _get_set_of_sequence_keys(self):
        """Get set with a sequence key.

        Returns:
            set:
                Set of keys.
        """
        if isinstance(self.sequence_key, tuple):
            return set(self.sequence_key)

        if isinstance(self.sequence_key, str):
            return {self.sequence_key}

        return set()

    def _validate_keys_dont_have_missing_values(self, data):
        errors = []
        keys = self._get_primary_and_alternate_keys()
        keys.update(self._get_set_of_sequence_keys())
        for key in sorted(keys):
            if pd.isna(data[key]).any():
                errors.append(f"Key column '{key}' contains missing values.")

        return errors

    def _validate_key_values_are_unique(self, data):
        errors = []
        keys = self._get_primary_and_alternate_keys()
        for key in sorted(keys):
            repeated_values = set(data[key][data[key].duplicated()])
            if repeated_values:
                repeated_values = _format_invalid_values_string(repeated_values, 3)
                errors.append(f"Key column '{key}' contains repeating values: " + repeated_values)

        return errors

    def _validate_primary_key(self, data):
        error = []
        is_int = self.primary_key and pd.api.types.is_integer_dtype(data[self.primary_key])
        regex = self.columns.get(self.primary_key, {}).get('regex_format')
        if is_int and regex:
            possible_characters = get_possible_chars(regex, 1)
            if '0' in possible_characters:
                error.append(f'Primary key "{self.primary_key}" {INT_REGEX_ZERO_ERROR_MESSAGE}')

        return error

    @staticmethod
    def _get_invalid_column_values(column, validation_function):
        valid = column.apply(validation_function).astype(bool)

        return set(column[~valid])

    def _validate_column_data(self, column, sdtype_warnings):
        """Validate the values of the given column against its specified sdtype properties.

        The function checks the sdtype of the column and validates the data accordingly. If there
        are any errors, those are being appended to a list of errors that will be returned.
        Additionally ``sdtype_warnings`` is being updated with ``datetime_format`` warnings
        to be raised later.

        Args:
            column (pd.Series):
                The data to validate against.
            sdtype_warnings (defaultdict[list]):
                A ``defaultdict`` with ``list`` to add warning messages.

        Returns:
            list:
                A list containing any validation error messages found during the process.
        """
        column_metadata = self.columns[column.name]
        sdtype = column_metadata['sdtype']
        invalid_values = None

        # boolean values must be True/False, None or missing values
        # int/str are not allowed
        if sdtype == 'boolean':
            invalid_values = self._get_invalid_column_values(column, _is_boolean_type)

        # numerical values must be int/float, None or missing values
        # str/bool are not allowed
        if sdtype == 'numerical':
            invalid_values = self._get_invalid_column_values(column, _is_numerical_type)

        # datetime values must be castable to datetime, None or missing values
        if sdtype == 'datetime':
            datetime_format = column_metadata.get('datetime_format')
            if datetime_format:
                invalid_values = _validate_datetime_format(column, datetime_format)
            else:
                # cap number of samples to be validated to improve performance
                num_samples_to_validate = min(len(column), 1000)

                invalid_values = self._get_invalid_column_values(
                    column.sample(num_samples_to_validate),
                    lambda x: pd.isna(x) | _is_datetime_type(x),
                )

            if datetime_format is None and column.dtype == 'O':
                sdtype_warnings['Column Name'].append(column.name)
                sdtype_warnings['sdtype'].append(sdtype)
                sdtype_warnings['datetime_format'].append(datetime_format)

        if invalid_values:
            invalid_values = _format_invalid_values_string(invalid_values, 3)
            return [f"Invalid values found for {sdtype} column '{column.name}': {invalid_values}."]

        return []

    def validate_data(self, data, sdtype_warnings=None):
        """Validate the data matches the metadata.

        Checks the metadata follows the following rules:
            * data columns match the metadata
            * keys don't have missing values
            * primary or alternate keys are unique
            * values of a column satisfy their sdtype
            * datetimes represented as objects have ``datetime_format`` (warning only).

        Args:
            data (pd.DataFrame):
                The data to validate.
            sdtype_warnings (defaultdict[list] or None):
                A ``defaultdict`` with ``list`` to add warning messages.

        Raises:
            InvalidDataError:
                This error is being raised if the data is not matching its sdtype requirements.

        Warns:
            A warning is being raised if ``datetime_format`` is missing from a column represented
            as ``object`` in the dataframe and its sdtype is ``datetime``.
        """
        sdtype_warnings = sdtype_warnings if sdtype_warnings is not None else defaultdict(list)
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be a DataFrame, not a {type(data)}.')

        # Both metadata and data must have the same set of columns
        self._validate_metadata_matches_data(data.columns)

        errors = []
        # Primary, sequence and alternate keys can't have missing values
        errors += self._validate_keys_dont_have_missing_values(data)

        # Primary and alternate key values must be unique
        errors += self._validate_key_values_are_unique(data)

        # Every column must satisfy the properties of their sdtypes
        for column in data:
            errors += self._validate_column_data(data[column], sdtype_warnings)

        errors += self._validate_primary_key(data)
        if sdtype_warnings is not None and len(sdtype_warnings):
            df = pd.DataFrame(sdtype_warnings)
            message = (
                "No 'datetime_format' is present in the metadata for the following columns:\n"
                f'{df.to_string(index=False)}\n'
                'Without this specification, SDV may not be able to accurately parse the data. '
                "We recommend adding datetime formats using 'update_column'."
            )

            warnings.warn(message)

        if errors:
            raise InvalidDataError(errors)

    def anonymize(self):
        """Anonymize metadata by obfuscating column names.

        Returns:
            SingleTableMetadata:
                An anonymized SingleTableMetadata instance.
        """
        anonymized_metadata = {'columns': {}}

        self._anonymized_column_map = {}
        counter = 1
        for column, column_metadata in self.columns.items():
            anonymized_column = f'col{counter}'
            self._anonymized_column_map[column] = anonymized_column
            anonymized_metadata['columns'][anonymized_column] = column_metadata
            counter += 1

        if self.primary_key:
            anonymized_metadata['primary_key'] = self._anonymized_column_map[self.primary_key]

        if self.alternate_keys:
            anonymized_alternate_keys = []
            for alternate_key in self.alternate_keys:
                anonymized_alternate_keys.append(self._anonymized_column_map[alternate_key])

            anonymized_metadata['alternate_keys'] = anonymized_alternate_keys

        if self.sequence_key:
            anonymized_metadata['sequence_key'] = self._anonymized_column_map[self.sequence_key]

        if self.sequence_index:
            anonymized_metadata['sequence_index'] = self._anonymized_column_map[self.sequence_index]

        return SingleTableMetadata.load_from_dict(anonymized_metadata)

    def visualize(self, show_table_details='full', output_filepath=None):
        """Create a visualization of the single-table dataset.

        Args:
            show_table_details (str):
                If 'full', the column names, primary, alternate and sequence keys are all
                shown. If 'summarized', primary, alternate and sequence keys are shown and a
                count of the different sdtypes. Defaults to 'full'.
            output_filepath (str):
                Full path of where to save the visualization. If None, the visualization is not
                saved. Defaults to None.

        Returns:
            ``graphviz.Digraph`` object.
        """
        if show_table_details not in ('full', 'summarized'):
            raise ValueError("'show_table_details' should be 'full' or 'summarized'.")

        if show_table_details == 'full':
            node = rf'{create_columns_node(self.columns)}\l'

        elif show_table_details == 'summarized':
            node = rf'{create_summarized_columns_node(self.columns)}\l'

        keys_node = ''
        if self.primary_key:
            keys_node = rf'{keys_node}Primary key: {self.primary_key}\l'

        if self.sequence_key:
            keys_node = rf'{keys_node}Sequence key: {self.sequence_key}\l'

        if self.sequence_index:
            keys_node = rf'{keys_node}Sequence index: {self.sequence_index}\l'

        if self.alternate_keys:
            alternate_keys = [rf'&nbsp; &nbsp; • {key}\l' for key in self.alternate_keys]
            alternate_keys = ''.join(alternate_keys)
            keys_node = rf'{keys_node}Alternate keys:\l {alternate_keys}'

        if keys_node != '':
            node = rf'{node}|{keys_node}'

        node = {'': f'{{{node}}}'}
        return visualize_graph(node, [], output_filepath)

    def save_to_json(self, filepath, mode='write'):
        """Save the current ``SingleTableMetadata`` in to a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file to be written.
            mode (str):
                String that determines the mode of the function. Defaults to ``write``.
                'write' mode will create and write a file if it does not exist.
                'overwrite' mode will overwrite a file if that file does exist.

        Raises:
            Raises an ``Error`` if the path already exists and the mode is 'write'.
        """
        _validate_file_mode(mode)
        if mode == 'write':
            validate_file_does_not_exist(filepath)
        metadata = self.to_dict()
        metadata['METADATA_SPEC_VERSION'] = self.METADATA_SPEC_VERSION
        SINGLETABLEMETADATA_LOGGER.info(
            '\nMetadata Save:\n'
            '  Timestamp: %s\n'
            '  Statistics about the metadata:\n'
            '    Total number of tables: 1'
            '    Total number of columns: %s'
            '    Total number of relationships: 0',
            datetime.now(),
            len(self.columns),
        )
        with open(filepath, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

        self._updated = False

    @classmethod
    def load_from_dict(cls, metadata_dict):
        """Create a ``SingleTableMetadata`` instance from a python ``dict``.

        Args:
            metadata_dict (dict):
                Python dictionary representing a ``SingleTableMetadata`` object.

        Returns:
            Instance of ``SingleTableMetadata``. Column names are converted to
            string type.
        """
        instance = cls()
        for key in instance._KEYS:
            value = deepcopy(metadata_dict.get(key))
            if value:
                if key == 'columns':
                    value = {
                        str(key) if not isinstance(key, str) else key: col
                        for key, col in value.items()
                    }
                setattr(instance, f'{key}', value)

        instance._primary_key_candidates = None
        return instance

    @classmethod
    def load_from_json(cls, filepath):
        """Create an instance from a ``json`` file.

        Args:
            filepath (str):
                String that represents the ``path`` to the ``json`` file.

        Raises:
            - An ``Error`` if the path does not exist.
            - An ``Error`` if the ``json`` file does not contain the ``METADATA_SPEC_VERSION``.

        Returns:
            A ``SingleTableMetadata`` instance.
        """
        metadata = read_json(filepath)
        if 'METADATA_SPEC_VERSION' not in metadata:
            raise InvalidMetadataError(
                'This metadata file is incompatible with the ``SingleTableMetadata`` '
                'class and version.'
            )

        return cls.load_from_dict(metadata)

    def __repr__(self):
        """Pretty print the ``SingleTableMetadata``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed

    @classmethod
    def upgrade_metadata(cls, filepath):
        """Upgrade an old metadata file to the ``V1`` schema.

        Args:
            filepath (str):
                String that represents the ``path`` to the old metadata ``json`` file.

        Raises:
            Raises a ``ValueError`` if the filepath does not exist.

        Returns:
            A ``SingleTableMetadata`` instance.
        """
        old_metadata = read_json(filepath)
        if 'tables' in old_metadata:
            tables = old_metadata.get('tables')
            if len(tables) > 1:
                raise InvalidMetadataError(
                    'There are multiple tables specified in the JSON. '
                    'Try using the MultiTableMetadata class to upgrade this file.'
                )

            else:
                old_metadata = list(tables.values())[0]

        new_metadata = convert_metadata(old_metadata)
        metadata = cls.load_from_dict(new_metadata)

        try:
            metadata.validate()
        except InvalidMetadataError as error:
            message = (
                'Successfully converted the old metadata, but the metadata was not valid. '
                f'To use this with the SDV, please fix the following errors.\n {str(error)}'
            )
            warnings.warn(message)

        return metadata
