"""DayZ parameter detection and creation."""

import pandas as pd

from sdv._utils import _datetime_string_matches_format, _get_datetime_format, _is_numerical
from sdv.errors import SynthesizerInputError, SynthesizerProcessingError
from sdv.single_table._dayz_utils import create_parameters

DAYZ_PARAMETER_KEYS = ['DAYZ_SPEC_VERSION', 'tables', 'relationships']
TABLE_PARAMETER_KEYS = ['columns', 'num_rows']
COLUMN_PARAMETER_KEYS = ['missing_values_proportion']
NUMERICAL_PARAMETER_KEYS = COLUMN_PARAMETER_KEYS + ['num_decimal_digits', 'min_value', 'max_value']
CATEGORICAL_PARAMETER_KEYS = COLUMN_PARAMETER_KEYS + ['category_values']
DATETIME_PARAMETER_KEYS = COLUMN_PARAMETER_KEYS + ['start_timestamp', 'end_timestamp']
SDTYPE_TO_PARAMETERS = {
    'numerical': NUMERICAL_PARAMETER_KEYS,
    'categorical': CATEGORICAL_PARAMETER_KEYS,
    'datetime': DATETIME_PARAMETER_KEYS,
}


def _validate_parameter_dict_keys(dayz_parameters):
    unknown_base_keys = dayz_parameters.keys() - DAYZ_PARAMETER_KEYS
    if unknown_base_keys:
        unknown_base_keys = "', '".join(unknown_base_keys)
        raise SynthesizerProcessingError(
            f"DayZ parameters contains unexpected key(s): '{unknown_base_keys}'."
        )


def _validate_table_parameter_dict_keys(table, table_parameters):
    unknown_table_keys = table_parameters.keys() - TABLE_PARAMETER_KEYS
    if unknown_table_keys:
        unknown_table_keys = "', '".join(unknown_table_keys)
        msg = (
            f"DayZ parameters contain unexpected key(s) '{unknown_table_keys}' for table '{table}'."
        )
        raise SynthesizerProcessingError(msg)


def _validate_parameter_structure(dayz_parameters):
    if not isinstance(dayz_parameters, dict):
        raise SynthesizerProcessingError(
            'DayZ parameters must be a dictionary of DayZSynthesizer parameters.'
        )

    _validate_parameter_dict_keys(dayz_parameters)

    table_parameters = dayz_parameters.get('tables', {})
    if not isinstance(table_parameters, dict) or not all(
        isinstance(value, dict) for value in table_parameters.values()
    ):
        raise SynthesizerProcessingError(
            "The 'tables' value in the DayZ parameters must be a dictionary of table parameters."
        )
    for table, table_parameters in dayz_parameters.get('tables', {}).items():
        _validate_table_parameter_dict_keys(table, table_parameters)


def _validate_numerical_parameters(column_parameters, column_table_msg):
    for param in ['min_value', 'max_value']:
        if param in column_parameters and not _is_numerical(column_parameters[param]):
            msg = f"The '{param}' parameter for {column_table_msg} must be a float."
            raise SynthesizerProcessingError(msg)

    if 'min_value' in column_parameters and 'max_value' in column_parameters:
        if column_parameters['min_value'] > column_parameters['max_value']:
            msg = (
                f"Invalid parameters for {column_table_msg}. The 'min_value' "
                "must be less than the 'max_value'"
            )
            raise SynthesizerProcessingError(msg)
    elif 'min_value' in column_parameters or 'max_value' in column_parameters:
        raise SynthesizerProcessingError(
            f'Invalid parameters for {column_table_msg}. Both the '
            "'min_value' and 'max_value' parameters must be set."
        )


def _validate_datetime_parameters(column_parameters, column_metadata, column_table_msg):
    """Validate that the timestamps are valid and match the datetime format.

    If a datetime format has not been provided in the metadata, the datetime format
    is detected from the provided timestamp. If the detected datetime format is None
    or if the provided timestamp does not match the datetime format, the validation
    errors.
    """
    datetime_format = column_metadata.get('datetime_format')
    for param in ['start_timestamp', 'end_timestamp']:
        if param not in column_parameters:
            continue

        timestamp = column_parameters[param]
        if not isinstance(timestamp, str):
            msg = f"The '{param}' parameter for {column_table_msg} must be a string."
            raise SynthesizerProcessingError(msg)

        datetime_format = datetime_format if datetime_format else _get_datetime_format(timestamp)
        if not datetime_format or not _datetime_string_matches_format(timestamp, datetime_format):
            if column_metadata.get('datetime_format'):
                msg = (
                    f"The '{param}' parameter for {column_table_msg} is not a valid datetime "
                    f'string or does not match the date time format ({datetime_format}).'
                )
            else:
                msg = (
                    f"The '{param}' parameter for {column_table_msg} is not a "
                    'valid datetime string.'
                )

            raise SynthesizerProcessingError(msg)

    if 'start_timestamp' in column_parameters and 'end_timestamp' in column_parameters:
        start_datetime = pd.to_datetime(
            column_parameters['start_timestamp'], format=datetime_format
        )
        end_datetime = pd.to_datetime(column_parameters['end_timestamp'], format=datetime_format)
        if start_datetime > end_datetime:
            msg = (
                f"Invalid parameters for {column_table_msg}. The 'start_timestamp' must be "
                "less than the 'end_timestamp'."
            )
            raise SynthesizerProcessingError(msg)
    elif 'start_timestamp' in column_parameters or 'end_timestamp' in column_parameters:
        raise SynthesizerProcessingError(
            f'Invalid parameters for {column_table_msg}. Both the '
            "'start_timestamp' and 'end_timestamp' parameters must be set."
        )


def _validate_categorical_parameters(column_parameters, column_table_msg):
    if not isinstance(column_parameters.get('category_values', []), list):
        msg = f"The 'category_values' parameter for {column_table_msg} must be a list."
        raise SynthesizerProcessingError(msg)


def _validate_missing_value_parameters(column_parameters, column_table_msg):
    missing_values_proportion = column_parameters['missing_values_proportion']
    if not _is_numerical(missing_values_proportion) or (
        missing_values_proportion < 0.0 or missing_values_proportion > 1.0
    ):
        msg = (
            f"The 'missing_values_proportion' parameter for {column_table_msg} "
            'must be a float between 0.0 and 1.0.'
        )
        raise SynthesizerProcessingError(msg)


def _validate_column_parameters(table, column, column_metadata, column_parameters):
    column_table_msg = f"column '{column}' in table '{table}'"
    sdtype = column_metadata['sdtype']
    sdtype_parameters = SDTYPE_TO_PARAMETERS.get(sdtype, COLUMN_PARAMETER_KEYS)
    unknown_column_parameters = column_parameters.keys() - set(sdtype_parameters)
    if unknown_column_parameters:
        unknown_column_parameters = "', '".join(unknown_column_parameters)
        msg = (
            f'The parameters for {column_table_msg} contains unexpected key(s) '
            f"'{unknown_column_parameters}'."
        )
        raise SynthesizerProcessingError(msg)

    if sdtype == 'numerical':
        _validate_numerical_parameters(column_parameters, column_table_msg)
    elif sdtype == 'datetime':
        _validate_datetime_parameters(column_parameters, column_metadata, column_table_msg)
    elif sdtype == 'categorical':
        _validate_categorical_parameters(column_parameters, column_table_msg)

    if 'missing_values_proportion' in column_parameters:
        _validate_missing_value_parameters(column_parameters, column_table_msg)


def _validate_table_parameters(table, table_metadata, table_parameters):
    missing_cols = table_parameters.get('columns', {}).keys() - table_metadata.columns.keys()
    if missing_cols:
        missing_cols = "', '".join(missing_cols)
        msg = (
            f"Invalid DayZ parameters provided, column(s) '{missing_cols}' are missing from "
            f"table '{table}' in the metadata."
        )
        raise SynthesizerProcessingError(msg)

    num_rows = table_parameters.get('num_rows')
    if 'num_rows' in table_parameters and (not isinstance(num_rows, int) or num_rows <= 0):
        msg = (
            f"Invalid DayZ parameter 'num_rows' for table '{table}'. The 'num_rows' parameter "
            'must be an integer greater than zero.'
        )
        raise SynthesizerProcessingError(msg)

    for column, column_parameters in table_parameters.get('columns', {}).items():
        _validate_column_parameters(
            table, column, table_metadata.columns[column], column_parameters
        )


def _validate_tables_parameter(metadata, dayz_parameters):
    tables = dayz_parameters.get('tables', {}).keys()
    missing_tables = tables - metadata.tables.keys()
    if missing_tables:
        missing_tables = "', '".join(missing_tables)
        msg = (
            f"Invalid DayZ parameters provided, table(s) '{missing_tables}' "
            'are missing from the metadata.'
        )
        raise SynthesizerProcessingError(msg)

    for table, table_parameters in dayz_parameters.get('tables', {}).items():
        table_metadata = metadata.tables[table]
        _validate_table_parameters(table, table_metadata, table_parameters)


def _validate_parameters(metadata, parameters):
    """Validate a DayZSynthesizer parameters dictionary.

    Args:
        metadata (sdv.Metadata):
            Metadata for the data.
        parameters (dict):
            The DayZ parameter dictionary.
    """
    metadata.validate()
    _validate_parameter_structure(parameters)
    if 'relationships' in parameters:
        msg = (
            "Invalid DayZ parameter 'relationships' for single-table DayZSynthesizer. "
            'Please use multi-table DayZSynthesizer instead.'
        )
        raise SynthesizerProcessingError(msg)

    _validate_tables_parameter(metadata, parameters)


class DayZSynthesizer:
    """Single-Table DayZSynthesizer for public SDV."""

    def __init__(self, metadata, locales=['en_US']):
        raise SynthesizerInputError(
            "Only the 'DayZSynthesizer.create_parameters' and the "
            'DayZSynthesizer.validate_parameters methods are an SDV public feature. To '
            'define and use a DayZSynthesizer object you must have SDV-Enterprise.'
        )

    @classmethod
    def create_parameters(cls, data, metadata, output_filename=None):
        """Create parameters for the DayZ synthesizer.

        Args:
            data (pd.DataFrame): The input data.
            metadata (Metadata): The metadata object.
            output_filename (str, optional): The output filename for the parameters.

        Returns:
            dict: The created parameters.
        """
        return create_parameters(data, metadata, output_filename)

    @staticmethod
    def validate_parameters(metadata, parameters):
        """Validate a DayZSynthesizer parameters dictionary.

        Args:
            metadata (sdv.Metadata):
                Metadata for the data.
            parameters (dict):
                The DayZ parameter dictionary.
        """
        _validate_parameters(metadata, parameters)
