import json

import pandas as pd
from rdt.transformers.utils import learn_rounding_digits


def detect_table_parameters(data):
    """Detect all table-level Dayz parameters.

    - Detect the `num_rows` of the table.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        dict: A dictionary containing the detected parameters.
    """
    return {'num_rows': len(data)}


def _compute_missing_values_proportion(series):
    """Compute missing value proportion with a safe fallback for empty series."""
    if len(series) == 0:
        return 0.0

    value = float(series.isna().mean())
    return 0.0 if pd.isna(value) else value


def _detect_numerical_column_parameters(series):
    """Detect numerical-specific parameters with fallbacks when undetectable.

    Returns only keys that can be reliably detected (no None values).
    """
    params = {}
    non_null = series.dropna()
    if non_null.empty:
        return params

    try:
        num_decimal_digits = learn_rounding_digits(series)
        if isinstance(num_decimal_digits, int) and num_decimal_digits >= 0:
            params['num_decimal_digits'] = num_decimal_digits
    except Exception:
        pass

    min_value = non_null.min()
    max_value = non_null.max()
    if not pd.isna(min_value):
        params['min_value'] = min_value.item() if hasattr(min_value, 'item') else float(min_value)
    if not pd.isna(max_value):
        params['max_value'] = max_value.item() if hasattr(max_value, 'item') else float(max_value)

    return params


def _detect_datetime_column_parameters(series, column_metadata):
    """Detect datetime-specific parameters with fallbacks when undetectable.

    Returns only keys that can be reliably detected (no None values).
    """
    params = {}
    datetime_format = column_metadata.get('datetime_format', None)
    if datetime_format:
        datetime_column = pd.to_datetime(series, format=datetime_format, errors='coerce')
    else:
        datetime_column = pd.to_datetime(series, errors='coerce')

    non_na = datetime_column[~pd.isna(datetime_column)]
    if non_na.empty:
        return params

    start_dt = non_na.min()
    end_dt = non_na.max()
    if datetime_format:
        params['start_timestamp'] = start_dt.strftime(datetime_format)
        params['end_timestamp'] = end_dt.strftime(datetime_format)
    else:
        params['start_timestamp'] = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        params['end_timestamp'] = end_dt.strftime('%Y-%m-%d %H:%M:%S')

    return params


def _detect_categorical_or_boolean_column_parameters(series):
    """Detect categorical/boolean parameters."""
    categorical_values = series.dropna().unique()
    if len(categorical_values) == 0:
        return {}

    return {'category_values': categorical_values.tolist()}


def detect_column_parameters(data, metadata, table_name):
    """Detect all column-level Dayz parameters.

    The column-level parameters are:
    - The missing value proportion
    - The boundaries for numerical and datetime columns
    - The categories for categorical columns
    - The 'num_decimal_digits' for numerical columns

    Args:
        data (pd.DataFrame): The input data.
        metadata (Metadata): The metadata object.

    Returns:
        dict: A dictionary containing the detected parameters.
    """
    table_metadata = metadata.tables[table_name]
    column_parameters = {}
    for column_name, column_metadata in table_metadata.columns.items():
        sdtype = column_metadata['sdtype']
        params = {}
        if sdtype == 'numerical':
            column_parameters[column_name] = {
                'num_decimal_digits': learn_rounding_digits(data[column_name]),
                'min_value': data[column_name].min(),
                'max_value': data[column_name].max(),
            }
        elif sdtype == 'datetime':
            datetime_format = column_metadata.get('datetime_format', None)
            if datetime_format:
                datetime_column = pd.to_datetime(
                    data[column_name], format=datetime_format, errors='coerce'
                )
                start_timestamp = datetime_column.min().strftime(datetime_format)
                end_timestamp = datetime_column.max().strftime(datetime_format)

            else:
                datetime_column = pd.to_datetime(data[column_name], errors='coerce')
                start_timestamp = str(datetime_column.min())
                end_timestamp = str(datetime_column.max())

            column_parameters[column_name] = {
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
            }
        elif sdtype == 'categorical':
            column_parameters[column_name] = {
                'category_values': data[column_name].dropna().unique().tolist()
            }

        column_parameters[column_name]['missing_values_proportion'] = float(
            data[column_name].isna().mean()
        )

    return {'columns': column_parameters}


def create_parameters(data, metadata, output_filename):
    """Detect and create a parameter dict for the DayZ model."""
    if len(data) == 0:
        raise ValueError('Data is empty')
    if len(metadata.tables) == 0:
        raise ValueError('Metadata is empty')

    metadata.validate()
    datas = data if isinstance(data, dict) else {metadata._get_single_table_name(): data}
    metadata.validate_data(datas)
    parameters = {'DAYZ_SPEC_VERSION': 'V1', 'tables': {}}
    for table_name, table_data in datas.items():
        parameters['tables'][table_name] = {}
        parameters['tables'][table_name].update(detect_table_parameters(table_data))
        parameters['tables'][table_name].update(
            detect_column_parameters(table_data, metadata, table_name)
        )

    if output_filename:
        with open(output_filename, 'w') as f:
            json.dump(parameters, f, indent=4)

    return parameters
