"""DayZ parameter detection and creation."""

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
        column_parameters[column_name] = {}
        sdtype = column_metadata['sdtype']
        if sdtype == 'numerical':
            column_parameters[column_name] = {
                'num_decimal_digits': learn_rounding_digits(data[column_name]),
                'min_value': data[column_name].min().item(),
                'max_value': data[column_name].max().item(),
            }
        elif sdtype == 'datetime':
            datetime_column = pd.to_datetime(data[column_name], errors='coerce')
            column_parameters[column_name] = {
                'start_timestamp': datetime_column.min(),
                'end_timestamp': datetime_column.max(),
            }
        elif sdtype in ['categorical', 'boolean']:
            column_parameters[column_name] = {
                'category_values': data[column_name].dropna().unique().tolist()
            }

        column_parameters[column_name]['missing_value_proportion'] = (
            data[column_name].isna().mean().item()
        )

    return {'columns': column_parameters}


def create_parameters(data, metadata):
    """Detect and create a parameter dict for the DayZ model."""
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

    return parameters


class DayZSynthesizer:
    """Single-Table DayZSynthesizer for public SDV."""

    @classmethod
    def create_parameters(cls, data, metadata, output_filename=None):
        """Create parameters for the DayZ synthesizer.

        Args:
            data (pd.DataFrame): The input data.
            metadata (Metadata): The metadata object.
            output_filename (str, optional): The output filename for the parameters JSON.

        Returns:
            dict: The created parameters.
        """
        parameters = create_parameters(data, metadata)
        if output_filename:
            with open(output_filename, 'w') as f:
                json.dump(parameters, f, indent=4)

        return parameters
