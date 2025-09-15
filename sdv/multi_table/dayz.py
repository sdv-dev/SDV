"""Multi-Table DayZ parameter detection and creation."""

import json

import pandas as pd

from sdv.errors import SynthesizerInputError
from sdv.single_table.dayz import create_parameters


def detect_relationship_parameters(data, metadata):
    """Detect all relationship-level for the DayZ parameters.

    The relationship-level parameters are:
    - The min and max cardinality

    Args:
        data (pd.DataFrame): The input data.
        metadata (Metadata): The metadata object.

    Returns:
        dict: A dictionary containing the detected parameters.
    """
    relationship_parameters = {}
    for relationship in metadata.relationships:
        rel_tuple = (
            relationship['parent_table_name'],
            relationship['child_table_name'],
            relationship['parent_primary_key'],
            relationship['child_foreign_key'],
        )
        cardinality_table = pd.DataFrame(index=data[rel_tuple[0]][rel_tuple[2]].copy())
        cardinality_table['cardinality'] = data[rel_tuple[1]][rel_tuple[3]].value_counts()
        cardinality_table = cardinality_table.fillna(0)
        relationship_parameters[json.dumps(rel_tuple)] = {
            'min_cardinality': cardinality_table['cardinality'].min(),
            'max_cardinality': cardinality_table['cardinality'].max(),
        }

    return relationship_parameters


def create_parameters_multi_table(data, metadata, output_filename):
    """Create parameters for the DayZSynthesizer."""
    parameters = create_parameters(data, metadata, None)
    parameters['relationships'] = detect_relationship_parameters(data, metadata)
    if output_filename:
        with open(output_filename, 'w') as f:
            json.dump(parameters, f, indent=4)

    return parameters


class DayZSynthesizer:
    """Multi-Table DayZSynthesizer for public SDV."""

    def __init__(self, metadata, locales=['en_US']):
        raise SynthesizerInputError(
            "Only the 'DayZSynthesizer.create_parameters' is a SDV public feature. "
            'To define and use and use a DayZSynthesizer object you must have an SDV-Enterprise'
            ' version.'
        )

    @classmethod
    def create_parameters(cls, data, metadata, output_filename=None):
        """Create parameters for the DayZSynthesizer.

        Args:
            data (dict[str, pd.DataFrame]): The input data.
            metadata (Metadata): The metadata object.
            output_filename (str, optional): The output filename for the parameters JSON.

        Returns:
            dict: The created parameters.
        """
        return create_parameters_multi_table(data, metadata, output_filename)
