"""Multi-Table DayZ parameter detection and creation."""

import json

import pandas as pd

from sdv.cag._utils import _is_list_of_type
from sdv.errors import SynthesizerInputError, SynthesizerProcessingError
from sdv.single_table.dayz import (
    _validate_parameter_structure,
    _validate_tables_parameter,
    create_parameters,
)

REQUIRED_RELATIONSHIP_KEYS = [
    'parent_table_name',
    'child_table_name',
    'parent_primary_key',
    'child_foreign_key',
]
RELATIONSHIP_PARAMETER_KEYS = REQUIRED_RELATIONSHIP_KEYS + [
    'min_cardinality',
    'max_cardinality',
]

DEFAULT_NUM_ROWS = 1000


def _detect_relationship_parameters(data, metadata):
    """Detect all relationship-level for the DayZ parameters.

    The relationship-level parameters are:
    - The min and max cardinality

    Args:
        data (dict[str, pd.DataFrame]): The input data.
        metadata (Metadata): The metadata object.

    Returns:
        dict: A list containing the detected parameters.
    """
    relationship_parameters = []
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
        relationship_parameters.append({
            'parent_table_name': rel_tuple[0],
            'child_table_name': rel_tuple[1],
            'parent_primary_key': rel_tuple[2],
            'child_foreign_key': rel_tuple[3],
            'min_cardinality': cardinality_table['cardinality'].min(),
            'max_cardinality': cardinality_table['cardinality'].max(),
        })

    return relationship_parameters


def create_parameters_multi_table(data, metadata, output_filename):
    """Create parameters for the DayZSynthesizer."""
    parameters = create_parameters(data, metadata, None)
    parameters['relationships'] = _detect_relationship_parameters(data, metadata)
    if output_filename:
        with open(output_filename, 'w') as f:
            json.dump(parameters, f, indent=4)

    return parameters


def _validate_min_cardinality(relationship):
    min_cardinality = relationship['min_cardinality']
    if not isinstance(min_cardinality, int) or min_cardinality < 0:
        msg = (
            f"Invalid 'min_cardinality' parameter ({min_cardinality}). The "
            "'min_cardinality' parameter must be an integer greater than or equal to zero."
        )
        raise SynthesizerProcessingError(msg)


def _validate_max_cardinality(relationship):
    max_cardinality = relationship['max_cardinality']
    if not isinstance(max_cardinality, int) or max_cardinality <= 0:
        msg = (
            f"Invalid 'max_cardinality' parameter ({max_cardinality}). The "
            "'max_cardinality' parameter must be an integer greater than zero."
        )
        raise SynthesizerProcessingError(msg)


def _validate_cardinality_bounds(relationship):
    if relationship['min_cardinality'] > relationship['max_cardinality']:
        msg = (
            "Invalid cardinality parameters, the 'min_cardinality' must be less than or "
            "equal to the 'max_cardinality'."
        )
        raise SynthesizerProcessingError(msg)


def _validate_relationship_structure(dayz_parameters):
    if not _is_list_of_type(dayz_parameters.get('relationships', []), dict):
        raise SynthesizerProcessingError(
            "The 'relationships' parameter value must be a list of dictionaries."
        )

    for relationship in dayz_parameters.get('relationships', []):
        unknown_relationship_parameters = relationship.keys() - set(RELATIONSHIP_PARAMETER_KEYS)
        if unknown_relationship_parameters:
            unknown_relationship_parameters = "', '".join(unknown_relationship_parameters)
            msg = (
                'Relationship parameter contains unexpected key(s) '
                f"'{unknown_relationship_parameters}'."
            )
            raise SynthesizerProcessingError(msg)
        missing_relationship_parameters = set(REQUIRED_RELATIONSHIP_KEYS) - relationship.keys()
        if missing_relationship_parameters:
            missing_relationship_parameters = "', '".join(missing_relationship_parameters)
            msg = (
                'Relationship parameter missing required key(s) '
                f"'{missing_relationship_parameters}'."
            )
            raise SynthesizerProcessingError(msg)

        if 'min_cardinality' in relationship:
            _validate_min_cardinality(relationship)
        if 'max_cardinality' in relationship:
            _validate_max_cardinality(relationship)
        if 'min_cardinality' in relationship and 'max_cardinality' in relationship:
            _validate_cardinality_bounds(relationship)


def _validate_cardinality(relationship_parameters, parent_num_rows, child_num_rows):
    """Validate that the relationship cardinality works with the set number of rows."""
    parent_num_rows = parent_num_rows or DEFAULT_NUM_ROWS
    child_num_rows = child_num_rows or DEFAULT_NUM_ROWS
    min_cardinality = relationship_parameters.get('min_cardinality', 0)
    max_cardinality = relationship_parameters.get('max_cardinality')

    min_child_size = min_cardinality * parent_num_rows
    max_child_size = max_cardinality * parent_num_rows if max_cardinality else None

    if child_num_rows < min_child_size:
        msg = (
            f'Invalid cardinality parameters for relationship {relationship_parameters}. '
            f'Minimum cardinality requires child table to be at least {min_child_size} rows.'
        )
        raise SynthesizerProcessingError(msg)

    if max_child_size and child_num_rows > max_child_size:
        msg = (
            f'Invalid cardinality parameters for relationship {relationship_parameters}. '
            f'Maximum cardinality requires child table to be less than {max_child_size} rows.'
        )
        raise SynthesizerProcessingError(msg)


def _validate_relationship_parameters(metadata, dayz_parameters):
    """Validate that every relationship exists in the metadata and the cardinality is valid."""
    seen_relationships = []
    for relationship_parameters in dayz_parameters.get('relationships', []):
        relationship = {
            key: value
            for key, value in relationship_parameters.items()
            if key in REQUIRED_RELATIONSHIP_KEYS
        }
        if relationship not in metadata.relationships:
            msg = (
                'Invalid relationship parameter: '
                f'relationship {relationship} does not exist in the metadata.'
            )
            raise SynthesizerProcessingError(msg)
        elif relationship in seen_relationships:
            msg = (
                'Invalid relationship parameter: '
                f'multiple entries for relationship {relationship} in parameters.'
            )
            raise SynthesizerProcessingError(msg)

        seen_relationships.append(relationship)

        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        parent_num_rows = dayz_parameters.get('tables', {}).get(parent_table, {}).get('num_rows')
        child_num_rows = dayz_parameters.get('tables', {}).get(child_table, {}).get('num_rows')
        _validate_cardinality(relationship_parameters, parent_num_rows, child_num_rows)


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
    _validate_relationship_structure(parameters)
    _validate_tables_parameter(metadata, parameters)
    _validate_relationship_parameters(metadata, parameters)


class DayZSynthesizer:
    """Multi-Table DayZSynthesizer for public SDV."""

    def __init__(self, metadata, locales=['en_US']):
        raise SynthesizerInputError(
            "Only the 'DayZSynthesizer.create_parameters' and the "
            'DayZSynthesizer.validate_parameters methods are an SDV public feature. To '
            'define and use a DayZSynthesizer object you must have SDV-Enterprise.'
        )

    @classmethod
    def create_parameters(cls, data, metadata, filepath=None):
        """Create parameters for the DayZSynthesizer.

        Args:
            data (dict[str, pd.DataFrame]): The input data.
            metadata (Metadata): The metadata object.
            filepath (str, optional): The output filename for the parameters.

        Returns:
            dict: The created parameters.
        """
        return create_parameters_multi_table(data, metadata, filepath)

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
