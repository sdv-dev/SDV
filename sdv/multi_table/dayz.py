"""Multi-table DayZSynthesizer."""

from sdv.errors import SynthesizerProcessingError
from sdv.single_table.dayz import _validate_parameter_structure, _validate_parameters

REQUIRED_RELATIONSHIP_KEYS = [
    'parent_table_name',
    'child_table_name',
    'parent_primary_key',
    'child_foreign_key',
]
RELATIONSHIP_PARAMETER_KEYS = REQUIRED_RELATIONSHIP_KEYS + [
    'min_cardinality',
    'cardinality',
]

DEFAULT_NUM_ROWS = 1000


def _validate_relationship_structure(dayz_parameters):
    if not isinstance(dayz_parameters.get('relationships', []), list):
        raise SynthesizerProcessingError("The 'relationships' key must be a list.")

    for relationship in dayz_parameters.get('relationships', []):
        unknown_relationship_parameters = relationship.keys() - set(RELATIONSHIP_PARAMETER_KEYS)
        if unknown_relationship_parameters:
            unknown_relationship_parameters = "', '".join(unknown_relationship_parameters)
            msg = f"Relationship contains unexpected key(s) '{unknown_relationship_parameters}'."
            raise SynthesizerProcessingError(msg)
        missing_relationship_parameters = set(REQUIRED_RELATIONSHIP_KEYS) - relationship.keys()
        if missing_relationship_parameters:
            missing_relationship_parameters = "', '".join(missing_relationship_parameters)
            msg = f"Relationship missing required key(s) '{missing_relationship_parameters}'."
            raise SynthesizerProcessingError(msg)


def _validate_cardinality(relationship_parameters, parent_num_rows, child_num_rows):
    """Validate that the relationship cardinality works with the set number of rows."""
    parent_num_rows = parent_num_rows or DEFAULT_NUM_ROWS
    child_num_rows = child_num_rows or DEFAULT_NUM_ROWS
    min_cardinality = relationship_parameters.get('min_cardinality', 0)
    max_cardinality = relationship_parameters.get('max_cardinality')

    min_parent_size = min_cardinality * child_num_rows
    max_parent_size = max_cardinality * child_num_rows if max_cardinality else None

    if parent_num_rows < min_parent_size:
        msg = (
            f'Invalid cardinality for relationship {relationship_parameters}. '
            f'Minimum cardinality requires parent table to be at least {min_parent_size} rows.'
        )
        raise SynthesizerProcessingError(msg)
    if max_parent_size and parent_num_rows > max_parent_size:
        msg = (
            f'Invalid cardinality for relationship {relationship_parameters}. '
            f'Maximum cardinality requires parent table to be less than {max_parent_size} rows.'
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
            msg = f'Relationship {relationship} does not exist in the metadata.'
            raise SynthesizerProcessingError(msg)
        elif relationship in seen_relationships:
            msg = f'Multiple entries for relationship {relationship} in parameters.'

        seen_relationships.append(relationship)

        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        parent_num_rows = dayz_parameters.get('tables', {}).get(parent_table, {}).get('num_rows')
        child_num_rows = dayz_parameters.get('tables', {}).get(child_table, {}).get('num_rows')
        _validate_cardinality(relationship_parameters, parent_num_rows, child_num_rows)


class DayZSynthesizer:
    """Multi-Table DayZSynthesizer for public SDV."""

    @staticmethod
    def validate_parameters(metadata, my_parameters):
        """Validate a DayZSynthesizer parameters dictionary.

        Args:
            metadata (sdv.Metadata):
                Metadata for the data.
            my_parameters (dict):
                The DayZ parameter dictionary.
        """
        metadata.validate()
        _validate_parameter_structure(my_parameters)
        _validate_relationship_structure(my_parameters)
        _validate_parameters(metadata, my_parameters)
        _validate_relationship_parameters(metadata, my_parameters)
