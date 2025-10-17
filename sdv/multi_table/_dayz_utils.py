import json

import pandas as pd

from sdv.single_table.dayz import create_parameters


def detect_relationship_parameters(data, metadata):
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
            'min_cardinality': int(cardinality_table['cardinality'].min()),
            'max_cardinality': int(cardinality_table['cardinality'].max()),
        })

    return relationship_parameters


def create_parameters_multi_table(data, metadata, output_filename):
    """Create parameters for the DayZSynthesizer."""
    parameters = create_parameters(data, metadata, None)
    parameters['relationships'] = detect_relationship_parameters(data, metadata)
    if output_filename:
        with open(output_filename, 'w') as f:
            json.dump(parameters, f, indent=4)

    return parameters
