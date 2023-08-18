# Setup
import pandas as pd

from sdv.constraints import create_custom_constraint_class
from sdv.metadata.multi_table import MultiTableMetadata
from sdv.multi_table.hma import HMASynthesizer


def is_valid(column_names, data):
    """Validate the constraint."""
    return pd.Series([value[0] > 1 for value in data[column_names].to_numpy()])


def transform(column_names, data):
    """Transform the constraint."""
    data[column_names] = data[column_names] ** 2
    return data


def reverse_transform(column_names, data):
    """Reverse transform the constraint."""
    data[column_names] = data[column_names] // 2
    return data


MyConstraint = create_custom_constraint_class(
    is_valid,
    transform,
    reverse_transform
)


def get_custom_constraint_data_and_metadata():
    """Return data and metadata for the custom constraint tests."""
    parent_data = pd.DataFrame({
        'primary_key': [1000, 1001, 1002],
        'numerical_col': [2, 3, 4],
        'categorical_col': ['a', 'b', 'a'],
    })
    child_data = pd.DataFrame({
        'user_id': [1000, 1001, 1000],
        'id': [1, 2, 3],
        'random': ['a', 'b', 'c'],
        'numerical_col': [0.2, 0.7, 1.3],
        'numerical_col_2': [2, 4, 6],
    })

    metadata = MultiTableMetadata()
    metadata.detect_table_from_dataframe('parent', parent_data)
    metadata.update_column('parent', 'primary_key', sdtype='id')
    metadata.detect_table_from_dataframe('child', child_data)
    metadata.update_column('child', 'user_id', sdtype='id')
    metadata.update_column('child', 'id', sdtype='id')
    metadata.set_primary_key('parent', 'primary_key')
    metadata.set_primary_key('child', 'id')
    metadata.add_relationship(
        parent_primary_key='primary_key',
        parent_table_name='parent',
        child_foreign_key='user_id',
        child_table_name='child'
    )

    return parent_data, child_data, metadata


parent_data, child_data, metadata = get_custom_constraint_data_and_metadata()
synthesizer = HMASynthesizer(metadata)
constraint = {
    'table_name': 'parent',
    'constraint_class': 'MyConstraint',
    'constraint_parameters': {
        'column_names': ['numerical_col']
    }
}
synthesizer.add_custom_constraint_class(MyConstraint, 'MyConstraint')

# Run
synthesizer.add_constraints(constraints=[constraint])
processed_data = synthesizer.preprocess({'parent': parent_data, 'child': child_data})
