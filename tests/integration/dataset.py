import pandas as pd

from sdv import Metadata


def load_multi_foreign_key():
    parent = pd.DataFrame({
        'parent_id': range(10),
        'value': range(10)
    })
    child = pd.DataFrame({
        'parent_1_id': range(10),
        'parent_2_id': range(10),
        'value': range(10)
    })

    metadata = Metadata()
    metadata.add_table('parent', parent, primary_key='parent_id')
    metadata.add_table('child', child, parent='parent', foreign_key='parent_1_id')
    metadata.add_relationship('parent', 'child', 'parent_2_id')

    return metadata, {'parent': parent, 'child': child}
