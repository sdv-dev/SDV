import numpy as np
import pandas as pd

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.utils import detect_discrete_columns


def test_detect_discrete_columns():
    """Test that the detect discrete columns returns a list columns that are not continuum."""
    # Setup
    metadata = SingleTableMetadata()
    metadata._columns = {
        'name': {
            'sdtype': 'categorical',
        },
        'age': {
            'sdtype': 'numerical',
        },
        'subscribed': {
            'sdtype': 'boolean',
        },
        'join_date': {
            'sdtype': 'datetime'
        }
    }
    data = pd.DataFrame({
        'name': ['John', 'Doe', 'John Doe', 'John Doe Doe'],
        'age': [1, 2, 3, 4],
        'subscribed': [None, True, False, np.nan],
        'join_date': ['2021-02-02', '2022-03-04', '2015-05-06', '2018-09-30'],
        'uses_synthetic': [np.nan, True, False, False],
        'surname.value': [object(), object(), object(), object()]
    })

    # Run
    result = detect_discrete_columns(metadata, data)

    # Assert
    assert result == ['name', 'subscribed', 'uses_synthetic', 'surname.value']
