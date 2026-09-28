import numpy as np
import pandas as pd

from sdv.evaluation._utils import _prepare_data_visualization
from sdv.metadata import SingleTableMetadata


def test__prepare_data_visualization():
    """Test ``_prepare_data_visualization``."""
    # Setup
    np.random.seed(0)
    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'col1': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'col2': {'sdtype': 'numerical'},
        }
    })
    column_names = ['col1', 'col2']
    sample_size = 2
    data = pd.DataFrame({
        'col1': ['2021-01-01', '2021-02-01', '2021-03-01'],
        'col2': [4, 5, 6],
    })

    # Run
    result = _prepare_data_visualization(data, metadata, column_names, sample_size)

    # Assert
    expected_result = pd.DataFrame(
        {
            'col1': pd.to_datetime(['2021-03-01', '2021-02-01']),
            'col2': [6, 5],
        },
        index=[2, 1],
    )
    pd.testing.assert_frame_equal(result, expected_result)
