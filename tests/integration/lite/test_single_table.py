import numpy as np
import pandas as pd

from sdv.lite import SingleTablePreset
from sdv.metadata.metadata import Metadata


def test_sample():
    """Test sampling for the ``SingleTablePreset``."""
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3, np.nan]})

    # Run
    metadata = Metadata.detect_from_dataframes({'adult': data})
    preset = SingleTablePreset(metadata, name='FAST_ML')
    preset.fit(data)
    samples = preset.sample(num_rows=10, max_tries_per_batch=20, batch_size=5)

    # Assert
    assert samples['a'].all() in [1, 2, 3, np.nan]
    assert len(samples) == 10
