import numpy as np
import pandas as pd

from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata


def test_sample():
    """Test sampling for the ``SingleTablePreset``."""
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3, np.nan]})

    # Run
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    preset = SingleTablePreset(metadata, name='FAST_ML')
    preset.fit(data)
    samples = preset.sample(
        num_rows=10,
        max_tries_per_batch=20,
        batch_size=5
    )

    # Assert
    assert samples['a'].all() in [1, 2, 3, np.nan]
    assert len(samples) == 10


test_sample()


def test_sample_with_constraints():
    """Test sampling for the ``SingleTablePreset``."""
    # Setup
    data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })

    # Run
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.add_constraint('Inequality', low_column_name='a', high_column_name='b')
    preset = SingleTablePreset(metadata, name='FAST_ML')
    preset.fit(data)
    samples = preset.sample(
        num_rows=10,
        max_tries_per_batch=20,
        batch_size=5
    )

    # Assert
    assert len(samples) == 10
    assert all(samples['a'] < samples['b'])
