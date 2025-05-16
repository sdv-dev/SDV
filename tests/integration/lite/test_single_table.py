import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.skip('Old-style constraints are deprecated')
def test_sample_with_constraints():
    """Test sampling for the ``SingleTablePreset``."""
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    # Run
    metadata = Metadata.detect_from_dataframes({'table': data})
    preset = SingleTablePreset(metadata, name='FAST_ML')
    constraints = [
        {
            'constraint_class': 'Inequality',
            'constraint_parameters': {'low_column_name': 'a', 'high_column_name': 'b'},
        }
    ]
    preset.add_constraints(constraints)
    preset.fit(data)
    samples = preset.sample(num_rows=10, max_tries_per_batch=20, batch_size=5)

    # Assert
    assert len(samples) == 10
    assert all(samples['a'] < samples['b'])


@pytest.mark.skip('Old-style constraints are deprecated')
def test_warnings_are_shown():
    """Test all actions with SingleTablePreset gives a FutureWarning"""
    warn_message = (
        "The 'SingleTablePreset' is deprecated. For equivalent Fast ML "
        "functionality, please use the 'GaussianCopulaSynthesizer'."
    )
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    # Run
    metadata = Metadata.detect_from_dataframes({'table': data})

    with pytest.warns(FutureWarning, match=warn_message):
        preset = SingleTablePreset(metadata, name='FAST_ML')

    constraints = [
        {
            'constraint_class': 'Inequality',
            'constraint_parameters': {'low_column_name': 'a', 'high_column_name': 'b'},
        }
    ]
    with pytest.warns(FutureWarning, match=warn_message):
        preset.add_constraints(constraints)

    with pytest.warns(FutureWarning, match=warn_message):
        preset.fit(data)

    with pytest.warns(FutureWarning, match=warn_message):
        samples = preset.sample(num_rows=10, max_tries_per_batch=20, batch_size=5)

    # Assert
    assert len(samples) == 10
    assert all(samples['a'] < samples['b'])
