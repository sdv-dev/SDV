import pandas as pd

from sdv.lite import TabularPreset


def test_sample():
    """Test sampling for the ``TabularPreset``."""
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3]})

    # Run
    preset = TabularPreset(name='FAST_ML')
    preset.fit(data)
    samples = preset.sample(
        num_rows=10,
        max_tries_per_batch=20,
        batch_size=5
    )

    # Assert
    assert samples['a'].all() in [1, 2, 3]
    assert len(samples) == 10
