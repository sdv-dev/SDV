import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


def test_backward_compatibility_old_style_constraints(tmpdir):
    """Test that the old-style constraints are still supported."""
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
    })

    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('A', 'table', sdtype='numerical')
    metadata.add_column('B', 'table', sdtype='numerical')

    synthesizer = GaussianCopulaSynthesizer(metadata)
    constraint = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'A',
            'high_column_name': 'B',
            'strict_boundaries': False,
        },
    }
    synthesizer._data_processor._constraints = [constraint]

    # Run
    synthesizer.fit(data)
    samples = synthesizer.sample(len(data))
    synthesizer.save(tmpdir / 'test.pkl')
    synthesizer_loaded = GaussianCopulaSynthesizer.load(tmpdir / 'test.pkl')
    samples_loaded = synthesizer_loaded.sample(len(data))

    # Assert
    assert all(samples['A'] < samples['B'])
    assert all(samples_loaded['A'] < samples_loaded['B'])
