import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


def test_fixed_combinations_with_nans():
    # Setup
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 1, 2, 1],
        'B': [10, 20, 30, 10, 20, 10],
    })
    metadata = SingleTableMetadata().load_from_dict({
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'categorical'},
        }
    })

    synthesizer = GaussianCopulaSynthesizer(metadata)
    my_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {'column_names': ['A', 'B']},
    }
    synthesizer.add_constraints(constraints=[my_constraint])

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(1000)

    # Assert
    assert len(synthetic_data) == 1000
    pd.testing.assert_frame_equal(
        synthetic_data.drop_duplicates(ignore_index=True),
        data.drop_duplicates(ignore_index=True),
        check_like=True,
    )
