import re

import numpy as np
import pandas as pd
import pytest

from sdv.cag import OneHotEncoding
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


@pytest.fixture()
def data():
    return pd.DataFrame({
        'a': [1, 0, 0],
        'b': [0, 1, 0],
        'c': [0, 0, 1],
    })


@pytest.fixture()
def metadata():
    return Metadata.load_from_dict({
        'columns': {
            'a': {'sdtype': 'numerical'},
            'b': {'sdtype': 'numerical'},
            'c': {'sdtype': 'numerical'},
        }
    })


def run_synthesizer(data, metadata):
    pattern = OneHotEncoding(column_names=['a', 'b', 'c'])
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_cag(patterns=[pattern])
    synthesizer.fit(data)
    return synthesizer


def test_validate_cag(data, metadata):
    # Setup
    synthesizer = run_synthesizer(data, metadata)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    for col in ['a', 'b', 'c']:
        assert synthetic_data[col].nunique() == 2
        assert sorted(synthetic_data[col].unique().tolist()) == [0, 1]


def test_validate_cag_raises(data, metadata):
    # Setup
    synthetic_data = pd.DataFrame({
        'a': [1, 2, 0],
        'b': [0, 1, np.nan],
        'c': [0, 0, 3],
    })
    synthesizer = run_synthesizer(data, metadata)
    msg = re.escape('The one hot encoding requirement is not met for row indices: 1, 2')

    # Run and Assert
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)
