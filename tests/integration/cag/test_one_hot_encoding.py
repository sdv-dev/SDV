import re

import numpy as np
import pandas as pd
import pytest

from sdv.cag import OneHotEncoding
from sdv.cag._errors import PatternNotMetError
from sdv.metadata import Metadata
from tests.utils import run_copula, run_hma


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


@pytest.fixture()
def data_multi(data):
    return {
        'table1': data,
        'table2': pd.DataFrame({'id': range(5)}),
    }


@pytest.fixture()
def metadata_multi():
    return Metadata.load_from_dict({
        'tables': {
            'table1': {
                'columns': {
                    'a': {'sdtype': 'numerical'},
                    'b': {'sdtype': 'numerical'},
                    'c': {'sdtype': 'numerical'},
                }
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                }
            },
        }
    })


def test_end_to_end(data, metadata):
    """Test end to end with OneHotEncoding."""
    # Setup
    synthesizer = run_copula(data, metadata, [OneHotEncoding(column_names=['a', 'b', 'c'])])
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    for col in ['a', 'b', 'c']:
        assert synthetic_data[col].nunique() == 2
        assert sorted(synthetic_data[col].unique().tolist()) == [0, 1]


def test_end_to_end_raises(data, metadata):
    """Test end to end raises an error with bad synthetic data with OneHotEncoding."""
    # Setup
    invalid_data = pd.DataFrame({
        'a': [1, 2, 0],
        'b': [0, 1, np.nan],
        'c': [0, 0, 3],
    })

    # Run and Assert
    msg = re.escape('The one hot encoding requirement is not met for row indices: [1, 2]')
    with pytest.raises(PatternNotMetError, match=msg):
        run_copula(invalid_data, metadata, [OneHotEncoding(column_names=['a', 'b', 'c'])])

    msg = re.escape('The one hot encoding requirement is not met for row indices: 1, 2')
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer = run_copula(data, metadata, [OneHotEncoding(column_names=['a', 'b', 'c'])])
        synthesizer.validate_cag(synthetic_data=invalid_data)


def test_end_to_end_multi(data_multi, metadata_multi):
    """Test end to end with OneHotEncoding with multitable data."""
    # Setup
    constraint = OneHotEncoding(column_names=['a', 'b', 'c'], table_name='table1')
    synthesizer = run_hma(data_multi, metadata_multi, [constraint])
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    for col in ['a', 'b', 'c']:
        assert synthetic_data['table1'][col].nunique() == 2
        assert sorted(synthetic_data['table1'][col].unique().tolist()) == [0, 1]


def test_end_to_end_multi_raises(data_multi, metadata_multi):
    """Test end to end raises an error with bad multitable synthetic data with OneHotEncoding."""
    # Setup
    constraint = OneHotEncoding(column_names=['a', 'b', 'c'], table_name='table1')
    synthesizer = run_hma(data_multi, metadata_multi, [constraint])
    invalid_data = {
        'table1': pd.DataFrame({
            'a': [1, 2, 0],
            'b': [0, 1, np.nan],
            'c': [0, 0, 3],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }

    # Run and Assert
    msg = re.escape('The one hot encoding requirement is not met for row indices: [1, 2]')
    with pytest.raises(PatternNotMetError, match=msg):
        run_hma(invalid_data, metadata_multi, [constraint])

    msg = "Table 'table1': The one hot encoding requirement is not met for row indices: 1, 2."
    with pytest.raises(PatternNotMetError, match=msg):
        synthesizer = run_hma(data_multi, metadata_multi, [constraint])
        synthesizer.validate_cag(synthetic_data=invalid_data)
