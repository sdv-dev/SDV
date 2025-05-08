import re

import numpy as np
import pandas as pd
import pytest

from sdv.cag import OneHotEncoding
from sdv.cag._errors import ConstraintNotMetError
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
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


@pytest.fixture()
def constraint_multi():
    return OneHotEncoding(column_names=['a', 'b', 'c'], table_name='table1')


def run_synthesizer(data, metadata):
    constraint = OneHotEncoding(column_names=['a', 'b', 'c'])
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.add_constraints(constraints=[constraint])
    synthesizer.fit(data)
    return synthesizer


def run_hma(data, metadata, constraint):
    synthesizer = HMASynthesizer(metadata)
    synthesizer.add_constraints(constraints=[constraint])
    synthesizer.fit(data)
    return synthesizer


def test_validate_cag(data, metadata):
    """Test validate_cag works with synthetic data generated with OneHotEncoding."""
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
    """Test validate_cag raises an error with bad synthetic data with OneHotEncoding."""
    # Setup
    synthetic_data = pd.DataFrame({
        'a': [1, 2, 0],
        'b': [0, 1, np.nan],
        'c': [0, 0, 3],
    })
    synthesizer = run_synthesizer(data, metadata)
    msg = re.escape('The one hot encoding requirement is not met for row indices: 1, 2')

    # Run and Assert
    with pytest.raises(ConstraintNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)


def test_validate_cag_multi(
    data_multi,
    metadata_multi,
    constraint_multi,
):
    """Test validate_cag with synthetic data generated with OneHotEncoding with multitable data."""
    # Setup
    data = data_multi
    metadata = metadata_multi
    constraint = constraint_multi
    synthesizer = run_hma(data, metadata, constraint)
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_cag(synthetic_data=synthetic_data)

    # Assert
    for col in ['a', 'b', 'c']:
        assert synthetic_data['table1'][col].nunique() == 2
        assert sorted(synthetic_data['table1'][col].unique().tolist()) == [0, 1]


def test_validate_cag_multi_raises(
    data_multi,
    metadata_multi,
    constraint_multi,
):
    """Test validate_cag raises an error with bad multitable synthetic data with OneHotEncoding."""
    data = data_multi
    metadata = metadata_multi
    constraint = constraint_multi
    synthesizer = run_hma(data, metadata, constraint)
    synthetic_data = {
        'table1': pd.DataFrame({
            'a': [1, 2, 0],
            'b': [0, 1, np.nan],
            'c': [0, 0, 3],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }
    msg = re.escape(
        "Table 'table1': The one hot encoding requirement is not met for row indices: 1, 2."
    )

    # Run and Assert
    with pytest.raises(ConstraintNotMetError, match=msg):
        synthesizer.validate_cag(synthetic_data=synthetic_data)
