import re

import numpy as np
import pandas as pd
import pytest

from sdv.cag import OneHotEncoding
from sdv.cag._errors import ConstraintNotMetError
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from tests.utils import run_constraint, run_copula, run_hma


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
    synthesizer.validate_constraints(synthetic_data=synthetic_data)

    # Assert
    assert (synthetic_data.sum(axis=1) == 1).all()
    for col in ['a', 'b', 'c']:
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
    msg = re.escape(
        "Data is not valid for the 'OneHotEncoding' constraint in table 'table':\n"
        '   a    b  c\n1  2  1.0  0\n2  0  NaN  3'
    )
    with pytest.raises(ConstraintNotMetError, match=msg):
        run_copula(invalid_data, metadata, [OneHotEncoding(column_names=['a', 'b', 'c'])])

    msg = re.escape('The one hot encoding requirement is not met for row indices: 1, 2')
    with pytest.raises(ConstraintNotMetError, match=msg):
        synthesizer = run_copula(data, metadata, [OneHotEncoding(column_names=['a', 'b', 'c'])])
        synthesizer.validate_constraints(synthetic_data=invalid_data)


def test_end_to_end_multi(data_multi, metadata_multi):
    """Test end to end with OneHotEncoding with multitable data."""
    # Setup
    constraint = OneHotEncoding(column_names=['a', 'b', 'c'], table_name='table1')
    synthesizer = run_hma(data_multi, metadata_multi, [constraint])
    synthetic_data = synthesizer.sample(100)

    # Run
    synthesizer.validate_constraints(synthetic_data=synthetic_data)

    # Assert
    assert (synthetic_data['table1'].sum(axis=1) == 1).all()
    for col in ['a', 'b', 'c']:
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
    msg = re.escape(
        "Data is not valid for the 'OneHotEncoding' constraint in table 'table1':\n   "
        'a    b  c\n1  2  1.0  0\n2  0  NaN  3'
    )
    with pytest.raises(ConstraintNotMetError, match=msg):
        run_hma(invalid_data, metadata_multi, [constraint])

    msg = "Table 'table1': The one hot encoding requirement is not met for row indices: 1, 2."
    with pytest.raises(ConstraintNotMetError, match=msg):
        synthesizer = run_hma(data_multi, metadata_multi, [constraint])
        synthesizer.validate_constraints(synthetic_data=invalid_data)


def test_end_to_end_numerical_and_categorical():
    """Test end to end with OneHotEncoding with numerical and categorical data."""
    # Setup one hot data
    categories = ['A', 'B', 'C']
    num_rows = 1000
    rng = np.random.default_rng(42)
    probabilities = [0.8, 0.15, 0.05]
    choices = rng.choice(len(categories), size=num_rows, p=probabilities)
    data = np.zeros((num_rows, len(categories)), dtype=int)
    data[np.arange(num_rows), choices] = 1
    columns = [f'cat_{c}' for c in categories]
    df = pd.DataFrame(data, columns=columns)

    # Setup metadata
    metadata = Metadata.detect_from_dataframe(df, table_name='one_hot')
    for sdtype in ['numerical', 'categorical']:
        metadata.update_columns(columns, sdtype=sdtype)
        synthesizer = GaussianCopulaSynthesizer(metadata)
        constraint = OneHotEncoding(column_names=columns)

        # Run
        synthesizer.add_constraints([constraint])
        synthesizer.fit(df)
        samples = synthesizer.sample(100)

        # Assert
        assert (samples.sum(axis=1) == 1).all()
        for col in columns:
            assert sorted(samples[col].unique().tolist()) == [0, 1]


def test_end_to_end_boolean():
    """Test end to end with OneHotEncoding with boolean data."""
    # Setup one hot data
    categories = ['A', 'B', 'C']
    num_rows = 1000
    rng = np.random.default_rng(42)
    probabilities = [0.8, 0.15, 0.05]
    choices = rng.choice(len(categories), size=num_rows, p=probabilities)
    data = np.zeros((num_rows, len(categories)), dtype=bool)
    data[np.arange(num_rows), choices] = True
    columns = [f'cat_{c}' for c in categories]
    df = pd.DataFrame(data, columns=columns)

    # Setup metadata
    metadata = Metadata.detect_from_dataframe(df, table_name='one_hot')
    metadata.update_columns(columns, sdtype='boolean')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    constraint = OneHotEncoding(column_names=columns)

    # Run
    synthesizer.add_constraints([constraint])
    synthesizer.fit(df)
    samples = synthesizer.sample(100)

    # Assert
    assert samples.dtypes.tolist() == [bool, bool, bool]
    assert (samples.sum(axis=1) == 1).all()
    for col in columns:
        assert sorted(samples[col].unique().tolist()) == [0, 1]


def test_end_to_end_categorical_single(data, metadata):
    """End-to-end with learning_strategy='categorical' for single-table data."""
    # Setup
    constraint = OneHotEncoding(column_names=['a', 'b', 'c'], learning_strategy='categorical')

    # Run
    synthesizer = run_copula(data, metadata, [constraint])
    synthetic_data = synthesizer.sample(200)
    synthesizer.validate_constraints(synthetic_data=synthetic_data)

    # Assert
    assert set(synthetic_data.columns) == {'a', 'b', 'c'}
    for col in ['a', 'b', 'c']:
        assert set(synthetic_data[col]) == {0, 1}
    assert (synthetic_data[['a', 'b', 'c']].sum(axis=1) == 1).all()


def test_end_to_end_categorical_single_raises(data, metadata):
    """Invalid synthetic data should raise with learning_strategy='categorical'."""
    # Setup
    invalid_data = pd.DataFrame({
        'a': [1, 2, 0],
        'b': [0, 1, np.nan],
        'c': [0, 0, 3],
    })
    constraint = OneHotEncoding(column_names=['a', 'b', 'c'], learning_strategy='categorical')

    # Run and Assert
    msg = re.escape(
        "Data is not valid for the 'OneHotEncoding' constraint in table 'table':\n"
        '   a    b  c\n'
        '1  2  1.0  0\n'
        '2  0  NaN  3'
    )
    with pytest.raises(ConstraintNotMetError, match=msg):
        run_copula(invalid_data, metadata, [constraint])

    # Run and Assert
    msg = re.escape('The one hot encoding requirement is not met for row indices: 1, 2')
    with pytest.raises(ConstraintNotMetError, match=msg):
        synthesizer = run_copula(data, metadata, [constraint])
        synthesizer.validate_constraints(synthetic_data=invalid_data)


def test_end_to_end_categorical_multi(data_multi, metadata_multi):
    """End-to-end with learning_strategy='categorical' for multi-table data."""
    # Setup
    constraint = OneHotEncoding(
        column_names=['a', 'b', 'c'], table_name='table1', learning_strategy='categorical'
    )

    # Run
    synthesizer = run_hma(data_multi, metadata_multi, [constraint])
    synthetic = synthesizer.sample(200)
    synthesizer.validate_constraints(synthetic_data=synthetic)

    # Assert
    assert set(synthetic['table1'].columns) == {'a', 'b', 'c'}
    for col in ['a', 'b', 'c']:
        assert set(synthetic['table1'][col]) == {0, 1}
    assert (synthetic['table1'][['a', 'b', 'c']].sum(axis=1) == 1).all()


def test_end_to_end_categorical_multi_raises(data_multi, metadata_multi):
    """Invalid multi-table synthetic data should raise with learning_strategy='categorical'."""
    # Setup
    constraint = OneHotEncoding(
        column_names=['a', 'b', 'c'], table_name='table1', learning_strategy='categorical'
    )
    invalid = {
        'table1': pd.DataFrame({
            'a': [1, 2, 0],
            'b': [0, 1, np.nan],
            'c': [0, 0, 3],
        }),
        'table2': pd.DataFrame({'id': range(5)}),
    }

    # Run and Assert
    msg = re.escape(
        "Data is not valid for the 'OneHotEncoding' constraint in table 'table1':\n   "
        'a    b  c\n1  2  1.0  0\n2  0  NaN  3'
    )
    with pytest.raises(ConstraintNotMetError, match=msg):
        run_hma(invalid, metadata_multi, [constraint])

    # Run and Assert
    msg = "Table 'table1': The one hot encoding requirement is not met for row indices: 1, 2."
    with pytest.raises(ConstraintNotMetError, match=msg):
        synthesizer = run_hma(data_multi, metadata_multi, [constraint])
        synthesizer.validate_constraints(synthetic_data=invalid)


def test_constraint_pipeline_categorical_single(data, metadata):
    """Constraint pipeline behavior for categorical strategy (single table)."""
    # Setup
    constraint = OneHotEncoding(column_names=['a', 'b', 'c'], learning_strategy='categorical')

    # Run
    updated_metadata, transformed, reverse_transformed = run_constraint(constraint, data, metadata)

    # Assert metadata
    assert updated_metadata.get_column_names() == ['a#b#c']

    # Assert transform
    assert transformed.shape[1] == 1
    assert not any(col in transformed.columns for col in ['a', 'b', 'c'])
    assert set(transformed.columns) == {'a#b#c'}

    # Assert reverse_transform
    assert set(reverse_transformed.columns) == {'a', 'b', 'c'}
    assert (reverse_transformed[['a', 'b', 'c']].sum(axis=1) == 1).all()
    assert set(reverse_transformed.columns) == {'a', 'b', 'c'}


def test_constraint_pipeline_categorical_multi(data_multi, metadata_multi):
    """Constraint pipeline behavior for categorical strategy (multi table)."""
    # Setup
    orig_cols = ['a', 'b', 'c']
    constraint = OneHotEncoding(
        column_names=orig_cols, table_name='table1', learning_strategy='categorical'
    )

    # Run
    updated_metadata, transformed, reverse_transformed = run_constraint(
        constraint, data_multi, metadata_multi
    )

    # Assert metadata
    assert updated_metadata.tables['table1'].get_column_names() == ['a#b#c']

    # Assert transform
    assert list(transformed['table1'].columns) != orig_cols
    assert transformed['table1'].shape[1] == 1
    assert list(transformed['table2'].columns) == list(data_multi['table2'].columns)

    # Assert reverse_transform
    assert set(reverse_transformed['table1'].columns) == set(orig_cols)
    assert (reverse_transformed['table1'][orig_cols].sum(axis=1) == 1).all()
