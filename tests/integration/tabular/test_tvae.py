import numpy as np
import pandas as pd

from sdv.demo import load_demo
from sdv.sampling import Condition
from sdv.tabular.ctgan import TVAE


def test_tvae():
    users = load_demo(metadata=False)['users']

    tvae = TVAE(
        primary_key='user_id',
        epochs=1
    )
    tvae.fit(users)

    sampled = tvae.sample(len(users))

    # test shape is right
    assert sampled.shape == users.shape

    # test user_id has been generated as an ID field
    assert list(sampled['user_id']) == list(range(0, len(users)))

    expected_metadata = {
        'fields': {
            'user_id': {
                'type': 'id',
                'subtype': 'integer',
                'transformer': 'integer',
            },
            'country': {
                'type': 'categorical',
                'transformer': None,
            },
            'gender': {
                'type': 'categorical',
                'transformer': None,
            },
            'age': {
                'type': 'numerical',
                'subtype': 'integer',
                'transformer': 'integer',
            }
        },
        'primary_key': 'user_id',
        'constraints': [],
        'sequence_index': None,
        'context_columns': [],
        'entity_columns': [],
        'model_kwargs': {},
        'name': None
    }
    assert tvae.get_metadata().to_dict() == expected_metadata


def test_recreate():
    data = load_demo(metadata=False)['users']

    # If distribution is non parametric, get_parameters fails
    model = TVAE(epochs=1)
    model.fit(data)
    sampled = model.sample(len(data))

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata
    model_meta = TVAE(epochs=1, table_metadata=model.get_metadata())
    model_meta.fit(data)
    sampled = model_meta.sample(len(data))

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata dict
    model_meta_dict = TVAE(epochs=1, table_metadata=model.get_metadata().to_dict())
    model_meta_dict.fit(data)
    sampled = model_meta_dict.sample(len(data))

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()


def test_conditional_sampling_dict():
    data = pd.DataFrame({
        'column1': [1.0, 0.5, 2.5] * 10,
        'column2': ['a', 'b', 'c'] * 10
    })

    model = TVAE(epochs=1)
    model.fit(data)
    conditions = [Condition({
        'column2': 'b'
    }, num_rows=30)]
    sampled = model.sample_conditions(conditions=conditions)

    assert sampled.shape == data.shape
    assert set(sampled['column2'].unique()) == set(['b'])


def test_conditional_sampling_dataframe():
    data = pd.DataFrame({
        'column1': [1.0, 0.5, 2.5] * 10,
        'column2': ['a', 'b', 'c'] * 10
    })

    model = TVAE(epochs=1)
    model.fit(data)
    conditions = pd.DataFrame({
        'column2': ['b', 'b', 'b', 'c', 'c']
    })
    sampled = model.sample_remaining_columns(conditions)

    assert sampled.shape[0] == len(conditions['column2'])
    assert (sampled['column2'] == np.array(['b', 'b', 'b', 'c', 'c'])).all()


def test_conditional_sampling_two_conditions():
    data = pd.DataFrame({
        'column1': [1.0, 0.5, 2.5] * 10,
        'column2': ['a', 'b', 'c'] * 10,
        'column3': ['d', 'e', 'f'] * 10
    })

    model = TVAE(epochs=1)
    model.fit(data)
    conditions = [Condition({
        'column2': 'b',
        'column3': 'f'
    }, num_rows=5)]
    samples = model.sample_conditions(conditions=conditions, max_tries_per_batch=200)
    assert list(samples.column2) == ['b'] * 5
    assert list(samples.column3) == ['f'] * 5


def test_conditional_sampling_numerical():
    data = pd.DataFrame({
        'column1': [1.0, 0.5, 2.5] * 10,
        'column2': ['a', 'b', 'c'] * 10,
        'column3': ['d', 'e', 'f'] * 10
    })

    model = TVAE(epochs=1)
    model.fit(data)
    conditions = [Condition({
        'column1': 1.0,
    }, num_rows=5)]
    sampled = model.sample_conditions(conditions=conditions)

    assert list(sampled.column1) == [1.0] * 5
