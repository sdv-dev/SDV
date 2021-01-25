import numpy as np
import pandas as pd
import pytest

from sdv.demo import load_demo
from sdv.tabular.base import NonParametricError
from sdv.tabular.copulas import GaussianCopula


def test_gaussian_copula():
    users = load_demo(metadata=False)['users']

    field_types = {
        'age': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'country': {
            'type': 'categorical'
        }
    }
    anonymize_fields = {
        'country': 'country_code'
    }

    # If distribution is non parametric, get_parameters fails
    gc = GaussianCopula(
        field_names=['user_id', 'country', 'gender', 'age'],
        field_types=field_types,
        primary_key='user_id',
        anonymize_fields=anonymize_fields,
        field_distributions={'age': 'gamma'},
        default_distribution='gaussian_kde',
    )
    gc.fit(users)
    with pytest.raises(NonParametricError):
        parameters = gc.get_parameters()

    # If distribution is parametric, copula can be recreated
    gc = GaussianCopula(
        field_names=['user_id', 'country', 'gender', 'age'],
        field_types=field_types,
        primary_key='user_id',
        anonymize_fields=anonymize_fields,
        field_distributions={'age': 'gamma'},
        default_distribution='bounded',
    )
    gc.fit(users)

    parameters = gc.get_parameters()
    new_gc = GaussianCopula(
        table_metadata=gc.get_metadata(),
    )
    new_gc.set_parameters(parameters)

    # Validate sampled dat
    sampled = new_gc.sample()

    # test shape is right
    assert sampled.shape == users.shape

    # test user_id has been generated as an ID field
    assert list(sampled['user_id']) == list(range(0, len(users)))

    # country codes have been replaced with new ones
    assert set(sampled.country.unique()) != set(users.country.unique())

    # Validate metadata
    metadata = gc.get_metadata().to_dict()
    assert metadata['fields'] == {
        'user_id': {
            'type': 'id',
            'subtype': 'integer',
            'transformer': 'integer',
        },
        'country': {
            'type': 'categorical',
            'pii': True,
            'pii_category': 'country_code',
            'transformer': 'one_hot_encoding',
        },
        'gender': {
            'type': 'categorical',
            'transformer': 'one_hot_encoding',
        },
        'age': {
            'type': 'numerical',
            'subtype': 'integer',
            'transformer': 'integer',
        }
    }

    assert 'model_kwargs' in metadata
    assert 'GaussianCopula' in metadata['model_kwargs']


def test_integer_categoricals():
    """Ensure integer categoricals are still sampled as integers.

    The origin of this tests can be found in the github issue #194:
    https://github.com/sdv-dev/SDV/issues/194
    """
    users = load_demo(metadata=False)['users']

    field_types = {
        'age': {
            'type': 'categorical',
        },
    }
    gc = GaussianCopula(field_types=field_types, categorical_transformer='categorical')
    gc.fit(users)

    sampled = gc.sample()

    assert users['age'].dtype == np.int64
    assert sampled['age'].dtype == np.int64


def test_parameters():
    gc = GaussianCopula(
        field_distributions={'foo': 'beta'},
        default_distribution='gaussian_kde',
        categorical_transformer='label_encoding'
    )
    new_gc = GaussianCopula(
        table_metadata=gc.get_metadata().to_dict()
    )

    assert new_gc._metadata._dtype_transformers['O'] == 'label_encoding'


def test_recreate():
    data = load_demo(metadata=False)['users']

    # If distribution is non parametric, get_parameters fails
    model = GaussianCopula()
    model.fit(data)
    sampled = model.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata
    model_meta = GaussianCopula(table_metadata=model.get_metadata())
    model_meta.fit(data)
    sampled = model_meta.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata dict
    model_meta_dict = GaussianCopula(table_metadata=model.get_metadata().to_dict())
    model_meta_dict.fit(data)
    sampled = model_meta_dict.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()


def test_ids_only():
    """Ensure that tables that do not contain anything other than id fields can be modeled."""
    ids_only = pd.DataFrame({
        'id': range(10),
        'other_id': range(10),
    })

    model = GaussianCopula(field_types={
        'id': {
            'type': 'id'
        },
        'other_id': {
            'type': 'id'
        }
    })
    model.fit(ids_only)
    sampled = model.sample()

    assert sampled.shape == ids_only.shape
    assert ids_only.equals(sampled)
