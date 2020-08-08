from sdv.demo import load_demo
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

    gc = GaussianCopula(
        field_names=['user_id', 'country', 'gender', 'age'],
        field_types=field_types,
        primary_key='user_id',
        anonymize_fields=anonymize_fields,
    )
    gc.fit(users)

    parameters = gc.get_parameters()
    new_gc = GaussianCopula(
        table_metadata=gc.get_metadata(),
    )
    new_gc.set_parameters(parameters)

    sampled = new_gc.sample()

    # test shape is right
    assert sampled.shape == users.shape

    # test user_id has been generated as an ID field
    assert list(sampled['user_id']) == list(range(0, len(users)))

    # country codes have been replaced with new ones
    assert set(sampled.country.unique()) != set(users.country.unique())

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
