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
        categorical_transformer='one_hot_encoding',
    )
    gc.fit(users)

    sampled = gc.sample()

    # test shape is right
    assert sampled.shape == users.shape

    # test user_id has been generated as an ID field
    assert list(sampled['user_id']) == list(range(0, len(users)))

    # country codes have been replaced with new ones
    assert set(sampled.country.unique()) & set(users.country.unique()) == set()

    assert gc.get_metadata().to_dict() == {
        'fields': {
            'user_id': {'type': 'id', 'subtype': 'integer'},
            'country': {'type': 'categorical'},
            'gender': {'type': 'categorical'},
            'age': {'type': 'numerical', 'subtype': 'integer'}
        }
    }
