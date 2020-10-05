from sdv.demo import load_demo
from sdv.tabular.copulagan import CopulaGAN


def test_copulagan():
    users = load_demo(metadata=False)['users']

    model = CopulaGAN(
        primary_key='user_id',
        epochs=1,
        field_distributions={
            'age': 'beta'
        },
        default_distribution='bounded'
    )
    model.fit(users)

    sampled = model.sample()

    # test shape is right
    assert sampled.shape == users.shape

    # test user_id has been generated as an ID field
    assert list(sampled['user_id']) == list(range(0, len(users)))

    assert model.get_metadata().to_dict() == {
        'fields': {
            'user_id': {
                'type': 'id',
                'subtype': 'integer',
                'transformer': 'integer',
            },
            'country': {
                'type': 'categorical',
                'transformer': 'label_encoding',
            },
            'gender': {
                'type': 'categorical',
                'transformer': 'label_encoding',
            },
            'age': {
                'type': 'numerical',
                'subtype': 'integer',
                'transformer': 'integer',
            }
        },
        'constraints': [],
        'model_kwargs': {},
        'name': None
    }
