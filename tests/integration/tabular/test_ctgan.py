from sdv.demo import load_demo
from sdv.tabular.ctgan import CTGAN
import pandas as pd


def test_ctgan():
    users = load_demo(metadata=False)['users']

    ctgan = CTGAN(
        primary_key='user_id',
        epochs=1
    )
    ctgan.fit(users)

    sampled = ctgan.sample()

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
        'primary_key': 'user_id',
        'constraints': [],
        'sequence_index': None,
        'context_columns': [],
        'entity_columns': [],
        'model_kwargs': {},
        'name': None
    }
    assert ctgan.get_metadata().to_dict() == expected_metadata


def test_recreate():
    data = load_demo(metadata=False)['users']

    # If distribution is non parametric, get_parameters fails
    model = CTGAN(epochs=1)
    model.fit(data)
    sampled = model.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata
    model_meta = CTGAN(epochs=1, table_metadata=model.get_metadata())
    model_meta.fit(data)
    sampled = model_meta.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata dict
    model_meta_dict = CTGAN(epochs=1, table_metadata=model.get_metadata().to_dict())
    model_meta_dict.fit(data)
    sampled = model_meta_dict.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()


@pytest.mark.xfail(reason="not implemented")
def test_conditional_sampling_n_rows():
    data = pd.DataFrame({
        "column1": [1.0, 0.5, 2.5]*10,
        "column2": ["a", "b", "c"]*10
    })

    model = CTGAN(epochs=1)
    model.fit(data)
    conditions = {
        "column2": "b"
    }
    sampled = model.sample(30, conditions=conditions)

    assert sampled.shape == data.shape
    assert sampled["column2"].unique() == ["b"]


@pytest.mark.xfail(reason="not implemented")
def test_conditional_sampling_two_rows():
    data = pd.DataFrame({
        "column1": [1.0, 0.5, 2.5]*10,
        "column2": ["a", "b", "c"]*10
    })

    model = CTGAN(epochs=1)
    model.fit(data)
    conditions = {
        "column2": ["b", "c"]
    }
    sampled = model.sample(conditions=conditions)

    assert sampled.shape[0] == 2
    assert sampled["column2"].unique() == ["b", "c"]


@pytest.mark.xfail(reason="not implemented")
def test_conditional_sampling_two_conditions():
    data = pd.DataFrame({
        "column1": [1.0, 0.5, 2.5]*10,
        "column2": ["a", "b", "c"]*10,
        "column3": ["d", "e", "f"]*10
    })

    model = CTGAN(epochs=1)
    model.fit(data)
    conditions = {
        "column2": ["b"],
        "column3": ["f"]
    }
    sampled = model.sample(30, conditions=conditions)

    assert sampled.shape == data.shape
    assert sampled["column2"].unique() == ["b"]
    assert sampled["column3"].unique() == ["f"]