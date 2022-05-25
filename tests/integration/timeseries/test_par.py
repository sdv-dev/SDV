import datetime

import pandas as pd
from deepecho import load_demo

from sdv.timeseries.deepecho import PAR


def test_par():
    data = load_demo()
    data['date'] = pd.to_datetime(data['date'])

    model = PAR(
        entity_columns=['store_id'],
        context_columns=['region'],
        sequence_index='date',
        epochs=1,
    )
    model.fit(data)

    sampled = model.sample(100)

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata
    model_meta = PAR(
        table_metadata=model.get_metadata(),
        epochs=1,
    )
    model_meta.fit(data)

    sampled = model_meta.sample(100)

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata dict
    model_meta_dict = PAR(
        table_metadata=model.get_metadata().to_dict(),
        epochs=1,
    )
    model_meta_dict.fit(data)

    sampled = model_meta_dict.sample(100)

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()


def test_column_after_date_simple():
    """Test that adding a column after the `sequence_index` column works."""
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'col': ['a', 'a'],
        'date': [date, date],
        'col2': ['hello', 'world'],
    })

    model = PAR(entity_columns=['col'], sequence_index='date', epochs=1)
    model.fit(data)
    sampled = model.sample(1)

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()


def test_column_after_date_complex():
    """Test that adding multiple columns after the `sequence_index` column works."""
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'column1': [1.0, 2.0, 1.5, 1.3],
        'date': [date, date, date, date],
        'column2': ['b', 'a', 'a', 'c'],
        'entity': ['person1', 'person1', 'person2', 'person2'],
        'context': ['a', 'a', 'b', 'b']
    })

    model = PAR(entity_columns=['entity'], context_columns=['context'], sequence_index='date',
                epochs=1)
    model.fit(data)
    sampled = model.sample(2)

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()
