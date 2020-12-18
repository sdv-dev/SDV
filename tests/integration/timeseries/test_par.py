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

    sampled = model.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata
    model_meta = PAR(
        table_metadata=model.get_metadata(),
        epochs=1,
    )
    model_meta.fit(data)

    sampled = model_meta.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()

    # Metadata dict
    model_meta_dict = PAR(
        table_metadata=model.get_metadata().to_dict(),
        epochs=1,
    )
    model_meta_dict.fit(data)

    sampled = model_meta_dict.sample()

    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notnull().sum(axis=1) != 0).all()
