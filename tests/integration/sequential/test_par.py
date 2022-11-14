import datetime

import pandas as pd
from deepecho import load_demo

from sdv.demo import load_timeseries_demo
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer


def test_par():
    """Test the ``PARSynthesizer`` end to end."""
    # Setup
    data = load_demo()
    data['date'] = pd.to_datetime(data['date'])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.set_sequence_key('store_id')
    metadata.set_sequence_index('date')
    model = PARSynthesizer(
        metadata=metadata,
        context_columns=['region'],
        epochs=1,
    )

    # Run
    model.fit(data)
    sampled = model.sample(100)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notna().sum(axis=1) != 0).all()


def test_column_after_date_simple():
    """Test that adding a column after the ``sequence_index`` column works."""
    # Setup
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'col': [1, 1],
        'date': [date, date],
        'col2': ['hello', 'world'],
    })
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.set_sequence_key('col')
    metadata.set_sequence_index('date')

    # Run
    model = PARSynthesizer(metadata=metadata, epochs=1)
    model.fit(data)
    sampled = model.sample(1)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notna().sum(axis=1) != 0).all()


def test_column_after_date_complex():
    """Test that adding multiple columns after the ``sequence_index`` column works."""
    # Setup
    date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    data = pd.DataFrame({
        'column1': [1.0, 2.0, 1.5, 1.3],
        'date': [date, date, date, date],
        'column2': ['b', 'a', 'a', 'c'],
        'entity': [1, 1, 2, 2],
        'context': ['a', 'a', 'b', 'b']
    })
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.set_sequence_key('entity')
    metadata.set_sequence_index('date')

    # Run
    model = PARSynthesizer(metadata=metadata, context_columns=['context'], epochs=1)
    model.fit(data)
    sampled = model.sample(2)

    # Assert
    assert sampled.shape == data.shape
    assert (sampled.dtypes == data.dtypes).all()
    assert (sampled.notna().sum(axis=1) != 0).all()


def test_sample_sequential_columns_with_context_out_of_order():
    """Test that the context columns can be out of order when sampling."""
    # Setup
    data = load_timeseries_demo()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column('Symbol', sdtype='text')
    metadata.set_sequence_key('Symbol')
    metadata.set_sequence_index('Date')
    context = pd.DataFrame(data={
        'Symbol': ['Apple', 'Google'],
        'Sector': ['Technology', 'Health Care'],
        'MarketCap': [1.2345e+11, 4.5678e+10],
        'Industry': ['Electronic Components', 'Medical/Nursing Services']
    })
    par = PARSynthesizer(
        metadata=metadata,
        context_columns=['MarketCap', 'Sector', 'Industry'],
        epochs=1
    )

    # Run
    par.fit(data)
    sampled = par.sample_sequential_columns(context_columns=context, sequence_length=10)

    # Assert
    expected_first_sequence = pd.DataFrame({
        'Symbol': ['Apple'] * 10,
        'MarketCap': [1.2345e+11] * 10,
        'Sector': ['Technology'] * 10,
        'Industry': ['Electronic Components'] * 10
    })
    expected_second_sequence = pd.DataFrame({
        'Symbol': ['Google'] * 10,
        'MarketCap': [4.5678e+10] * 10,
        'Sector': ['Health Care'] * 10,
        'Industry': ['Medical/Nursing Services'] * 10
    }, index=range(10, 20))
    context_columns = ['Symbol', 'MarketCap', 'Sector', 'Industry']
    pd.testing.assert_frame_equal(sampled[context_columns].iloc[0:10], expected_first_sequence)
    pd.testing.assert_frame_equal(sampled[context_columns].iloc[10:], expected_second_sequence)
