def test_truncated_gaussian_error():
    from sdv.timeseries import PAR
    from sdv.demo import load_timeseries_demo

    data = load_timeseries_demo()
    entity_columns = ['Symbol']
    context_columns = ['MarketCap', 'Sector', 'Industry']
    sequence_index = 'Date'
    field_types = {
        'Symbol': {
            'type': 'id',
            'subtype': 'string',
            'regex': '[A-Z]{2,4}'
        }
    }
    model = PAR(
        entity_columns=entity_columns,
        context_columns=context_columns,
        sequence_index=sequence_index,
        field_types=field_types
    )
    model.fit(data)