from sdv.metadata.metadata_upgrader import convert_metadata


def test_convert_metadata():
    """Test the ``convert_metadata`` method.

    The method should take a dictionary of the old metadata format and convert it to the new
    format.

    Input:
        - Dictionary of single table metadata in the old schema.

    Output:
        - Dictionary of the same metadata with the new schema.
    """
    # Setup
    old_metadata = {
        'fields': {
            'start_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
            'end_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
            'salary': {'type': 'numerical', 'subtype': 'integer'},
            'duration': {'type': 'categorical'},
            'student_id': {'type': 'id', 'subtype': 'integer'},
            'high_perc': {'type': 'numerical', 'subtype': 'float'},
            'placed': {'type': 'boolean'},
            'ssn': {'type': 'categorical', 'pii': True, 'pii_category': 'ssn'},
            'credit_card': {
                'type': 'categorical',
                'pii': True,
                'pii_category': ['credit_card_number', 'visa'],
            },
            'drivers_license': {'type': 'id', 'subtype': 'string', 'regex': 'regex'},
        },
        'primary_key': 'student_id',
    }

    # Run
    new_metadata = convert_metadata(old_metadata)

    # Assert
    expected_metadata = {
        'columns': {
            'start_date': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'end_date': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'salary': {'sdtype': 'numerical'},
            'duration': {'sdtype': 'categorical'},
            'student_id': {'sdtype': 'id', 'regex_format': r'\d{30}'},
            'high_perc': {'sdtype': 'numerical'},
            'placed': {'sdtype': 'boolean'},
            'ssn': {'sdtype': 'ssn', 'pii': True},
            'credit_card': {'sdtype': 'credit_card_number', 'pii': True},
            'drivers_license': {'sdtype': 'id', 'regex_format': 'regex'},
        },
        'primary_key': 'student_id',
        'alternate_keys': ['drivers_license'],
    }
    assert new_metadata == expected_metadata
