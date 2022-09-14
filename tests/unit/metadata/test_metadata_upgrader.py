from sdv.metadata.metadata_upgrader import _upgrade_constraints, convert_metadata


def test__upgrade_constraints_no_constraints():
    """Test the ``_upgrade_constraints`` method with no constraints.

    Input:
        - Metadata dict with no constraints.

    Output:
        - None.
    """
    # Setup
    old_metadata = {'fields': {}}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    assert new_constraints is None


def test__upgrade_constraints_no_upgrade_needed():
    """Test the ``_upgrade_constraints`` method when new constraints are already used.

    Input:
        - Old metadata with new constraints.

    Output:
        - Same as input.
    """
    # Setup
    old_constraints = [
        {
            'constraint_name': 'OneHotEncoding',
            'column_names': ['a', 'b']
        },
        {
            'constraint_name': 'Unique',
            'column_names': ['c', 'd']
        },
    ]
    old_metadata = {'constraints': old_constraints}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    old_constraints == new_constraints


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
            'start_date': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
            'end_date': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
            'salary': {
                'type': 'numerical',
                'subtype': 'integer'
            },
            'duration': {
                'type': 'categorical'
            },
            'student_id': {
                'type': 'id',
                'subtype': 'integer'
            },
            'high_perc': {
                'type': 'numerical',
                'subtype': 'float'
            },
            'placed': {
                'type': 'boolean'
            },
            'ssn': {
                'type': 'id',
                'subtype': 'integer'
            },
            'drivers_license': {
                'type': 'id',
                'subtype': 'string',
                'regex': 'regex'
            }
        },
        'primary_key': 'student_id'
    }

    # Run
    new_metadata = convert_metadata(old_metadata)

    # Assert
    expected_metadata = {
        'columns': {
            'start_date': {
                'sdtype': 'datetime',
                'datetime_format': '%Y-%m-%d'
            },
            'end_date': {
                'sdtype': 'datetime',
                'datetime_format': '%Y-%m-%d'
            },
            'salary': {
                'sdtype': 'numerical',
                'representation': 'int64'
            },
            'duration': {
                'sdtype': 'categorical'
            },
            'student_id': {
                'sdtype': 'numerical'
            },
            'high_perc': {
                'sdtype': 'numerical',
                'representation': 'float64'
            },
            'placed': {
                'sdtype': 'boolean'
            },
            'ssn': {
                'sdtype': 'numerical'
            },
            'drivers_license': {
                'sdtype': 'text',
                'regex_format': 'regex'
            }
        },
        'primary_key': 'student_id',
        'alternate_keys': ['ssn', 'drivers_license']
    }
    assert new_metadata == expected_metadata
