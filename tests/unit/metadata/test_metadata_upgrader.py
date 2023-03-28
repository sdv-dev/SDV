from unittest.mock import call, patch

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
    assert old_constraints == new_constraints


def test__upgrade_constraints_greater_than():
    """Test the ``_upgrade_constraints`` method with ``GreaterThan`` constraints.

    Input:
        - Old metadata dict with the following constraints:
        - ``GreaterThan`` with ``scalar`` set to None.
        - ``GreaterThan`` with ``scalar`` set to 'high'.
        - ``GreaterThan`` with ``scalar`` set to 'high' and multiple columns.
        - ``GreaterThan`` with ``scalar`` set to 'low'.
        - ``GreaterThan`` with ``scalar`` set to 'low' and multiple columns.

    Ouput:
        - Metadata with the following constraints:
        - ``Inequality``.
        - ``ScalarInequality`` constraints with the value as low and relation as '>'.
        - ``ScalarInequality`` constraints with the value as high and relation as '>'.
    """
    # Setup
    old_constraints = [
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': None,
            'high': 'a',
            'low': 'b',
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': None,
            'high': ['c'],
            'low': 'd',
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': None,
            'high': 'e',
            'low': ['f'],
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': 'low',
            'high': 'a',
            'low': 10
        },
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': 'low',
            'high': ['b', 'c'],
            'low': 5
        },
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': 'high',
            'high': 10,
            'low': 'f'
        },
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': 'high',
            'high': 5,
            'low': ['c', 'd']
        }
    ]
    old_metadata = {'constraints': old_constraints}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    expected_constraints = [
        {
            'constraint_name': 'Inequality',
            'high_column_name': 'a',
            'low_column_name': 'b',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Inequality',
            'high_column_name': 'c',
            'low_column_name': 'd',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Inequality',
            'high_column_name': 'e',
            'low_column_name': 'f',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'a',
            'relation': '>=',
            'value': 10
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'b',
            'relation': '>=',
            'value': 5
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'c',
            'relation': '>=',
            'value': 5
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'f',
            'relation': '<=',
            'value': 10
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'c',
            'relation': '<=',
            'value': 5
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'd',
            'relation': '<=',
            'value': 5
        }
    ]

    assert len(expected_constraints) == len(new_constraints)
    for constraint in expected_constraints:
        assert constraint in new_constraints


@patch('sdv.metadata.metadata_upgrader.warnings')
def test__upgrade_constraints_greater_than_error(warnings_mock):
    """Test the ``_upgrade_constraints`` method with ``GreaterThan`` constraints.

    Input:
        - Old metadata dict with the following constraints:
        - ``GreaterThan`` with one of high or low being a list of multiple strings.

    Ouput:
        - Empty list.

    Side effect:
        - Warnings should be called.
    """
    # Setup
    old_constraints = [
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': None,
            'high': ['a', 'b'],
            'low': 'c',
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.GreaterThan',
            'scalar': None,
            'high': 'a',
            'low': ['b', 'c'],
            'strict': True
        }
    ]
    old_metadata = {'constraints': old_constraints}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    assert new_constraints == []
    warnings_mock.warn.assert_has_calls([
        call(
            "Unable to upgrade the GreaterThan constraint specified for 'high' ['a', 'b'] "
            "and 'low' 'c'. Manually add Inequality constraints to capture this logic."
        ),
        call(
            "Unable to upgrade the GreaterThan constraint specified for 'high' 'a' "
            "and 'low' ['b', 'c']. Manually add Inequality constraints to capture this logic."
        )
    ])


def test__upgrade_constraints_between():
    """Test the ``_upgrade_constraints`` method with ``Between`` constraints.

    Input:
        - Old metadata dict with the following constraints:
        - ``Between`` constraint with 'high_is_scalar' and 'low_is_scalar' set to True.
        - ``Between`` constraint with 'high_is_scalar' and 'low_is_scalar' set to False.
        - ``Between`` constraint with 'high_is_scalar' as True and 'low_is_scalar' as False.
        - ``Between`` constraint with 'high_is_scalar' as False and 'low_is_scalar' as True.

    Output:
        - Metadata with the following constraints:
        - ``ScalarRange``.
        - ``Range``.
        - ``Inequality`` paired with a ``ScalarInequality`` with relation set to '<='.
        - ``Inequality`` paired with a ``ScalarInequality`` with relation set to '>='.
    """
    # Setup
    old_constraints = [
        {
            'constraint': 'sdv.constraints.tabular.Between',
            'constraint_column': 'z',
            'high_is_scalar': True,
            'low_is_scalar': True,
            'low': 5,
            'high': 10,
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.Between',
            'constraint_column': 'z',
            'high_is_scalar': False,
            'low_is_scalar': False,
            'low': 'a',
            'high': 'b',
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.Between',
            'constraint_column': 'z',
            'high_is_scalar': True,
            'low_is_scalar': False,
            'low': 'a',
            'high': 10
        },
        {
            'constraint': 'sdv.constraints.tabular.Between',
            'constraint_column': 'z',
            'high_is_scalar': False,
            'low_is_scalar': True,
            'low': 5,
            'high': 'b'
        }
    ]
    old_metadata = {'constraints': old_constraints}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    expected_constraints = [
        {
            'constraint_name': 'ScalarRange',
            'column_name': 'z',
            'low_value': 5,
            'high_value': 10,
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Range',
            'middle_column_name': 'z',
            'low_column_name': 'a',
            'high_column_name': 'b',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Inequality',
            'low_column_name': 'a',
            'high_column_name': 'z',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'z',
            'relation': '<=',
            'value': 10,
        },
        {
            'constraint_name': 'Inequality',
            'low_column_name': 'z',
            'high_column_name': 'b',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'ScalarInequality',
            'column_name': 'z',
            'relation': '>=',
            'value': 5,
        }
    ]
    assert len(expected_constraints) == len(new_constraints)
    for constraint in expected_constraints:
        assert constraint in new_constraints


def test__upgrade_constraints_positive_and_negative():
    """Test the ``_upgrade_constraints`` method with ``Positive`` and ``Negative`` constraints.

    Input:
        - Old metadata dict with the following constraints:
        - ``Positive`` constraint with one column and 'strict' as True.
        - ``Positive`` constraint with one column and 'strict' as False.
        - ``Positive`` constraint with multiple columns and 'strict' as True.
        - ``Positive`` constraint with multiple columns and 'strict' as False.
        - ``Negative`` constraint with one column and 'strict' as True.
        - ``Negative`` constraint with one column and 'strict' as False.
        - ``Negative`` constraint with multiple columns and 'strict' as True.
        - ``Negative`` constraint with multiple columns and 'strict' as False.

    Output:
        - Metadata with the following constraints:
        - ``Positive`` constraint for every ``Positive`` that had strict as True.
        - ``ScalarInequality`` for every ``Positive`` that had strict as False.
        - ``Negative`` constraint for every ``Negative`` that had strict as True.
        - ``ScalarInequality`` for every ``Negative`` that had strict as False.
    """
    # Setup
    old_constraints = [
        {
            'constraint': 'sdv.constraints.tabular.Positive',
            'columns': 'a',
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.Positive',
            'columns': ['b', 'c'],
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.Positive',
            'columns': 'd',
            'strict': False
        },
        {
            'constraint': 'sdv.constraints.tabular.Positive',
            'columns': ['e', 'f'],
            'strict': False
        },
        {
            'constraint': 'sdv.constraints.tabular.Negative',
            'columns': 'a',
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.Negative',
            'columns': ['b', 'c'],
            'strict': True
        },
        {
            'constraint': 'sdv.constraints.tabular.Negative',
            'columns': 'd',
            'strict': False
        },
        {
            'constraint': 'sdv.constraints.tabular.Negative',
            'columns': ['e', 'f'],
            'strict': False
        },
    ]
    old_metadata = {'constraints': old_constraints}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    expected_constraints = [
        {
            'constraint_name': 'Positive',
            'column_name': 'a',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Positive',
            'column_name': 'b',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Positive',
            'column_name': 'c',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Positive',
            'column_name': 'd',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'Positive',
            'column_name': 'e',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'Positive',
            'column_name': 'f',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'Negative',
            'column_name': 'a',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Negative',
            'column_name': 'b',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Negative',
            'column_name': 'c',
            'strict_boundaries': True
        },
        {
            'constraint_name': 'Negative',
            'column_name': 'd',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'Negative',
            'column_name': 'e',
            'strict_boundaries': False
        },
        {
            'constraint_name': 'Negative',
            'column_name': 'f',
            'strict_boundaries': False
        }
    ]
    assert len(expected_constraints) == len(new_constraints)
    for constraint in expected_constraints:
        assert constraint in new_constraints


def test__upgrade_constraints_simple_constraints():
    """Test the ``_upgrade_constraints`` method with constraints that are easy to convert.

    Input:
        - Old metadata with the following constraints:
        - A ``UniqueCombinations`` constraint.
        - A ``OneHotEncoding`` constraint.
        - A ``Unique`` constraint.

    Ouput:
        - A ``FixedCombinations`` constraint.
        - A ``OneHotEncoding`` constraint.
        - A ``Unique`` constraint.
    """
    # Setup
    old_constraints = [
        {
            'constraint': 'sdv.constraints.tabular.UniqueCombinations',
            'columns': ['a', 'b']
        },
        {
            'constraint': 'sdv.constraints.tabular.OneHotEncoding',
            'columns': ['c', 'd']
        },
        {
            'constraint': 'sdv.constraints.tabular.Unique',
            'columns': ['e', 'f']
        },
    ]
    old_metadata = {'constraints': old_constraints}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    expected_constraints = [
        {
            'constraint_name': 'FixedCombinations',
            'column_names': ['a', 'b']
        },
        {
            'constraint_name': 'OneHotEncoding',
            'column_names': ['c', 'd']
        },
        {
            'constraint_name': 'Unique',
            'column_names': ['e', 'f']
        },
    ]
    assert len(expected_constraints) == len(new_constraints)
    for constraint in expected_constraints:
        assert constraint in new_constraints


@patch('sdv.metadata.metadata_upgrader.warnings')
def test__upgrade_constraints_constraint_has_no_upgrade(warnings_mock):
    """Test the ``_upgrade_constraints`` method with constraints that can't be upgraded.

    Setup:
        - Patch warnings.

    Input:
        - Old metadata with following constraints:
        - ``Rounding``.
        - ``ColumnFormula``.
        - ``CustomConstraint``.
        - A fake constraint.

    Output:
        - An empty list.

    Side effect:
        - Warnings should be raised.
    """
    # Setup
    old_constraints = [
        {
            'constraint': 'sdv.constraints.tabular.Rounding'
        },
        {
            'constraint': 'sdv.constraints.tabular.ColumnFormula'
        },
        {
            'constraint': 'sdv.constraints.tabular.CustomConstraint'
        },
        {
            'constraint': 'Fake'
        }
    ]
    old_metadata = {'constraints': old_constraints}

    # Run
    new_constraints = _upgrade_constraints(old_metadata)

    # Assert
    assert new_constraints == []
    assert len(warnings_mock.warn.mock_calls) == 4
    warnings_mock.warn.assert_has_calls([
        call(
            "The SDV no longer supports the 'Rounding' constraint. Most models automatically "
            'round to the same number of decimal digits as the original data.'
        ),
        call(
            'This method does not upgrade ColumnFormula constraints. Please convert your '
            'logic using the new CustomConstraint API.'
        ),
        call(
            'This method does not upgrade CustomConstraint objects. Please convert '
            'your logic using the new CustomConstraint API.'
        ),
        call(
            'Unable to upgrade the Fake constraint. Please add in the constraint '
            'using the new Constraints API.'
        )
    ])


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
                'type': 'categorical',
                'pii': True,
                'pii_category': 'ssn'
            },
            'credit_card': {
                'type': 'categorical',
                'pii': True,
                'pii_category': ['credit_card_number', 'visa']
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
                'computer_representation': 'Int64'
            },
            'duration': {
                'sdtype': 'categorical'
            },
            'student_id': {
                'sdtype': 'id',
                'regex_format': r'\d{30}'
            },
            'high_perc': {
                'sdtype': 'numerical',
                'computer_representation': 'Float'
            },
            'placed': {
                'sdtype': 'boolean'
            },
            'ssn': {
                'sdtype': 'ssn',
                'pii': True
            },
            'credit_card': {
                'sdtype': 'credit_card_number',
                'pii': True
            },
            'drivers_license': {
                'sdtype': 'id',
                'regex_format': 'regex'
            }
        },
        'primary_key': 'student_id',
        'alternate_keys': ['drivers_license']
    }
    assert new_metadata == expected_metadata


@patch('sdv.metadata.metadata_upgrader._upgrade_constraints')
def test_convert_metadata_with_constraints(upgrade_constraints_mock):
    """Test the ``convert_metadata`` method with constraints.

    The method should take a dictionary of the old metadata format and convert it to the new
    format.

    Setup:
        - Mock the ``_upgrade_constraints`` method.

    Input:
        - Dictionary of single table metadata in the old schema and constraints.

    Output:
        - Dictionary of the same metadata with the new schema and constraints.
    """
    # Setup
    old_metadata = {
        'fields': {
            'salary': {
                'type': 'numerical',
                'subtype': 'integer'
            },
            'student_id': {
                'type': 'id',
                'subtype': 'integer'
            },
        },
        'primary_key': 'student_id',
        'constraints': [
            {
                'constraint': 'sdv.constraints.tabular.UniqueCombinations',
                'columns': ['a', 'b']
            },
            {
                'constraint': 'sdv.constraints.tabular.OneHotEncoding',
                'columns': ['c', 'd']
            }
        ]
    }
    new_constraints = [
        {
            'constraint_name': 'FixedCombinations',
            'column_names': ['a', 'b']
        },
        {
            'constraint_name': 'OneHotEncoding',
            'column_names': ['c', 'd']
        }
    ]
    upgrade_constraints_mock.return_value = new_constraints

    # Run
    new_metadata = convert_metadata(old_metadata)

    # Assert
    expected_metadata = {
        'columns': {
            'salary': {
                'sdtype': 'numerical',
                'computer_representation': 'Int64'
            },
            'student_id': {
                'sdtype': 'id',
                'regex_format': r'\d{30}'
            }
        },
        'primary_key': 'student_id'
    }
    assert new_metadata == expected_metadata
