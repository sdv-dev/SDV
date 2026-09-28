from copy import deepcopy

import pytest

import sdv.cag

CONSTRAINTS = [
    (
        'FixedCombinations',
        {
            'column_names': ['A', 'B', 'C'],
            'table_name': 'table',
        },
    ),
    (
        'FixedIncrements',
        {
            'column_name': 'column',
            'increment_value': 5,
            'table_name': 'table',
        },
    ),
    (
        'Inequality',
        {
            'low_column_name': 'low',
            'high_column_name': 'high',
            'strict_boundaries': True,
            'table_name': None,
        },
    ),
    (
        'OneHotEncoding',
        {
            'column_names': ['col1', 'col2', 'col3', 'col4', 'col5'],
            'table_name': 'table',
            'learning_strategy': 'one_hot',
        },
    ),
    (
        'Range',
        {
            'low_column_name': 'low',
            'middle_column_name': 'middle',
            'high_column_name': 'high',
            'strict_boundaries': False,
            'table_name': None,
        },
    ),
]


@pytest.fixture()
def constraints_as_dicts():
    """Return a list of dictionaries (each dictionary is a constraint definition)."""
    return CONSTRAINTS


@pytest.fixture(params=CONSTRAINTS, ids=[c_name for c_name, _ in CONSTRAINTS])
def constraint_tuple(request):
    """Return a constraint class name and constraint parameters as a tuple."""
    constraint_class_name, constraint_params = request.param

    return constraint_class_name, deepcopy(constraint_params)


@pytest.fixture()
def constraint_object(constraint_tuple):
    """Return a constraint object."""
    constraint_class_name, constraint_params = constraint_tuple
    constraint_class = getattr(sdv.cag, constraint_class_name)

    return constraint_class(**constraint_params)
