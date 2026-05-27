import inspect

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


def test_all_available_constraints_included_in_constraint_test_list():
    """Test that all available constraints are included in the test list."""
    # Setup
    skipped_cag_module_classes = ['ProgrammableConstraint', 'ConstraintList']
    available_constraints = inspect.getmembers(sdv.cag, lambda x: inspect.isclass(x))

    available_constraints = {
        constraint
        for constraint, cls in available_constraints
        if constraint not in skipped_cag_module_classes
    }
    tested_constraints = {constraint[0] for constraint in CONSTRAINTS}

    # Run and Assert
    assert available_constraints == tested_constraints


@pytest.mark.parametrize(['constraint', 'constraint_params'], CONSTRAINTS)
def test_get_constraint_dict_and_load_constraint_from_dict(constraint, constraint_params):
    """Test ``get_constraint_dict`` and ``load_constraint_from_dict for all constraints."""
    # Setup
    constraint_class = getattr(sdv.cag, constraint)
    constraint_instance = constraint_class(**constraint_params)

    # Run
    constraint_dict = constraint_instance.get_constraint_dict()
    loaded_constraint = constraint_class.load_constraint_from_dict(constraint_dict['parameters'])

    # Assert
    assert constraint_dict == {'class_name': constraint, 'parameters': constraint_params}
    for param, param_value in constraint_params.items():
        instanced_param = getattr(
            constraint_instance, param, getattr(constraint_instance, f'_{param}', None)
        )
        loaded_param = getattr(
            loaded_constraint, param, getattr(loaded_constraint, f'_{param}', None)
        )
        assert instanced_param == loaded_param == param_value
