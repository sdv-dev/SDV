import inspect

import sdv.cag


def test_all_available_constraints_included_in_constraint_test_list(constraints_as_dicts):
    """Test that all available constraints are included in the test list."""
    # Setup
    skipped_cag_module_classes = [
        'ProgrammableConstraint',
        'SingleTableProgrammableConstraint',
        'ConstraintList',
    ]
    available_constraints = inspect.getmembers(sdv.cag, lambda x: inspect.isclass(x))

    available_constraints = {
        constraint
        for constraint, cls in available_constraints
        if constraint not in skipped_cag_module_classes
    }
    tested_constraints = {constraint[0] for constraint in constraints_as_dicts}

    # Run and Assert
    assert available_constraints == tested_constraints


def test_get_constraint_dict_and_load_constraint_from_dict(constraint_tuple):
    """Test ``get_constraint_dict`` and ``load_constraint_from_dict for all constraints."""
    # Setup
    constraint_class_name, constraint_params = constraint_tuple

    constraint_class = getattr(sdv.cag, constraint_class_name)
    constraint_instance = constraint_class(**constraint_params)

    # Run
    constraint_dict = constraint_instance.get_constraint_dict()
    loaded_constraint = constraint_class.load_constraint_from_dict(constraint_dict['parameters'])

    # Assert
    assert constraint_dict == {'class_name': constraint_class_name, 'parameters': constraint_params}
    for param, param_value in constraint_params.items():
        instanced_param = getattr(
            constraint_instance, param, getattr(constraint_instance, f'_{param}', None)
        )
        loaded_param = getattr(
            loaded_constraint, param, getattr(loaded_constraint, f'_{param}', None)
        )
        assert instanced_param == loaded_param == param_value
