import re

import numpy as np
import pandas as pd
import pytest

from sdv.constraints import (
    FixedCombinations, FixedIncrements, Inequality, Negative, OneHotEncoding, Positive, Range,
    ScalarInequality, ScalarRange, Unique)
from sdv.constraints.errors import MultipleConstraintsErrors
from sdv.constraints.tabular import create_custom_constraint
from sdv.demo import load_tabular_demo
from sdv.sampling import Condition
from sdv.tabular import GaussianCopula


def test_constraints(tmpdir):
    # Setup
    employees = load_tabular_demo()
    fixed_company_department_constraint = FixedCombinations(column_names=['company', 'department'])
    age_gt_age_when_joined_constraint = Inequality(
        low_column_name='age_when_joined',
        high_column_name='age'
    )
    age_range_constraint = ScalarRange('age', 29, 50)
    constraints = [
        fixed_company_department_constraint,
        age_gt_age_when_joined_constraint,
        age_range_constraint
    ]

    # Run
    gc = GaussianCopula(constraints=constraints, min_value=None, max_value=None)
    gc.fit(employees)
    gc.save(tmpdir / 'test.pkl')
    gc = gc.load(tmpdir / 'test.pkl')
    sampled = gc.sample(10)

    # Assert
    assert all(age_gt_age_when_joined_constraint.is_valid(sampled))
    assert all(age_range_constraint.is_valid(sampled))
    assert all(fixed_company_department_constraint.is_valid(sampled))


def test_constraints_with_conditions():
    # Setup
    data = pd.DataFrame(data={
        'low_col': [i for i in range(50)],
        'mid_col': [i + 1 for i in range(50)],
        'high_col': [i + 2 for i in range(50)]
    })

    low_lt_mid_constraint = Inequality(
        low_column_name='low_col',
        high_column_name='mid_col'
    )
    mid_lt_high_constraint = Inequality(
        low_column_name='mid_col',
        high_column_name='high_col'
    )

    # Run
    model = GaussianCopula(constraints=[low_lt_mid_constraint, mid_lt_high_constraint])
    model.fit(data)

    my_condition = Condition(column_values={'low_col': 1, 'mid_col': 2}, num_rows=2)
    sampled = model.sample_conditions(conditions=[my_condition])

    # Assert
    assert all(low_lt_mid_constraint.is_valid(sampled))
    assert all(mid_lt_high_constraint.is_valid(sampled))


def test_failing_constraints():
    data = pd.DataFrame({
        'a': [0, 0, 0, 0, 0, 0, 0],
        'b': [1, -1, 2, -2, 3, -3, 0],
        'c': [-1, -1, -1, -1, -1, -1, -1],
        'd': [1, -1, 2, -2, 3, -3, 5],
        'e': [1, 2, 3, 4, 5, 6, 'a'],
        'f': [1, 1, 2, 2, 3, 3, -1],
        'g': [1, 0, 1, 0, 0, 1, 0],
        'h': [1, 1, 1, 0, 0, 10, 0],
        'i': [1, 1, 1, 1, 1, 1, 1],
        'j': [2, 3, 4, 5, 6, 7, 5.5],
        'k': [1, -1, 2, -2, 3, -3, 5],
        'l': [25, 50, 70, np.nan, 50, 25, 25],
    })

    custom_constraint = create_custom_constraint(
        lambda _, x: pd.Series([True if x_i > 0 else False for x_i in x['k']])
    )
    constraints = [
        Inequality('a', 'b'),
        Positive('c'),
        Negative('d'),
        OneHotEncoding(['g', 'h']),
        Unique(['i']),
        ScalarInequality('j', '>=', 5.5),
        Range('a', 'b', 'c'),
        ScalarRange('a', 0, 0),
        custom_constraint('k'),
        FixedIncrements(column_name='l', increment_value=25),
    ]
    gc = GaussianCopula(constraints=constraints)

    err_msg = re.escape(
        "Data is not valid for the 'Inequality' constraint:"
        '\n   a  b'
        '\n1  0 -1'
        '\n3  0 -2'
        '\n5  0 -3'
        '\n'
        "\nData is not valid for the 'Positive' constraint:"
        '\n   c'
        '\n0 -1'
        '\n1 -1'
        '\n2 -1'
        '\n3 -1'
        '\n4 -1'
        '\n+2 more'
        '\n'
        "\nData is not valid for the 'Negative' constraint:"
        '\n   d'
        '\n0  1'
        '\n2  2'
        '\n4  3'
        '\n6  5'
        '\n'
        "\nData is not valid for the 'OneHotEncoding' constraint:"
        '\n   g   h'
        '\n0  1   1'
        '\n2  1   1'
        '\n3  0   0'
        '\n4  0   0'
        '\n5  1  10'
        '\n+1 more'
        '\n'
        "\nData is not valid for the 'Unique' constraint:"
        '\n   i'
        '\n1  1'
        '\n2  1'
        '\n3  1'
        '\n4  1'
        '\n5  1'
        '\n+1 more'
        '\n'
        "\nData is not valid for the 'ScalarInequality' constraint:"
        '\n     j'
        '\n0  2.0'
        '\n1  3.0'
        '\n2  4.0'
        '\n3  5.0'
        '\n'
        "\nData is not valid for the 'Range' constraint:"
        '\n   a  b  c'
        '\n0  0  1 -1'
        '\n1  0 -1 -1'
        '\n2  0  2 -1'
        '\n3  0 -2 -1'
        '\n4  0  3 -1'
        '\n+2 more'
        '\n'
        "\nData is not valid for the 'ScalarRange' constraint:"
        '\n   a'
        '\n0  0'
        '\n1  0'
        '\n2  0'
        '\n3  0'
        '\n4  0'
        '\n+2 more'
        '\n'
        "\nData is not valid for the 'CustomConstraint' constraint:"
        '\n   k'
        '\n1 -1'
        '\n3 -2'
        '\n5 -3'
        '\n'
        "\nData is not valid for the 'FixedIncrements' constraint:"
        '\n      l'
        '\n2  70.0'
    )

    with pytest.raises(MultipleConstraintsErrors, match=err_msg):
        gc.fit(data)
