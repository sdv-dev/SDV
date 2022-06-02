import re

import pandas as pd
import pytest

from sdv.constraints import (
    Between, ColumnFormula, FixedCombinations, GreaterThan, Negative, OneHotEncoding, Positive,
    Unique)
from sdv.constraints.errors import MultipleConstraintsErrors
from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula


def years_in_the_company(data):
    return data['age'] - data['age_when_joined']


def test_constraints(tmpdir):

    employees = load_tabular_demo()

    fixed_company_department_constraint = FixedCombinations(
        column_names=['company', 'department'],
        handling_strategy='transform',
        fit_columns_model=False
    )

    age_gt_age_when_joined_constraint = GreaterThan(
        low='age_when_joined',
        high='age',
        handling_strategy='reject_sampling',
        fit_columns_model=False
    )

    years_in_the_company_constraint = ColumnFormula(
        column='years_in_the_company',
        formula=years_in_the_company,
        handling_strategy='transform'
    )

    constraints = [
        fixed_company_department_constraint,
        age_gt_age_when_joined_constraint,
        years_in_the_company_constraint
    ]
    gc = GaussianCopula(constraints=constraints)
    gc.fit(employees)
    gc.save(tmpdir / 'test.pkl')
    gc = gc.load(tmpdir / 'test.pkl')
    gc.sample(10)


def test_failing_constraints():
    data = pd.DataFrame({
        'a': [0, 0, 0, 0, 0, 0, 0],
        'b': [1, -1, 2, -2, 3, -3, 5],
        'c': [-1, -1, -1, -1, -1, -1, -1],
        'd': [1, -1, 2, -2, 3, -3, 5],
        'e': [1, 2, 3, 4, 5, 6, 'a'],
        'f': [1, 1, 2, 2, 3, 3, -1],
        'g': [1, 0, 1, 0, 0, 1, 0],
        'h': [1, 1, 1, 0, 0, 10, 0],
        'i': [1, 1, 1, 1, 1, 1, 1]
    })

    constraints = [
        GreaterThan('a', 'b'),
        Positive('c'),
        Negative('d'),
        Between('f', 0, 3),
        OneHotEncoding(['g', 'h']),
        Unique('i')
    ]
    gc = GaussianCopula(constraints=constraints)

    err_msg = re.escape(
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
        "\nData is not valid for the 'GreaterThan' constraint:"
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
        "\nData is not valid for the 'Between' constraint:"
        '\n   f'
        '\n6 -1'
    )

    with pytest.raises(MultipleConstraintsErrors, match=err_msg):
        gc.fit(data)
