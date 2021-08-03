import pandas as pd

from sdv.constraints import ColumnFormula, CustomConstraint, GreaterThan, UniqueCombinations
from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula


def years_in_the_company(data):
    return data['age'] - data['age_when_joined']


def test_constraints(tmpdir):

    employees = load_tabular_demo()

    unique_company_department_constraint = UniqueCombinations(
        columns=['company', 'department'],
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
        unique_company_department_constraint,
        age_gt_age_when_joined_constraint,
        years_in_the_company_constraint
    ]
    gc = GaussianCopula(constraints=constraints)
    gc.fit(employees)
    gc.save(tmpdir / 'test.pkl')
    gc = gc.load(tmpdir / 'test.pkl')
    gc.sample(10)


_IS_VALID_CALLED = []


def _is_valid(rows):
    if not _IS_VALID_CALLED:
        _IS_VALID_CALLED.append(True)
        return pd.Series([False] * len(rows), index=rows.index)

    return pd.Series([True] * len(rows), index=rows.index)


def test_constraints_reject_sampling_zero_valid():
    """Ensure everything works if no rows are valid on the first try.

    See https://github.com/sdv-dev/SDV/issues/285
    """
    employees = load_tabular_demo()

    _IS_VALID_CALLED.clear()
    constraint = CustomConstraint(is_valid=_is_valid)

    gc = GaussianCopula(constraints=[constraint])
    gc.fit(employees)
    gc.sample(10)


def test_constraints_condition_on_constraint_column():
    """Test the CustomFormula constraint when conditioning on the constraint column.

    Expect conditioning on the constraint column to work with CustomFormula.

    Setup:
        - Load employee data
        - Setup CustomFormula on `years_in_the_company` column
        - Create and fit the model
        - Sample, conditioning on `years_in_the_company` column
    Side effects:
        - Expect no errors
    """
    employees = load_tabular_demo()

    years_in_the_company_constraint = ColumnFormula(
        column='years_in_the_company',
        formula=years_in_the_company,
        handling_strategy='transform'
    )

    constraints = [years_in_the_company_constraint]
    gc = GaussianCopula(constraints=constraints)
    gc.fit(employees)
    gc.sample(10, conditions={'years_in_the_company': 1})
